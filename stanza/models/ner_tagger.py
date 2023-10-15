"""
Entry point for training and evaluating an NER tagger.

This tagger uses BiLSTM layers with character and word-level representations, and a CRF decoding layer 
to produce NER predictions.
For details please refer to paper: https://nlp.stanford.edu/pubs/qi2018universal.pdf.
"""

import sys
import os
import time
from datetime import datetime
import argparse
import logging
import numpy as np
import random
import json
import torch
from torch import nn, optim

from stanza.models.ner.data import DataLoader
from stanza.models.ner.trainer import Trainer
from stanza.models.ner import scorer
from stanza.models.common import utils
from stanza.models.common.pretrain import Pretrain
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import *
from stanza.models import _training_logging

from stanza.utils.confusion import confusion_to_weighted_f1, format_confusion

logger = logging.getLogger('stanza')

def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/ner', help='Directory of NER data.')
    parser.add_argument('--wordvec_dir', type=str, default='extern_data/word2vec', help='Directory of word vectors')
    parser.add_argument('--wordvec_file', type=str, default='', help='File that contains word vectors')
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')
    parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--eval_output_file', type=str, default=None, help='Where to write results: text, gold, pred.  If None, no results file printed')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--finetune', action='store_true', help='Load existing model during `train` mode from `save_dir` path')
    parser.add_argument('--finetune_load_name', type=str, default=None, help='Model to load when finetuning')
    parser.add_argument('--train_classifier_only', action='store_true',
                        help='In case of applying Transfer-learning approach and training only the classifier layer this will freeze gradient propagation for all other layers.')
    parser.add_argument('--shorthand', type=str, help="Treebank shorthand")

    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--char_hidden_dim', type=int, default=100)
    parser.add_argument('--word_emb_dim', type=int, default=100)
    parser.add_argument('--char_emb_dim', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--pretrain_max_vocab', type=int, default=100000)
    parser.add_argument('--word_dropout', type=float, default=0.01, help="How often to remove a word at training time.  Set to a small value to train unk when finetuning word embeddings")
    parser.add_argument('--locked_dropout', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--rec_dropout', type=float, default=0, help="Word recurrent dropout")
    parser.add_argument('--char_rec_dropout', type=float, default=0, help="Character recurrent dropout")
    parser.add_argument('--char_dropout', type=float, default=0, help="Character-level language model dropout")
    parser.add_argument('--no_char', dest='char', action='store_false', help="Turn off training a character model.")
    parser.add_argument('--charlm', action='store_true', help="Turn on contextualized char embedding using pretrained character-level language model.")
    parser.add_argument('--charlm_save_dir', type=str, default='saved_models/charlm', help="Root dir for pretrained character-level language model.")
    parser.add_argument('--charlm_shorthand', type=str, default=None, help="Shorthand for character-level language model training corpus.")
    parser.add_argument('--charlm_forward_file', type=str, default=None, help="Exact path to use for forward charlm")
    parser.add_argument('--charlm_backward_file', type=str, default=None, help="Exact path to use for backward charlm")
    parser.add_argument('--char_lowercase', dest='char_lowercase', action='store_true', help="Use lowercased characters in character model.")
    parser.add_argument('--no_lowercase', dest='lowercase', action='store_false', help="Use cased word vectors.")
    parser.add_argument('--no_emb_finetune', dest='emb_finetune', action='store_false', help="Turn off finetuning of the embedding matrix.")
    parser.add_argument('--emb_finetune_known_only', dest='emb_finetune_known_only', action='store_true', help="Finetune the embedding matrix only for words in the embedding.  (Default: finetune words not in the embedding as well)  This may be useful for very large datasets where obscure words are only trained once in a while, such as French-WikiNER")
    parser.add_argument('--no_input_transform', dest='input_transform', action='store_false', help="Do not use input transformation layer before tagger lstm.")
    parser.add_argument('--scheme', type=str, default='bioes', help="The tagging scheme to use: bio or bioes.")


    parser.add_argument('--bert_model', type=str, default=None, help="Use an external bert model (requires the transformers package)")
    parser.add_argument('--no_bert_model', dest='bert_model', action="store_const", const=None, help="Don't use bert")
    parser.add_argument('--bert_finetune', default=False, action='store_true', help='Finetune the bert (or other transformer)')
    parser.add_argument('--no_bert_finetune', dest='bert_finetune', action='store_false', help="Don't finetune the bert (or other transformer)")
    parser.add_argument('--bert_learning_rate', default=1.0, type=float, help='Scale the learning rate for transformer finetuning by this much')

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='sgd', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='Minimum learning rate to stop training.')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum for SGD.')
    parser.add_argument('--lr_decay', type=float, default=0.5, help="LR decay rate.")
    parser.add_argument('--patience', type=int, default=3, help="Patience for LR decay.")

    parser.add_argument('--connect_output_layers', action='store_true', default=False, help='Connect one output layer to the input of the next output layer.  By default, those layers are all separate')
    parser.add_argument('--predict_tagset', type=int, default=None, help='Which tagset to predict if there are multiple tagsets.  Will default to 0.  Default of None allows the model to remember the value from training time, but be overridden at test time')

    parser.add_argument('--ignore_tag_scores', type=str, default=None, help="Which tags to ignore, if any, when scoring dev & test sets")

    parser.add_argument('--max_steps', type=int, default=200000)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--log_norms', action='store_true', default=False, help='Log the norms of all the parameters (noisy!)')
    parser.add_argument('--save_dir', type=str, default='saved_models/ner', help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default="{shorthand}_{embedding}_{finetune}_nertagger.pt", help="File name to save the model")

    parser.add_argument('--seed', type=int, default=1234)
    utils.add_device_args(parser)

    parser.add_argument('--wandb', action='store_true', help='Start a wandb session and write the results of training.  Only applies to training.  Use --wandb_name instead to specify a name')
    parser.add_argument('--wandb_name', default=None, help='Name of a wandb session to start when training.  Will default to the dataset short name')
    return parser

def parse_args(args=None):
    parser = build_argparse()
    args = parser.parse_args(args=args)

    if args.wandb_name:
        args.wandb = True

    args = vars(args)
    return args

def main(args=None):
    args = parse_args(args=args)

    utils.set_random_seed(args['seed'])

    logger.info("Running NER tagger in {} mode".format(args['mode']))

    if args['mode'] == 'train':
        return train(args)
    else:
        evaluate(args)

def load_pretrain(args):
    # load pretrained vectors
    if args['wordvec_pretrain_file']:
        pretrain_file = args['wordvec_pretrain_file']
        pretrain = Pretrain(pretrain_file, None, args['pretrain_max_vocab'], save_to_file=False)
    else:
        if len(args['wordvec_file']) == 0:
            vec_file = utils.get_wordvec_file(args['wordvec_dir'], args['shorthand'])
        else:
            vec_file = args['wordvec_file']
        # do not save pretrained embeddings individually
        pretrain = Pretrain(None, vec_file, args['pretrain_max_vocab'], save_to_file=False)
    return pretrain

def model_file_name(args):
    return utils.standard_model_file_name(args, "nertagger")

def get_known_tags(tags):
    """
    Tags are stored in the dataset as a list of list of tags

    This returns a sorted list for each column of tags in the dataset
    """
    max_columns = max(len(word) for sent in tags for word in sent)
    known_tags = [set() for _ in range(max_columns)]
    for sent in tags:
        for word in sent:
            for tag_idx, tag in enumerate(word):
                known_tags[tag_idx].add(tag)
    return [sorted(x) for x in known_tags]

def warn_missing_tags(tag_vocab, data_tags, error_msg):
    """
    Check for tags missing from the tag_vocab.

    Given a tag_vocab and the known tags in the format used by
    ner.data, go through the tags in the dataset and look for any
    which aren't in the tag_vocab.

    error_msg is something like "training set" or "eval file" to
    indicate where the missing tags came from.
    """
    tag_depth = max(max(len(tags) for tags in sentence) for sentence in data_tags)

    if tag_depth != len(tag_vocab.lens()):
        logger.warning("Test dataset has a different number of tag types compared to the model: %d vs %d", tag_depth, len(tag_vocab.lens()))
    for tag_set_idx in range(min(tag_depth, len(tag_vocab.lens()))):
        tag_set = tag_vocab.items(tag_set_idx)
        if len(tag_vocab.lens()) > 1:
            current_error_msg = error_msg + " tag set %d" % tag_set_idx
        else:
            current_error_msg = error_msg

        current_tags = set([word[tag_set_idx] for sentence in data_tags for word in sentence])
        utils.warn_missing_tags(tag_set, current_tags, current_error_msg)

def train(args):
    model_file = model_file_name(args)

    save_dir, save_name = os.path.split(model_file)
    utils.ensure_dir(save_dir)
    if args['save_dir'] is None:
        args['save_dir'] = save_dir
    args['save_name'] = save_name

    utils.log_training_args(args, logger)

    pretrain = None
    vocab = None
    trainer = None

    if args['finetune'] and args['finetune_load_name']:
        logger.warning('Finetune is ON. Using model from "{}"'.format(args['finetune_load_name']))
        _, trainer, vocab = load_model(args, args['finetune_load_name'])
    elif args['finetune'] and os.path.exists(model_file):
        logger.warning('Finetune is ON. Using model from "{}"'.format(model_file))
        _, trainer, vocab = load_model(args, model_file)
    else:
        if args['finetune']:
            raise FileNotFoundError('Finetune is set to true but model file is not found: {}'.format(model_file))

        pretrain = load_pretrain(args)

        if pretrain is not None:
            word_emb_dim = pretrain.emb.shape[1]
            if args['word_emb_dim'] and args['word_emb_dim'] != word_emb_dim:
                logger.warning("Embedding file has a dimension of {}.  Model will be built with that size instead of {}".format(word_emb_dim, args['word_emb_dim']))
            args['word_emb_dim'] = word_emb_dim

        if args['charlm']:
            if args['charlm_shorthand'] is None:
                raise ValueError("CharLM Shorthand is required for loading pretrained CharLM model...")
            logger.info('Using pretrained contextualized char embedding')
            if not args['charlm_forward_file']:
                args['charlm_forward_file'] = '{}/{}_forward_charlm.pt'.format(args['charlm_save_dir'], args['charlm_shorthand'])
            if not args['charlm_backward_file']:
                args['charlm_backward_file'] = '{}/{}_backward_charlm.pt'.format(args['charlm_save_dir'], args['charlm_shorthand'])

    # load data
    logger.info("Loading training data with batch size %d from %s", args['batch_size'], args['train_file'])
    with open(args['train_file']) as fin:
        train_doc = Document(json.load(fin))
    logger.info("Loaded %d sentences of training data", len(train_doc.sentences))
    if len(train_doc.sentences) == 0:
        raise ValueError("File %s exists but has no usable training data" % args['train_file'])
    train_batch = DataLoader(train_doc, args['batch_size'], args, pretrain, vocab=vocab, evaluation=False)
    vocab = train_batch.vocab
    logger.info("Loading dev data from %s", args['eval_file'])
    with open(args['eval_file']) as fin:
        dev_doc = Document(json.load(fin))
    logger.info("Loaded %d sentences of dev data", len(dev_doc.sentences))
    if len(dev_doc.sentences) == 0:
        raise ValueError("File %s exists but has no usable dev data" % args['train_file'])
    dev_batch = DataLoader(dev_doc, args['batch_size'], args, pretrain, vocab=vocab, evaluation=True)

    train_tags = get_known_tags(train_batch.tags)
    logger.info("Training data has %d columns of tags", len(train_tags))
    for tag_idx, tags in enumerate(train_tags):
        logger.info("Tags present in training set at column %d:\n  Tags without BIES markers: %s\n  Tags with B-, I-, E-, or S-: %s",
                    tag_idx,
                    " ".join(sorted(set(i for i in tags if i[:2] not in ('B-', 'I-', 'E-', 'S-')))),
                    " ".join(sorted(set(i[2:] for i in tags if i[:2] in ('B-', 'I-', 'E-', 'S-')))))

    # skip training if the language does not have training or dev data
    if len(train_batch) == 0 or len(dev_batch) == 0:
        logger.info("Skip training because no data available...")
        return

    logger.info("Training tagger...")
    if trainer is None: # init if model was not loaded previously from file
        trainer = Trainer(args=args, vocab=vocab, pretrain=pretrain, device=args['device'],
                          train_classifier_only=args['train_classifier_only'])

    if args['finetune']:
        warn_missing_tags(trainer.vocab['tag'], train_batch.tags, "training set")
    warn_missing_tags(trainer.vocab['tag'], dev_batch.tags, "dev set")

    # TODO: might still want to add multiple layers of tag evaluation to the scorer
    dev_gold_tags = [[x[trainer.args['predict_tagset']] for x in tags] for tags in dev_batch.tags]

    logger.info(trainer.model)

    global_step = 0
    max_steps = args['max_steps']
    dev_score_history = []
    best_dev_preds = []
    current_lr = trainer.optimizer.param_groups[0]['lr']
    global_start_time = time.time()
    format_str = '{}: step {}/{}, loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'

    # LR scheduling
    if args['lr_decay'] > 0:
        # learning rate changes on plateau -- no improvement on model for patience number of epochs
        # change is made as a factor of the learning rate decay
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, mode='max', factor=args['lr_decay'],
                                                               patience=args['patience'], verbose=True, min_lr=args['min_lr'])
    else:
        scheduler = None

    if args['wandb']:
        import wandb
        wandb_name = args['wandb_name'] if args['wandb_name'] else "%s_ner" % args['shorthand']
        wandb.init(name=wandb_name, config=args)
        wandb.run.define_metric('train_loss', summary='min')
        wandb.run.define_metric('dev_score', summary='max')

    # start training
    train_loss = 0
    while True:
        should_stop = False
        for i, batch in enumerate(train_batch):
            start_time = time.time()
            global_step += 1
            loss = trainer.update(batch, eval=False) # update step
            train_loss += loss
            if global_step % args['log_step'] == 0:
                duration = time.time() - start_time
                logger.info(format_str.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), global_step,
                                              max_steps, loss, duration, current_lr))
                if args['log_norms']:
                    trainer.model.log_norms()

            if global_step % args['eval_interval'] == 0:
                # eval on dev
                logger.info("Evaluating on dev set...")
                dev_preds = []
                for batch in dev_batch:
                    preds = trainer.predict(batch)
                    dev_preds += preds
                _, _, dev_score = scorer.score_by_entity(dev_preds, dev_gold_tags, ignore_tags=args['ignore_tag_scores'])

                train_loss = train_loss / args['eval_interval'] # avg loss per batch
                logger.info("step {}: train_loss = {:.6f}, dev_score = {:.4f}".format(global_step, train_loss, dev_score))
                if args['wandb']:
                    wandb.log({'train_loss': train_loss, 'dev_score': dev_score})
                train_loss = 0

                # save best model
                if len(dev_score_history) == 0 or dev_score > max(dev_score_history):
                    trainer.save(model_file)
                    logger.info("New best model saved.")
                    best_dev_preds = dev_preds

                dev_score_history += [dev_score]
                logger.info("")

                # lr schedule
                if scheduler is not None:
                    scheduler.step(dev_score)
            
            # check stopping
            current_lr = trainer.optimizer.param_groups[0]['lr']
            if global_step >= args['max_steps'] or current_lr <= args['min_lr']:
                should_stop = True
                break

        if should_stop:
            break

        train_batch.reshuffle()

    logger.info("Training ended with {} steps.".format(global_step))

    if args['wandb']:
        wandb.finish()

    if len(dev_score_history) > 0:
        best_f, best_eval = max(dev_score_history)*100, np.argmax(dev_score_history)+1
        logger.info("Best dev F1 = {:.2f}, at iteration = {}".format(best_f, best_eval * args['eval_interval']))
    else:
        logger.info("Dev set never evaluated.  Saving final model.")
        trainer.save(model_file)

    return trainer

def write_ner_results(filename, batch, preds, predict_tagset):
    if len(batch.tags) != len(preds):
        raise ValueError("Unexpected batch vs pred lengths: %d vs %d" % (len(batch.tags), len(preds)))

    with open(filename, "w", encoding="utf-8") as fout:
        tag_idx = 0
        for b in batch:
            # b[0] is words, b[5] is orig_idx
            # a namedtuple would make this cleaner without being much slower
            text = utils.unsort(b[0], b[5])
            for sentence in text:
                # TODO: if we change the predict_tagset mechanism, will have to change this
                sentence_gold = [x[predict_tagset] for x in batch.tags[tag_idx]]
                sentence_pred = preds[tag_idx]
                tag_idx += 1
                for word, gold, pred in zip(sentence, sentence_gold, sentence_pred):
                    fout.write("%s\t%s\t%s\n" % (word, gold, pred))
                fout.write("\n")

def evaluate(args):
    # file paths
    model_file = model_file_name(args)

    loaded_args, trainer, vocab = load_model(args, model_file)
    logger.debug("Loaded model for eval from %s", model_file)
    logger.debug("Using the %d tagset for evaluation", loaded_args['predict_tagset'])

    # load data
    logger.info("Loading data with batch size {}...".format(args['batch_size']))
    with open(args['eval_file']) as fin:
        doc = Document(json.load(fin))
    batch = DataLoader(doc, args['batch_size'], loaded_args, vocab=vocab, evaluation=True, bert_tokenizer=trainer.model.bert_tokenizer)
    warn_missing_tags(trainer.vocab['tag'], batch.tags, "eval_file")

    logger.info("Start evaluation...")
    preds = []
    for i, b in enumerate(batch):
        preds += trainer.predict(b)

    gold_tags = batch.tags
    # TODO: might still want to add multiple layers of tag evaluation to the scorer
    gold_tags = [[x[trainer.args['predict_tagset']] for x in tags] for tags in gold_tags]

    _, _, score = scorer.score_by_entity(preds, gold_tags, ignore_tags=args['ignore_tag_scores'])
    _, _, _, confusion = scorer.score_by_token(preds, gold_tags, ignore_tags=args['ignore_tag_scores'])
    logger.info("Weighted f1 for non-O tokens: %5f", confusion_to_weighted_f1(confusion, exclude=["O"]))

    logger.info("NER tagger score: %s %s %s %.2f", args['shorthand'], model_file, args['eval_file'], score*100)
    logger.info("NER token confusion matrix:\n{}".format(format_confusion(confusion)))

    if args['eval_output_file']:
        write_ner_results(args['eval_output_file'], batch, preds, trainer.args['predict_tagset'])

    return confusion

def load_model(args, model_file):
    # load model
    charlm_args = {}
    if 'charlm_forward_file' in args:
        charlm_args['charlm_forward_file'] = args['charlm_forward_file']
    if 'charlm_backward_file' in args:
        charlm_args['charlm_backward_file'] = args['charlm_backward_file']
    if args['predict_tagset'] is not None:
        charlm_args['predict_tagset'] = args['predict_tagset']
    pretrain = load_pretrain(args)
    trainer = Trainer(args=charlm_args, model_file=model_file, pretrain=pretrain, device=args['device'], train_classifier_only=args['train_classifier_only'])
    loaded_args, vocab = trainer.args, trainer.vocab

    # load config
    for k in args:
        if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand', 'mode', 'scheme']:
            loaded_args[k] = args[k]
    return loaded_args, trainer, vocab


if __name__ == '__main__':
    main()
