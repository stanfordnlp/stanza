"""
Entry point for training and evaluating a POS/morphological features tagger.

This tagger uses highway BiLSTM layers with character and word-level representations, and biaffine classifiers
to produce consistent POS and UFeats predictions.
For details please refer to paper: https://nlp.stanford.edu/pubs/qi2018universal.pdf.
"""

import argparse
import logging
import os
import time
import zipfile

import numpy as np
import torch
from torch import nn, optim

from stanza.models.pos.data import Dataset, ShuffledDataset
from stanza.models.pos.trainer import Trainer
from stanza.models.pos import scorer
from stanza.models.common import utils
from stanza.models.common import pretrain
from stanza.models.common.doc import *
from stanza.models.common.foundation_cache import FoundationCache
from stanza.models.common.peft_config import add_peft_args, resolve_peft_args
from stanza.models import _training_logging
from stanza.utils.conll import CoNLL

logger = logging.getLogger('stanza')

def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/pos', help='Root dir for saving models.')
    parser.add_argument('--wordvec_dir', type=str, default='extern_data/wordvec', help='Directory of word vectors.')
    parser.add_argument('--wordvec_file', type=str, default=None, help='Word vectors filename.')
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')
    parser.add_argument('--train_file', type=str, default=None, help='Input file for training.')
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for scoring.')
    parser.add_argument('--output_file', type=str, default=None, help='Output CoNLL-U file.')
    parser.add_argument('--no_gold_labels', dest='gold_labels', action='store_false', help="Don't score the eval file - perhaps it has no gold labels, for example.  Cannot be used at training time")

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')
    parser.add_argument('--shorthand', type=str, help="Treebank shorthand")

    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--char_hidden_dim', type=int, default=400)
    parser.add_argument('--deep_biaff_hidden_dim', type=int, default=400)
    parser.add_argument('--composite_deep_biaff_hidden_dim', type=int, default=100)
    parser.add_argument('--word_emb_dim', type=int, default=75, help='Dimension of the finetuned word embedding.  Set to 0 to turn off')
    parser.add_argument('--word_cutoff', type=int, default=7, help='How common a word must be to include it in the finetuned word embedding')
    parser.add_argument('--char_emb_dim', type=int, default=100)
    parser.add_argument('--tag_emb_dim', type=int, default=50)
    parser.add_argument('--charlm_transform_dim', type=int, default=None, help='Transform the pretrained charlm to this dimension.  If not set, no transform is used')
    parser.add_argument('--transformed_dim', type=int, default=125)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--pretrain_max_vocab', type=int, default=250000)
    parser.add_argument('--word_dropout', type=float, default=0.33)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--char_rec_dropout', type=float, default=0, help="Recurrent dropout")

    # TODO: refactor charlm arguments for models which use it?
    parser.add_argument('--no_char', dest='char', action='store_false', help="Turn off character model.")
    parser.add_argument('--char_bidirectional', dest='char_bidirectional', action='store_true', help="Use a bidirectional version of the non-pretrained charlm.  Doesn't help much, makes the models larger")
    parser.add_argument('--char_lowercase', dest='char_lowercase', action='store_true', help="Use lowercased characters in character model.")
    parser.add_argument('--charlm', action='store_true', help="Turn on contextualized char embedding using pretrained character-level language model.")
    parser.add_argument('--charlm_save_dir', type=str, default='saved_models/charlm', help="Root dir for pretrained character-level language model.")
    parser.add_argument('--charlm_shorthand', type=str, default=None, help="Shorthand for character-level language model training corpus.")
    parser.add_argument('--charlm_forward_file', type=str, default=None, help="Exact path to use for forward charlm")
    parser.add_argument('--charlm_backward_file', type=str, default=None, help="Exact path to use for backward charlm")

    parser.add_argument('--bert_model', type=str, default=None, help="Use an external bert model (requires the transformers package)")
    parser.add_argument('--no_bert_model', dest='bert_model', action="store_const", const=None, help="Don't use bert")
    parser.add_argument('--bert_hidden_layers', type=int, default=None, help="How many layers of hidden state to use from the transformer")
    parser.add_argument('--bert_finetune', default=False, action='store_true', help='Finetune the bert (or other transformer)')
    parser.add_argument('--no_bert_finetune', dest='bert_finetune', action='store_false', help="Don't finetune the bert (or other transformer)")
    parser.add_argument('--bert_learning_rate', default=1.0, type=float, help='Scale the learning rate for transformer finetuning by this much')

    parser.add_argument('--no_pretrain', dest='pretrain', action='store_false', help="Turn off pretrained embeddings.")
    parser.add_argument('--share_hid', action='store_true', help="Share hidden representations for UPOS, XPOS and UFeats.")
    parser.set_defaults(share_hid=False)

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam, adamw, adamax, or adadelta.  madgrad as an optional dependency')
    parser.add_argument('--second_optim', type=str, default='amsgrad', help='Optimizer for the second half of training.  Default is Adam with AMSGrad')
    parser.add_argument('--second_optim_reload', default=False, action='store_true', help='Reload the best model instead of continuing from current model if the first optimizer stalls out.  This does not seem to help, but might be useful for further experiments')
    parser.add_argument('--no_second_optim', action='store_const', const=None, dest='second_optim', help="Don't use a second optimizer - only use the first optimizer")
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--second_lr', type=float, default=None, help='Alternate learning rate for the second optimizer')
    parser.add_argument('--initial_weight_decay', type=float, default=None, help='Optimizer weight decay for the first optimizer')
    parser.add_argument('--second_weight_decay', type=float, default=None, help='Optimizer weight decay for the second optimizer')
    parser.add_argument('--beta2', type=float, default=0.95)

    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--fix_eval_interval', dest='adapt_eval_interval', action='store_false', \
            help="Use fixed evaluation interval for all treebanks, otherwise by default the interval will be increased for larger treebanks.")
    parser.add_argument('--max_steps_before_stop', type=int, default=3000, help='Changes learning method or early terminates after this many steps if the dev scores are not improving')
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--batch_maximum_tokens', type=int, default=5000, help='When run in a Pipeline, limit a batch to this many tokens to help avoid OOM for long sentences')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--log_norms', action='store_true', default=False, help='Log the norms of all the parameters (noisy!)')
    parser.add_argument('--save_dir', type=str, default='saved_models/pos', help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default="{shorthand}_{embedding}_tagger.pt", help="File name to save the model")
    parser.add_argument('--save_each', default=False, action='store_true', help="Save each checkpoint to its own model.  Will take up a bunch of space")

    parser.add_argument('--seed', type=int, default=1234)
    add_peft_args(parser)
    utils.add_device_args(parser)

    parser.add_argument('--augment_nopunct', type=float, default=None, help='Augment the training data by copying this fraction of punct-ending sentences as non-punct.  Default of None will aim for roughly 50%%')

    parser.add_argument('--wandb', action='store_true', help='Start a wandb session and write the results of training.  Only applies to training.  Use --wandb_name instead to specify a name')
    parser.add_argument('--wandb_name', default=None, help='Name of a wandb session to start when training.  Will default to the dataset short name')
    return parser

def parse_args(args=None):
    parser = build_argparse()
    args = parser.parse_args(args=args)
    resolve_peft_args(args, logger)

    if args.augment_nopunct is None:
        args.augment_nopunct = 0.25

    if args.wandb_name:
        args.wandb = True

    args = vars(args)
    return args

def main(args=None):
    args = parse_args(args=args)

    utils.set_random_seed(args['seed'])

    logger.info("Running tagger in {} mode".format(args['mode']))

    if args['mode'] == 'train':
        train(args)
    else:
        evaluate(args)

def model_file_name(args):
    return utils.standard_model_file_name(args, "tagger")

def save_each_file_name(args):
    model_file = model_file_name(args)
    pieces = os.path.splitext(model_file)
    return pieces[0] + "_%05d" + pieces[1]

def load_pretrain(args):
    pt = None
    if args['pretrain']:
        pretrain_file = pretrain.find_pretrain_file(args['wordvec_pretrain_file'], args['save_dir'], args['shorthand'], args['lang'])
        if os.path.exists(pretrain_file):
            vec_file = None
        else:
            vec_file = args['wordvec_file'] if args['wordvec_file'] else utils.get_wordvec_file(args['wordvec_dir'], args['shorthand'])
        pt = pretrain.Pretrain(pretrain_file, vec_file, args['pretrain_max_vocab'])
    return pt

def get_eval_type(dev_batch):
    """
    If there is only one column to score in the dev set, use that instead of AllTags
    """
    if dev_batch.has_xpos and not dev_batch.has_upos and not dev_batch.has_feats:
        return "XPOS"
    elif dev_batch.has_upos and not dev_batch.has_xpos and not dev_batch.has_feats:
        return "UPOS"
    else:
        return "AllTags"

def load_training_data(args, pretrain):
    train_docs = []
    raw_train_files = args['train_file'].split(";")
    train_files = []
    for train_file in raw_train_files:
        if zipfile.is_zipfile(train_file):
            logger.info("Decompressing %s" % train_file)
            with zipfile.ZipFile(train_file) as zin:
                for zipped_train_file in zin.namelist():
                    with zin.open(zipped_train_file) as fin:
                        logger.info("Reading %s from %s" % (zipped_train_file, train_file))
                        train_str = fin.read()
                        train_str = train_str.decode("utf-8")
                        train_file_data, _, _ = CoNLL.conll2dict(input_str=train_str)
                        logger.info("Train File {} from {}, Data Size: {}".format(zipped_train_file, train_file, len(train_file_data)))
                        train_docs.append(Document(train_file_data))
                        train_files.append("%s %s" % (train_file, zipped_train_file))
        else:
            logger.info("Reading %s" % train_file)
            # train_data is now a list of sentences, where each sentence is a
            # list of words, in which each word is a dict of conll attributes
            train_file_data, _, _ = CoNLL.conll2dict(input_file=train_file)
            logger.info("Train File {}, Data Size: {}".format(train_file, len(train_file_data)))
            train_docs.append(Document(train_file_data))
            train_files.append(train_file)
    if sum(len(x.sentences) for x in train_docs) == 0:
        raise RuntimeError("Training data for the tagger is empty: %s" % args['train_file'])
    # we want to ensure that the model is able te output _ for empty columns,
    # but create batches whereby if a doc has upos/xpos tags we include them all.
    # therefore, we create seperate datasets and loaders for each input training file,
    # which will ensure the system be able to see batches with both upos available
    # and upos unavailable depending on what the availability in the file is.
    vocab = Dataset.init_vocab(train_docs, args)
    train_data = [Dataset(i, args, pretrain, vocab=vocab, evaluation=False)
                  for i in train_docs]
    for train_file, td in zip(train_files, train_data):
        if not td.has_upos:
            logger.info("No UPOS in %s" % train_file)
        if not td.has_xpos:
            logger.info("No XPOS in %s" % train_file)
        if not td.has_feats:
            logger.info("No feats in %s" % train_file)

    # reject partially tagged upos or xpos documents
    # otherwise, the model will learn to output blanks for some words,
    # which is probably a confusing result
    # (and definitely throws off the depparse)
    # another option would be to treat those as masked out
    for td_idx, td in enumerate(train_data):
        if td.has_upos:
            upos_data = td.doc.get(UPOS, as_sentences=True)
            for sentence_idx, sentence in enumerate(upos_data):
                for word_idx, upos in enumerate(sentence):
                    if upos == '_' or upos is None:
                        conll = "{:C}".format(td.doc.sentences[sentence_idx])
                        raise RuntimeError("Found a blank tag in the UPOS at sentence %d word %d of %s.\n%s" % ((sentence_idx+1), (word_idx+1), train_files[td_idx], conll))

    # here we make sure the model will learn to output _ for empty columns
    # if *any* dataset has data for the upos, xpos, or feature column,
    # we consider that data enough to train the model on that column
    # otherwise, we want to train the model to always output blanks
    if not any(td.has_upos for td in train_data):
        for td in train_data:
            td.has_upos = True
    if not any(td.has_xpos for td in train_data):
        for td in train_data:
            td.has_xpos = True
    if not any(td.has_feats for td in train_data):
        for td in train_data:
            td.has_feats = True
    # calculate the batches
    train_batches = ShuffledDataset(train_data, args["batch_size"])
    return vocab, train_data, train_batches

def train(args):
    model_file = model_file_name(args)
    utils.ensure_dir(os.path.split(model_file)[0])

    if args['save_each']:
        # so models.pt -> models_0001.pt, etc
        model_save_each_file = save_each_file_name(args)
        logger.info("Saving each checkpoint to %s" % model_save_each_file)

    # load pretrained vectors if needed
    pretrain = load_pretrain(args)

    if args['charlm']:
        if args['charlm_shorthand'] is None:
            raise ValueError("CharLM Shorthand is required for loading pretrained CharLM model...")
        logger.info('Using pretrained contextualized char embedding')
        if not args['charlm_forward_file']:
            args['charlm_forward_file'] = '{}/{}_forward_charlm.pt'.format(args['charlm_save_dir'], args['charlm_shorthand'])
        if not args['charlm_backward_file']:
            args['charlm_backward_file'] = '{}/{}_backward_charlm.pt'.format(args['charlm_save_dir'], args['charlm_shorthand'])

    # load data
    logger.info("Loading data with batch size {}...".format(args['batch_size']))
    vocab, train_data, train_batches = load_training_data(args, pretrain)

    dev_doc = CoNLL.conll2doc(input_file=args['eval_file'])
    dev_data = Dataset(dev_doc, args, pretrain, vocab=vocab, evaluation=True, sort_during_eval=True)
    dev_batch = dev_data.to_loader(batch_size=args["batch_size"])

    eval_type = get_eval_type(dev_data)

    # pred and gold path
    system_pred_file = args['output_file']

    # skip training if the language does not have training or dev data
    # sum(...) to check if all of the training files are empty
    if sum(len(td) for td in train_data) == 0 or len(dev_data) == 0:
        logger.info("Skip training because no data available...")
        return

    if args['wandb']:
        import wandb
        wandb_name = args['wandb_name'] if args['wandb_name'] else "%s_tagger" % args['shorthand']
        wandb.init(name=wandb_name, config=args)
        wandb.run.define_metric('train_loss', summary='min')
        wandb.run.define_metric('dev_score', summary='max')

    logger.info("Training tagger...")
    foundation_cache = FoundationCache()
    trainer = Trainer(args=args, vocab=vocab, pretrain=pretrain, device=args['device'], foundation_cache=foundation_cache)

    global_step = 0
    max_steps = args['max_steps']
    dev_score_history = []
    best_dev_preds = []
    current_lr = args['lr']
    global_start_time = time.time()
    format_str = 'Finished STEP {}/{}, loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'

    if args['adapt_eval_interval']:
        args['eval_interval'] = utils.get_adaptive_eval_interval(dev_data.num_examples, 2000, args['eval_interval'])
        logger.info("Evaluating the model every {} steps...".format(args['eval_interval']))

    if args['save_each']:
        logger.info("Saving initial checkpoint to %s" % (model_save_each_file % global_step))
        trainer.save(model_save_each_file % global_step)

    using_amsgrad = False
    last_best_step = 0
    # start training
    train_loss = 0
    if args['log_norms']:
        trainer.model.log_norms()
    while True:
        do_break = False
        for i, batch in enumerate(train_batches):
            start_time = time.time()
            global_step += 1
            loss = trainer.update(batch, eval=False) # update step
            train_loss += loss
            if global_step % args['log_step'] == 0:
                duration = time.time() - start_time
                logger.info(format_str.format(global_step, max_steps, loss, duration, current_lr))
                if args['log_norms']:
                    trainer.model.log_norms()

            if global_step % args['eval_interval'] == 0:
                # eval on dev
                logger.info("Evaluating on dev set...")
                dev_preds = []
                indices = []
                for batch in dev_batch:
                    preds = trainer.predict(batch)
                    dev_preds += preds
                    indices.extend(batch[-1])
                dev_preds = utils.unsort(dev_preds, indices)
                dev_data.doc.set([UPOS, XPOS, FEATS], [y for x in dev_preds for y in x])
                CoNLL.write_doc2conll(dev_data.doc, system_pred_file)

                _, _, dev_score = scorer.score(system_pred_file, args['eval_file'], eval_type=eval_type)

                train_loss = train_loss / args['eval_interval'] # avg loss per batch
                logger.info("step {}: train_loss = {:.6f}, dev_score = {:.4f}".format(global_step, train_loss, dev_score))

                if args['wandb']:
                    wandb.log({'train_loss': train_loss, 'dev_score': dev_score})

                train_loss = 0

                if args['save_each']:
                    logger.info("Saving checkpoint to %s" % (model_save_each_file % global_step))
                    trainer.save(model_save_each_file % global_step)

                # save best model
                if len(dev_score_history) == 0 or dev_score > max(dev_score_history):
                    last_best_step = global_step
                    trainer.save(model_file)
                    logger.info("new best model saved.")
                    best_dev_preds = dev_preds

                dev_score_history += [dev_score]

            if global_step - last_best_step >= args['max_steps_before_stop']:
                if not using_amsgrad and args['second_optim'] is not None:
                    logger.info("Switching to second optimizer: {}".format(args['second_optim']))
                    if args['second_optim_reload']:
                        logger.info('Reloading best model to continue from current local optimum')
                        trainer = Trainer(args=args, vocab=trainer.vocab, pretrain=pretrain, model_file=model_file, device=args['device'], foundation_cache=foundation_cache)
                    last_best_step = global_step
                    using_amsgrad = True
                    lr = args['second_lr']
                    if lr is None:
                        lr = args['lr']
                    trainer.optimizer = utils.get_optimizer(args['second_optim'], trainer.model, lr=lr, betas=(.9, args['beta2']), eps=1e-6, weight_decay=args['second_weight_decay'])
                else:
                    logger.info("Early termination: have not improved in {} steps".format(args['max_steps_before_stop']))
                    do_break = True
                    break

            if global_step >= args['max_steps']:
                do_break = True
                break

        if do_break: break

    logger.info("Training ended with {} steps.".format(global_step))

    if args['wandb']:
        wandb.finish()

    if len(dev_score_history) > 0:
        best_f, best_eval = max(dev_score_history)*100, np.argmax(dev_score_history)+1
        logger.info("Best dev F1 = {:.2f}, at iteration = {}".format(best_f, best_eval * args['eval_interval']))
    else:
        logger.info("Dev set never evaluated.  Saving final model.")
        trainer.save(model_file)


def evaluate(args):
    # file paths
    model_file = model_file_name(args)

    pretrain = load_pretrain(args)

    load_args = {'charlm_forward_file': args.get('charlm_forward_file', None),
                 'charlm_backward_file': args.get('charlm_backward_file', None)}

    # load model
    logger.info("Loading model from: {}".format(model_file))
    trainer = Trainer(pretrain=pretrain, model_file=model_file, device=args['device'], args=load_args)
    evaluate_trainer(args, trainer, pretrain)

def evaluate_trainer(args, trainer, pretrain):
    system_pred_file = args['output_file']
    loaded_args, vocab = trainer.args, trainer.vocab

    # load config
    for k in args:
        if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand'] or k == 'mode':
            loaded_args[k] = args[k]

    # load data
    logger.info("Loading data with batch size {}...".format(args['batch_size']))
    doc = CoNLL.conll2doc(input_file=args['eval_file'])
    dev_data = Dataset(doc, loaded_args, pretrain, vocab=vocab, evaluation=True, sort_during_eval=True)
    dev_batch = dev_data.to_loader(batch_size=args['batch_size'])
    eval_type = get_eval_type(dev_data)
    if len(dev_batch) > 0:
        logger.info("Start evaluation...")
        preds = []
        indices = []
        with torch.no_grad():
            for b in dev_batch:
                preds += trainer.predict(b)
                indices.extend(b[-1])
    else:
        # skip eval if dev data does not exist
        preds = []
    preds = utils.unsort(preds, indices)

    # write to file and score
    dev_data.doc.set([UPOS, XPOS, FEATS], [y for x in preds for y in x])
    CoNLL.write_doc2conll(dev_data.doc, system_pred_file)

    if args['gold_labels']:
        _, _, score = scorer.score(system_pred_file, args['eval_file'], eval_type=eval_type)

        logger.info("POS Tagger score: %s %.2f", args['shorthand'], score*100)

if __name__ == '__main__':
    main()
