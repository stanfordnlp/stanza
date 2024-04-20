"""
Entry point for training and evaluating a dependency parser.

This implementation combines a deep biaffine graph-based parser with linearization and distance features.
For details please refer to paper: https://nlp.stanford.edu/pubs/qi2018universal.pdf.
"""

"""
Training and evaluation for the parser.
"""

import sys
import os
import copy
import shutil
import time
import argparse
import logging
import numpy as np
import random
import torch
from torch import nn, optim

import stanza.models.depparse.data as data
from stanza.models.depparse.data import DataLoader
from stanza.models.depparse.trainer import Trainer
from stanza.models.depparse import scorer
from stanza.models.common import utils
from stanza.models.common import pretrain
from stanza.models.common.data import augment_punct
from stanza.models.common.doc import *
from stanza.models.common.peft_config import add_peft_args, resolve_peft_args
from stanza.utils.conll import CoNLL
from stanza.models import _training_logging

logger = logging.getLogger('stanza')

def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/depparse', help='Root dir for saving models.')
    parser.add_argument('--wordvec_dir', type=str, default='extern_data/word2vec', help='Directory of word vectors.')
    parser.add_argument('--wordvec_file', type=str, default=None, help='Word vectors filename.')
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')
    parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--output_file', type=str, default=None, help='Output CoNLL-U file.')
    parser.add_argument('--no_gold_labels', dest='gold_labels', action='store_false', help="Don't score the eval file - perhaps it has no gold labels, for example.  Cannot be used at training time")
    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')
    parser.add_argument('--shorthand', type=str, help="Treebank shorthand")

    parser.add_argument('--hidden_dim', type=int, default=400)
    parser.add_argument('--char_hidden_dim', type=int, default=400)
    parser.add_argument('--deep_biaff_hidden_dim', type=int, default=400)
    parser.add_argument('--composite_deep_biaff_hidden_dim', type=int, default=100)
    parser.add_argument('--word_emb_dim', type=int, default=75)
    parser.add_argument('--char_emb_dim', type=int, default=100)
    parser.add_argument('--tag_emb_dim', type=int, default=50)
    parser.add_argument('--no_upos', dest='use_upos', action='store_false', default=True, help="Don't use upos tags as part of the tag embedding")
    parser.add_argument('--no_xpos', dest='use_xpos', action='store_false', default=True, help="Don't use xpos tags as part of the tag embedding")
    parser.add_argument('--no_ufeats', dest='use_ufeats', action='store_false', default=True, help="Don't use ufeats as part of the tag embedding")
    parser.add_argument('--transformed_dim', type=int, default=125)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--checkpoint_save_name', type=str, default=None, help="File name to save the most recent checkpoint")
    parser.add_argument('--no_checkpoint', dest='checkpoint', action='store_false', help="Don't save checkpoints")
    parser.add_argument('--pretrain_max_vocab', type=int, default=250000)
    parser.add_argument('--word_dropout', type=float, default=0.33)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--char_rec_dropout', type=float, default=0, help="Recurrent dropout")

    parser.add_argument('--no_char', dest='char', action='store_false', help="Turn off character model.")
    parser.add_argument('--charlm', action='store_true', help="Turn on contextualized char embedding using pretrained character-level language model.")
    parser.add_argument('--charlm_save_dir', type=str, default='saved_models/charlm', help="Root dir for pretrained character-level language model.")
    parser.add_argument('--charlm_shorthand', type=str, default=None, help="Shorthand for character-level language model training corpus.")
    parser.add_argument('--charlm_forward_file', type=str, default=None, help="Exact path to use for forward charlm")
    parser.add_argument('--charlm_backward_file', type=str, default=None, help="Exact path to use for backward charlm")

    parser.add_argument('--bert_model', type=str, default=None, help="Use an external bert model (requires the transformers package)")
    parser.add_argument('--no_bert_model', dest='bert_model', action="store_const", const=None, help="Don't use bert")
    parser.add_argument('--bert_hidden_layers', type=int, default=4, help="How many layers of hidden state to use from the transformer")
    parser.add_argument('--bert_hidden_layers_original', action='store_const', const=None, dest='bert_hidden_layers', help='Use layers 2,3,4 of the Bert embedding')
    parser.add_argument('--bert_finetune', default=False, action='store_true', help='Finetune the bert (or other transformer)')
    parser.add_argument('--no_bert_finetune', dest='bert_finetune', action='store_false', help="Don't finetune the bert (or other transformer)")
    parser.add_argument('--bert_finetune_layers', default=None, type=int, help='Only finetune this many layers from the transformer')
    parser.add_argument('--bert_learning_rate', default=1.0, type=float, help='Scale the learning rate for transformer finetuning by this much')
    parser.add_argument('--second_bert_learning_rate', default=1e-3, type=float, help='Secondary stage transformer finetuning learning rate scale')
    parser.add_argument('--bert_start_finetuning', default=200, type=int, help='When to start finetuning the transformer')
    parser.add_argument('--bert_warmup_steps', default=200, type=int, help='How many steps for a linear warmup when finetuning the transformer')
    parser.add_argument('--bert_weight_decay', default=0.0, type=float, help='Weight decay bert parameters by this much')

    parser.add_argument('--no_pretrain', dest='pretrain', action='store_false', help="Turn off pretrained embeddings.")
    parser.add_argument('--no_linearization', dest='linearization', action='store_false', help="Turn off linearization term.")
    parser.add_argument('--no_distance', dest='distance', action='store_false', help="Turn off distance term.")

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--second_optim', type=str, default=None, help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--second_lr', type=float, default=3e-4, help='Secondary stage learning rate')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay for the first optimizer')
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--second_optim_start_step', type=int, default=None, help='If set, switch to the second optimizer when stalled or at this step regardless of performance.  Normally, the optimizer only switches when the dev scores have stalled for --max_steps_before_stop steps')
    parser.add_argument('--second_warmup_steps', type=int, default=200, help="If set, give the 2nd optimizer a linear warmup.  Idea being that the optimizer won't have a good grasp on the initial gradients and square gradients when it first starts")

    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--checkpoint_interval', type=int, default=500)
    parser.add_argument('--max_steps_before_stop', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--second_batch_size', type=int, default=None, help='Use a different batch size for the second optimizer.  Can be relevant for models with different transformer finetuning settings between optimizers, for example, where the larger batch size is impossible for FT the transformer"')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--log_norms', action='store_true', default=False, help='Log the norms of all the parameters (noisy!)')
    parser.add_argument('--save_dir', type=str, default='saved_models/depparse', help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default="{shorthand}_{embedding}_parser.pt", help="File name to save the model")
    parser.add_argument('--continue_from', type=str, default=None, help="File name to preload the model to continue training from")

    parser.add_argument('--seed', type=int, default=1234)
    add_peft_args(parser)
    utils.add_device_args(parser)

    parser.add_argument('--augment_nopunct', type=float, default=None, help='Augment the training data by copying this fraction of punct-ending sentences as non-punct.  Default of None will aim for roughly 10%%')

    parser.add_argument('--wandb', action='store_true', help='Start a wandb session and write the results of training.  Only applies to training.  Use --wandb_name instead to specify a name')
    parser.add_argument('--wandb_name', default=None, help='Name of a wandb session to start when training.  Will default to the dataset short name')
    return parser

def parse_args(args=None):
    parser = build_argparse()
    args = parser.parse_args(args=args)
    resolve_peft_args(args, logger)

    if args.wandb_name:
        args.wandb = True

    args = vars(args)
    return args

def main(args=None):
    args = parse_args(args=args)

    utils.set_random_seed(args['seed'])

    logger.info("Running parser in {} mode".format(args['mode']))

    if args['mode'] == 'train':
        return train(args)
    else:
        evaluate(args)

def model_file_name(args):
    return utils.standard_model_file_name(args, "parser")

# TODO: refactor with everywhere
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

def predict_dataset(trainer, dev_batch):
    dev_preds = []
    if len(dev_batch) > 0:
        for batch in dev_batch:
            preds = trainer.predict(batch)
            dev_preds += preds
        dev_preds = utils.unsort(dev_preds, dev_batch.data_orig_idx)
    return dev_preds

def train(args):
    model_file = model_file_name(args)
    utils.ensure_dir(os.path.split(model_file)[0])

    # load pretrained vectors if needed
    pretrain = load_pretrain(args)

    # TODO: refactor.  the exact same thing is done in the tagger
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
    train_data, _, _ = CoNLL.conll2dict(input_file=args['train_file'])
    # possibly augment the training data with some amount of fake data
    # based on the options chosen
    logger.info("Original data size: {}".format(len(train_data)))
    train_data.extend(augment_punct(train_data, args['augment_nopunct'],
                                    keep_original_sentences=False))
    logger.info("Augmented data size: {}".format(len(train_data)))
    train_doc = Document(train_data)
    train_batch = DataLoader(train_doc, args['batch_size'], args, pretrain, evaluation=False)
    vocab = train_batch.vocab
    dev_doc = CoNLL.conll2doc(input_file=args['eval_file'])
    dev_batch = DataLoader(dev_doc, args['batch_size'], args, pretrain, vocab=vocab, evaluation=True, sort_during_eval=True)

    # pred path
    system_pred_file = args['output_file']

    # skip training if the language does not have training or dev data
    if len(train_batch) == 0 or len(dev_batch) == 0:
        logger.info("Skip training because no data available...")
        sys.exit(0)

    if args['wandb']:
        import wandb
        wandb_name = args['wandb_name'] if args['wandb_name'] else "%s_depparse" % args['shorthand']
        wandb.init(name=wandb_name, config=args)
        wandb.run.define_metric('train_loss', summary='min')
        wandb.run.define_metric('dev_score', summary='max')

    logger.info("Training parser...")
    checkpoint_file = None
    if args.get("checkpoint"):
        # calculate checkpoint file name from the save filename
        checkpoint_file = utils.checkpoint_name(args.get("save_dir"), model_file, args.get("checkpoint_save_name"))
        args["checkpoint_save_name"] = checkpoint_file

    if args.get("checkpoint") and os.path.exists(args["checkpoint_save_name"]):
        trainer = Trainer(args=args, pretrain=pretrain, vocab=vocab, model_file=args["checkpoint_save_name"], device=args['device'], ignore_model_config=True)
        if len(trainer.dev_score_history) > 0:
            logger.info("Continuing from checkpoint %s  Model was previously trained for %d steps, with a best dev score of %.4f", args["checkpoint_save_name"], trainer.global_step, max(trainer.dev_score_history))
    elif args["continue_from"]:
        if not os.path.exists(args["continue_from"]):
            raise FileNotFoundError("--continue_from specified, but the file %s does not exist" % args["continue_from"])
        trainer = Trainer(args=args, pretrain=pretrain, vocab=vocab, model_file=args["continue_from"], device=args['device'], ignore_model_config=True, reset_history=True)
    else:
        trainer = Trainer(args=args, vocab=vocab, pretrain=pretrain, device=args['device'])

    max_steps = args['max_steps']
    current_lr = args['lr']
    global_start_time = time.time()
    format_str = 'Finished STEP {}/{}, loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'

    is_second_stage = False
    # start training
    train_loss = 0
    if args['log_norms']:
        trainer.model.log_norms()
    while True:
        do_break = False
        for i, batch in enumerate(train_batch):
            start_time = time.time()
            trainer.global_step += 1
            loss = trainer.update(batch, eval=False) # update step
            train_loss += loss

            # will checkpoint if we switch optimizers or score a new best score
            force_checkpoint = False
            if trainer.global_step % args['log_step'] == 0:
                duration = time.time() - start_time
                logger.info(format_str.format(trainer.global_step, max_steps, loss, duration, current_lr))

            if trainer.global_step % args['eval_interval'] == 0:
                # eval on dev
                logger.info("Evaluating on dev set...")
                dev_preds = predict_dataset(trainer, dev_batch)

                dev_batch.doc.set([HEAD, DEPREL], [y for x in dev_preds for y in x])
                CoNLL.write_doc2conll(dev_batch.doc, system_pred_file)
                _, _, dev_score = scorer.score(system_pred_file, args['eval_file'])

                train_loss = train_loss / args['eval_interval'] # avg loss per batch
                logger.info("step {}: train_loss = {:.6f}, dev_score = {:.4f}".format(trainer.global_step, train_loss, dev_score))

                if args['wandb']:
                    wandb.log({'train_loss': train_loss, 'dev_score': dev_score})

                train_loss = 0

                # save best model
                trainer.dev_score_history += [dev_score]
                if dev_score >= max(trainer.dev_score_history):
                    trainer.last_best_step = trainer.global_step
                    trainer.save(model_file)
                    logger.info("new best model saved.")
                    force_checkpoint = True

                for scheduler_name, scheduler in trainer.scheduler.items():
                    logger.info('scheduler %s learning rate: %s', scheduler_name, scheduler.get_last_lr())
                if args['log_norms']:
                    trainer.model.log_norms()

            if not is_second_stage and args.get('second_optim', None) is not None:
                if trainer.global_step - trainer.last_best_step >= args['max_steps_before_stop'] or (args['second_optim_start_step'] is not None and trainer.global_step >= args['second_optim_start_step']):
                    logger.info("Switching to second optimizer: {}".format(args.get('second_optim', None)))
                    global_step = trainer.global_step
                    args["second_stage"] = True
                    # if the loader gets a model file, it uses secondary optimizer
                    # (because of the second_stage = True argument)
                    trainer = Trainer(args=args, vocab=trainer.vocab, pretrain=pretrain,
                                      model_file=model_file, device=args['device'])
                    logger.info('Reloading best model to continue from current local optimum')

                    dev_preds = predict_dataset(trainer, dev_batch)
                    dev_batch.doc.set([HEAD, DEPREL], [y for x in dev_preds for y in x])
                    CoNLL.write_doc2conll(dev_batch.doc, system_pred_file)
                    _, _, dev_score = scorer.score(system_pred_file, args['eval_file'])
                    logger.info("Reloaded model with dev score %.4f", dev_score)

                    is_second_stage = True
                    trainer.global_step = global_step
                    trainer.last_best_step = global_step
                    if args['second_batch_size'] is not None:
                        train_batch.set_batch_size(args['second_batch_size'])
                    force_checkpoint = True
            else:
                if trainer.global_step - trainer.last_best_step >= args['max_steps_before_stop']:
                    do_break = True
                    break

            if trainer.global_step % args['eval_interval'] == 0 or force_checkpoint:
                # if we need to save checkpoint, do so
                # (save after switching the optimizer, if applicable, so that
                # the new optimizer is the optimizer used if a restart happens)
                if checkpoint_file is not None:
                    trainer.save(checkpoint_file, save_optimizer=True)
                    logger.info("new model checkpoint saved.")

            if trainer.global_step >= args['max_steps']:
                do_break = True
                break

        if do_break: break

        train_batch.reshuffle()

    logger.info("Training ended with {} steps.".format(trainer.global_step))

    if args['wandb']:
        wandb.finish()

    if len(trainer.dev_score_history) > 0:
        # TODO: technically the iteration position will be wrong if
        # the eval_interval changed when running from a checkpoint
        # could fix this by saving step & score instead of just score
        best_f, best_eval = max(trainer.dev_score_history)*100, np.argmax(trainer.dev_score_history)+1
        logger.info("Best dev F1 = {:.2f}, at iteration = {}".format(best_f, best_eval * args['eval_interval']))
    else:
        logger.info("Dev set never evaluated.  Saving final model.")
        trainer.save(model_file)

    return trainer

def evaluate(args):
    model_file = model_file_name(args)
    # load pretrained vectors if needed
    pretrain = load_pretrain(args)

    load_args = {'charlm_forward_file': args.get('charlm_forward_file', None),
                 'charlm_backward_file': args.get('charlm_backward_file', None)}

    # load model
    logger.info("Loading model from: {}".format(model_file))
    trainer = Trainer(pretrain=pretrain, model_file=model_file, device=args['device'], args=load_args)
    return evaluate_trainer(args, trainer, pretrain)

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
    batch = DataLoader(doc, args['batch_size'], loaded_args, pretrain, vocab=vocab, evaluation=True, sort_during_eval=True)

    preds = predict_dataset(trainer, batch)

    # write to file and score
    batch.doc.set([HEAD, DEPREL], [y for x in preds for y in x])
    CoNLL.write_doc2conll(batch.doc, system_pred_file)

    if args['gold_labels']:
        gold_doc = CoNLL.conll2doc(input_file=args['eval_file'])

        # Check for None ... otherwise an inscrutable error occurs later in the scorer
        for sent_idx, sentence in enumerate(gold_doc.sentences):
            for word_idx, word in enumerate(sentence.words):
                if word.deprel is None:
                    raise ValueError("Gold document {} has a None at sentence {} word {}\n{:C}".format(args['eval_file'], sent_idx, word_idx, sentence))

        scorer.score_named_dependencies(batch.doc, gold_doc)
        _, _, score = scorer.score(system_pred_file, args['eval_file'])

        logger.info("Parser score:")
        logger.info("{} {:.2f}".format(args['shorthand'], score*100))

if __name__ == '__main__':
    main()
