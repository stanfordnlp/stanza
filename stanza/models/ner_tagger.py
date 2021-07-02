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

logger = logging.getLogger('stanza')

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/ner', help='Directory of NER data.')
    parser.add_argument('--wordvec_dir', type=str, default='extern_data/word2vec', help='Directory of word vectors')
    parser.add_argument('--wordvec_file', type=str, default='', help='File that contains word vectors')
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')
    parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--finetune', action='store_true', help='Load existing model during `train` mode from `save_dir` path')
    parser.add_argument('--train_classifier_only', action='store_true',
                        help='In case of applying Transfer-learning approach and training only the classifier layer this will freeze gradient propagation for all other layers.')
    parser.add_argument('--lang', type=str, help='Language')
    parser.add_argument('--shorthand', type=str, help="Treebank shorthand")

    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--char_hidden_dim', type=int, default=100)
    parser.add_argument('--word_emb_dim', type=int, default=100)
    parser.add_argument('--char_emb_dim', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--pretrain_max_vocab', type=int, default=100000)
    parser.add_argument('--word_dropout', type=float, default=0)
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
    parser.add_argument('--no_input_transform', dest='input_transform', action='store_false', help="Do not use input transformation layer before tagger lstm.")
    parser.add_argument('--scheme', type=str, default='bioes', help="The tagging scheme to use: bio or bioes.")

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='sgd', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='Minimum learning rate to stop training.')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum for SGD.')
    parser.add_argument('--lr_decay', type=float, default=0.5, help="LR decay rate.")
    parser.add_argument('--patience', type=int, default=3, help="Patience for LR decay.")

    parser.add_argument('--max_steps', type=int, default=200000)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--save_dir', type=str, default='saved_models/ner', help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

    args = parser.parse_args(args=args)
    return args

def main(args=None):
    args = parse_args(args=args)

    if args.cpu:
        args.cuda = False
    utils.set_random_seed(args.seed, args.cuda)

    args = vars(args)
    logger.info("Running NER tagger in {} mode".format(args['mode']))

    if args['mode'] == 'train':
        train(args)
    else:
        evaluate(args)

def train(args):
    utils.ensure_dir(args['save_dir'])
    model_file = os.path.join(args['save_dir'], args['save_name']) if args['save_name'] is not None \
        else '{}/{}_nertagger.pt'.format(args['save_dir'], args['shorthand'])

    pretrain = None
    vocab = None
    trainer = None

    if args['finetune'] and os.path.exists(model_file):
        logger.warning('Finetune is ON. Using model from "{}"'.format(model_file))
        _, trainer, vocab = load_model(args, model_file)
    else:
        if args['finetune']:
            logger.warning('Finetune is set to true but model file is not found. Continuing with training from scratch.')

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
    train_doc = Document(json.load(open(args['train_file'])))
    train_batch = DataLoader(train_doc, args['batch_size'], args, pretrain, vocab=vocab, evaluation=False)
    vocab = train_batch.vocab
    dev_doc = Document(json.load(open(args['eval_file'])))
    dev_batch = DataLoader(dev_doc, args['batch_size'], args, pretrain, vocab=vocab, evaluation=True)
    dev_gold_tags = dev_batch.tags

    # skip training if the language does not have training or dev data
    if len(train_batch) == 0 or len(dev_batch) == 0:
        logger.info("Skip training because no data available...")
        sys.exit(0)

    logger.info("Training tagger...")
    if trainer is None: # init if model was not loaded previously from file
        trainer = Trainer(args=args, vocab=vocab, pretrain=pretrain, use_cuda=args['cuda'],
                          train_classifier_only=args['train_classifier_only'])
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, mode='max', factor=args['lr_decay'], \
            patience=args['patience'], verbose=True, min_lr=args['min_lr'])
    else:
        scheduler = None

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
                logger.info(format_str.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), global_step,\
                        max_steps, loss, duration, current_lr))

            if global_step % args['eval_interval'] == 0:
                # eval on dev
                logger.info("Evaluating on dev set...")
                dev_preds = []
                for batch in dev_batch:
                    preds = trainer.predict(batch)
                    dev_preds += preds
                _, _, dev_score = scorer.score_by_entity(dev_preds, dev_gold_tags)

                train_loss = train_loss / args['eval_interval'] # avg loss per batch
                logger.info("step {}: train_loss = {:.6f}, dev_score = {:.4f}".format(global_step, train_loss, dev_score))
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

    best_f, best_eval = max(dev_score_history)*100, np.argmax(dev_score_history)+1
    logger.info("Best dev F1 = {:.2f}, at iteration = {}".format(best_f, best_eval * args['eval_interval']))

def evaluate(args):
    # file paths
    model_file = os.path.join(args['save_dir'], args['save_name']) if args['save_name'] is not None \
        else '{}/{}_nertagger.pt'.format(args['save_dir'], args['shorthand'])

    loaded_args, trainer, vocab = load_model(args, model_file)

    # load data
    logger.info("Loading data with batch size {}...".format(args['batch_size']))
    doc = Document(json.load(open(args['eval_file'])))
    batch = DataLoader(doc, args['batch_size'], loaded_args, vocab=vocab, evaluation=True)
    
    logger.info("Start evaluation...")
    preds = []
    for i, b in enumerate(batch):
        preds += trainer.predict(b)

    gold_tags = batch.tags
    _, _, score = scorer.score_by_entity(preds, gold_tags)

    logger.info("NER tagger score:")
    logger.info("{} {:.2f}".format(args['shorthand'], score*100))


def load_model(args, model_file):
    # load model
    use_cuda = args['cuda'] and not args['cpu']
    charlm_args = {}
    if 'charlm_forward_file' in args:
        charlm_args['charlm_forward_file'] = args['charlm_forward_file']
    if 'charlm_backward_file' in args:
        charlm_args['charlm_backward_file'] = args['charlm_backward_file']
    trainer = Trainer(args=charlm_args, model_file=model_file, use_cuda=use_cuda, train_classifier_only=args['train_classifier_only'])
    loaded_args, vocab = trainer.args, trainer.vocab

    # load config
    for k in args:
        if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand', 'mode', 'scheme']:
            loaded_args[k] = args[k]
    return loaded_args, trainer, vocab


if __name__ == '__main__':
    main()
