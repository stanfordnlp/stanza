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
import numpy as np
import random
import json
import torch
from torch import nn, optim

from stanfordnlp.models.ner.data import DataLoader
from stanfordnlp.models.ner.trainer import Trainer
from stanfordnlp.models.ner import scorer
from stanfordnlp.models.common import utils
from stanfordnlp.models.common.pretrain import Pretrain
from stanfordnlp.utils.conll import CoNLL
from stanfordnlp.models.common.doc import *
from stanfordnlp.models import _training_logging

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/ner', help='Root dir for saving models.')
    parser.add_argument('--wordvec_dir', type=str, default='extern_data/word2vec', help='Directory of word vectors')
    parser.add_argument('--wordvec_file', type=str, default='', help='File that contains word vectors')
    parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')
    parser.add_argument('--shorthand', type=str, help="Treebank shorthand")

    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--char_hidden_dim', type=int, default=25)
    parser.add_argument('--word_emb_dim', type=int, default=100)
    parser.add_argument('--char_emb_dim', type=int, default=25)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--pretrain_max_vocab', type=int, default=-1)
    parser.add_argument('--word_dropout', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--rec_dropout', type=float, default=0, help="Word recurrent dropout")
    parser.add_argument('--char_rec_dropout', type=float, default=0, help="Character recurrent dropout")
    parser.add_argument('--char_dropout', type=float, default=0, help="Character-level language model dropout")
    parser.add_argument('--no_char', dest='char', action='store_false', help="Turn off character model.")
    parser.add_argument('--char_context', action='store_true', help="Turn on contextualized char embedding model.")
    parser.add_argument('--charlm_save_dir', type=str, default='saved_models/charlm', help="Root dir for pretrained character-level language model.")
    parser.add_argument('--no_lowercase', dest='lowercase', action='store_false', help="Use cased word vectors.")
    parser.add_argument('--scheme', type=str, default='bioes', help="The tagging scheme to use: bio or bioes.")

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='sgd', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
    parser.add_argument('--lr_decay', type=float, default=0.9, help="LR decay rate.")
    parser.add_argument('--patience', type=int, default=3, help="Patience for LR decay.")

    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--save_dir', type=str, default='saved_models/ner', help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)

    args = vars(args)
    print("Running tagger in {} mode".format(args['mode']))

    if args['mode'] == 'train':
        train(args)
    else:
        evaluate(args)

def train(args):

    utils.ensure_dir(args['save_dir'])
    model_file = args['save_dir'] + '/' + args['save_name'] if args['save_name'] is not None \
            else '{}/{}_nertagger.pt'.format(args['save_dir'], args['shorthand'])

    # load pretrained vectors
    if len(args['wordvec_file']) == 0:
        vec_file = utils.get_wordvec_file(args['wordvec_dir'], args['shorthand'])
    else:
        vec_file = args['wordvec_file']
    # do not save pretrained embeddings individually

    pretrain_file = '{}/{}.pretrain.pt'.format(args['save_dir'], args['shorthand'])
    pretrain = Pretrain(pretrain_file, vec_file, args['pretrain_max_vocab'])

    if args['char_context']:
        print('Use pretrained contextualized char embedding')
        args['charlm_forward_file'] = '{}/{}_forward_charlm.pt'.format(args['charlm_save_dir'], args['lang'])
        args['charlm_backward_file'] = '{}/{}_backward_charlm.pt'.format(args['charlm_save_dir'], args['lang'])

    # load data
    print("Loading data with batch size {}...".format(args['batch_size']))
    train_doc = Document(json.load(open(args['train_file'])))
    train_batch = DataLoader(train_doc, args['batch_size'], args, pretrain, evaluation=False)
    vocab = train_batch.vocab
    dev_doc = Document(json.load(open(args['eval_file'])))
    dev_batch = DataLoader(dev_doc, args['batch_size'], args, pretrain, vocab=vocab, evaluation=True)
    dev_gold_tags = dev_batch.tags

    # skip training if the language does not have training or dev data
    if len(train_batch) == 0 or len(dev_batch) == 0:
        print("Skip training because no data available...")
        sys.exit(0)

    print("Training tagger...")
    trainer = Trainer(args=args, vocab=vocab, pretrain=pretrain, use_cuda=args['cuda'])

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
            patience=args['patience'], verbose=True, min_lr=1e-6)
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
                current_lr = trainer.optimizer.param_groups[0]['lr']
                duration = time.time() - start_time
                print(format_str.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), global_step,\
                        max_steps, loss, duration, current_lr))

            if global_step % args['eval_interval'] == 0:
                # eval on dev
                print("Evaluating on dev set...")
                dev_preds = []
                for batch in dev_batch:
                    preds = trainer.predict(batch)
                    dev_preds += preds
                _, _, dev_score = scorer.score_by_chunk(dev_gold_tags, dev_preds, scheme=args['scheme'].lower())

                train_loss = train_loss / args['eval_interval'] # avg loss per batch
                print("step {}: train_loss = {:.6f}, dev_score = {:.4f}".format(global_step, train_loss, dev_score))
                train_loss = 0

                # save best model
                if len(dev_score_history) == 0 or dev_score > max(dev_score_history):
                    trainer.save(model_file)
                    print("new best model saved.")
                    best_dev_preds = dev_preds

                dev_score_history += [dev_score]
                print("")

                # lr schedule
                if scheduler is not None:
                    scheduler.step(dev_score)
            
            # check stopping
            if global_step >= args['max_steps']:
                should_stop = True
                break

        if should_stop:
            break

        train_batch.reshuffle()

    print("Training ended with {} steps.".format(global_step))

    best_f, best_eval = max(dev_score_history)*100, np.argmax(dev_score_history)+1
    print("Best dev F1 = {:.2f}, at iteration = {}".format(best_f, best_eval * args['eval_interval']))

def evaluate(args):
    # file paths
    model_file = args['save_dir'] + '/' + args['save_name'] if args['save_name'] is not None \
            else '{}/{}_nertagger.pt'.format(args['save_dir'], args['shorthand'])

    # load model
    use_cuda = args['cuda'] and not args['cpu']
    trainer = Trainer(model_file=model_file, use_cuda=use_cuda)
    loaded_args, vocab = trainer.args, trainer.vocab

    # load config
    for k in args:
        if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand', 'mode', 'scheme']:
            loaded_args[k] = args[k]

    # load data
    print("Loading data with batch size {}...".format(args['batch_size']))
    doc = Document(json.load(open(args['eval_file'])))
    batch = DataLoader(doc, args['batch_size'], loaded_args, vocab=vocab, evaluation=True)
    
    print("Start evaluation...")
    preds = []
    for i, b in enumerate(batch):
        preds += trainer.predict(b)

    gold_tags = batch.tags
    _, _, score = scorer.score_by_chunk(gold_tags, preds, scheme=loaded_args['scheme'].lower())

    print("Tagger score:")
    print("{} {:.2f}".format(args['shorthand'], score*100))

if __name__ == '__main__':
    main()
