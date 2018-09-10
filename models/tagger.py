"""
Training and evaluation for the tagger.
"""

import os
import shutil
import time
from datetime import datetime
import argparse
import numpy as np
import random
import torch
from torch import nn, optim

from models.pos.data import DataLoader
from models.pos.trainer import Trainer
from models.pos import scorer
from models.common import utils, param
import models.common.seq2seq_constant as constant

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/pos', help='Root dir for saving models.')
    parser.add_argument('--wordvec_dir', type=str, default='extern_data/word2vec', help='Directory of word vectors')
    parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--output_file', type=str, default=None, help='Output CoNLL-U file.')
    parser.add_argument('--gold_file', type=str, default=None, help='Output CoNLL-U file.')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')
    parser.add_argument('--shorthand', type=str, help="Treebank shorthand")
    parser.add_argument('--best_param', action='store_true', help='Train with best language-specific parameters.')

    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--char_hidden_dim', type=int, default=400)
    parser.add_argument('--deep_biaff_hidden_dim', type=int, default=400)
    parser.add_argument('--ufeats_hidden_dim', type=int, default=400)
    parser.add_argument('--word_emb_dim', type=int, default=75)
    parser.add_argument('--char_emb_dim', type=int, default=100)
    parser.add_argument('--tag_emb_dim', type=int, default=50)
    parser.add_argument('--transformed_dim', type=int, default=125)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--word_dropout', type=float, default=0.33)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--char_dropout', type=float, default=0.1)
    parser.add_argument('--rec_dropout', type=float, default=0.5, help="Recurrent dropout")
    parser.add_argument('--char_rec_dropout', type=float, default=0.25, help="Recurrent dropout")

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--beta2', type=float, default=0.95)

    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--max_steps_before_stop', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--save_dir', type=str, default='saved_models/pos', help='Root dir for saving models.')
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
    random.seed(1234)
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
    # load data
    print("Loading data with batch size {}...".format(args['batch_size']))
    train_batch = DataLoader(args['train_file'], args['batch_size'], args, evaluation=False)
    vocab = train_batch.vocab
    args['vocab_size'] = len(vocab)
    dev_batch = DataLoader(args['eval_file'], args['batch_size'], args, evaluation=True)

    model_file = args['save_dir'] + '/' + args['save_name'] if args['save_name'] is not None \
            else '{}/{}_tagger.pt'.format(args['save_dir'], args['shorthand'])

    # pred and gold path
    system_pred_file = args['output_file']
    gold_file = args['gold_file']

    # activate param manager and save config
    param_manager = param.ParamManager('params/pos', args['shorthand'])
    if args['best_param']: # use best param in file, otherwise use command line params
        args = param_manager.load_to_args(args)
    utils.save_config(args, '{}/{}_config.json'.format(args['save_dir'], args['shorthand']))

    # skip training if the language does not have training or dev data
    if len(train_batch) == 0 or len(dev_batch) == 0:
        print("Skip training because no data available...")
        exit()

    print("Training tagger...")
    trainer = Trainer(args, vocab, train_batch.pretrained_emb)

    global_step = 0
    max_steps = args['max_steps']
    dev_score_history = []
    best_dev_preds = []
    current_lr = args['lr']
    global_start_time = time.time()
    format_str = '{}: step {}/{}, loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'

    using_amsgrad = False
    last_best_step = 0
    # start training
    train_loss = 0
    while True:
        do_break = False
        for i, batch in enumerate(train_batch):
            start_time = time.time()
            global_step += 1
            loss = trainer.update(batch, eval=False) # update step
            train_loss += loss
            if global_step % args['log_step'] == 0:
                duration = time.time() - start_time
                print(format_str.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), global_step,\
                        max_steps, loss, duration, current_lr))

            if global_step % args['eval_interval'] == 0:
                # eval on dev
                print("Evaluating on dev set...")
                dev_preds = []
                for i, batch in enumerate(dev_batch):
                    preds = trainer.predict(batch)
                    dev_preds += preds
                dev_batch.conll.set(['upos', 'xpos', 'feats'], [y for x in dev_preds for y in x])
                dev_batch.conll.write_conll(system_pred_file)
                _, _, dev_score = scorer.score(system_pred_file, gold_file)

                train_loss = train_loss / args['eval_interval'] # avg loss per batch
                print("step {}: train_loss = {:.6f}, dev_score = {:.4f}".format(global_step, train_loss, dev_score))
                train_loss = 0

                # save best model
                if len(dev_score_history) == 0 or dev_score > max(dev_score_history):
                    last_best_step = global_step
                    trainer.save(model_file)
                    print("new best model saved.")
                    best_dev_preds = dev_preds

                dev_score_history += [dev_score]
                print("")

            if global_step - last_best_step >= args['max_steps_before_stop']:
                if not using_amsgrad:
                    print("Switching to AMSGrad")
                    last_best_step = global_step
                    using_amsgrad = True
                    trainer.optimizer = optim.Adam(trainer.model.parameters(), amsgrad=True, lr=args['lr'], betas=(.9, args['beta2']), eps=1e-12)
                else:
                    do_break = True
                    break

            if global_step >= args['max_steps']:
                if not using_amsgrad:
                    print("Switching to AMSGrad")
                    last_best_step = global_step
                    using_amsgrad = True
                    trainer.optimizer = optim.Adam(trainer.model.parameters(), amsgrad=True, lr=args['lr'], betas=(.9, args['beta2']), eps=1e-12)
                elif global_step >= args['max_steps'] * 2:
                    do_break = True
                    break

        if do_break: break

        print('Reshuffling training data...')
        train_batch.reshuffle()

    print("Training ended with {} epochs.".format(epoch))

    best_f, best_epoch = max(dev_score_history)*100, np.argmax(dev_score_history)+1
    print("Best dev F1 = {:.2f}, at epoch = {}".format(best_f, best_epoch))

    param_manager.update(args, best_f)

def evaluate(args):
    # load config
    config_file = '{}/{}_config.json'.format(args['save_dir'], args['shorthand'])
    loaded_args = utils.load_config(config_file)
    for k in args:
        if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand']:
            loaded_args[k] = args[k]
    loaded_args['cuda'] = args['cuda'] and not args['cpu']
    # load data
    print("Loading data with batch size {}...".format(args['batch_size']))
    batch = DataLoader(args['eval_file'], args['batch_size'], loaded_args, evaluation=True)
    vocab = batch.vocab

    # file paths
    system_pred_file = args['output_file']
    gold_file = args['gold_file']
    model_file = args['save_dir'] + '/' + args['save_name'] if args['save_name'] is not None \
            else '{}/{}_tagger.pt'.format(args['save_dir'], args['shorthand'])

    if len(batch) > 0:
        trainer = Trainer(loaded_args, vocab, batch.pretrained_emb)
        trainer.load(model_file)
        print("Start evaluation...")
        preds = []
        for i, b in enumerate(batch):
            preds += trainer.predict(b)
    else:
        # skip eval if dev data does not exist
        preds = []

    # write to file and score
    batch.conll.set(['upos', 'xpos', 'feats'], [y for x in preds for y in x])
    batch.conll.write_conll(system_pred_file)

    if gold_file is not None:
        _, _, score = scorer.score(system_pred_file, gold_file)

        print("Tagger score:")
        print("{} {:.2f}".format(args['shorthand'], score*100))

if __name__ == '__main__':
    main()
