"""
Entry point for training and evaluating a lemmatizer.

This lemmatizer combines a neural sequence-to-sequence architecture with an `edit` classifier 
and two dictionaries to produce robust lemmas from word forms.
For details please refer to paper: https://nlp.stanford.edu/pubs/qi2018universal.pdf.
"""

import sys
import os
import shutil
import time
from datetime import datetime
import argparse
import numpy as np
import random
import torch
from torch import nn, optim

from stanfordnlp.models.lemma.data import DataLoader
from stanfordnlp.models.lemma.vocab import Vocab
from stanfordnlp.models.lemma.trainer import Trainer
from stanfordnlp.models.lemma import scorer, edit
from stanfordnlp.models.common import utils
import stanfordnlp.models.common.seq2seq_constant as constant

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/lemma', help='Directory for all lemma data.')
    parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--output_file', type=str, default=None, help='Output CoNLL-U file.')
    parser.add_argument('--gold_file', type=str, default=None, help='Output CoNLL-U file.')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')

    parser.add_argument('--no_dict', dest='ensemble_dict', action='store_false', help='Do not ensemble dictionary with seq2seq. By default use ensemble.')
    parser.add_argument('--dict_only', action='store_true', help='Only train a dictionary-based lemmatizer.')

    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--emb_dim', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--emb_dropout', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--max_dec_len', type=int, default=50)
    parser.add_argument('--beam_size', type=int, default=1)

    parser.add_argument('--attn_type', default='soft', choices=['soft', 'mlp', 'linear', 'deep'], help='Attention type')
    parser.add_argument('--pos', action='store_true', help='Use UPOS in lemmatization.')
    parser.add_argument('--pos_dim', type=int, default=50)
    parser.add_argument('--pos_dropout', type=float, default=0.5)
    parser.add_argument('--no_edit', dest='edit', action='store_false', help='Do not use edit classifier in lemmatization. By default use an edit classifier.')
    parser.add_argument('--num_edit', type=int, default=len(edit.EDIT_TO_ID))
    parser.add_argument('--alpha', type=float, default=1.0)

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--decay_epoch', type=int, default=30, help="Decay the lr starting from this epoch.")
    parser.add_argument('--num_epoch', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--model_dir', type=str, default='saved_models/lemma', help='Root dir for saving models.')

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
    print("Running lemmatizer in {} mode".format(args['mode']))

    # manually correct for training epochs
    if args['lang'] in ['cs_pdt', 'ru_syntagrus']:
        args['num_epoch'] = 30

    if args['mode'] == 'train':
        train(args)
    else:
        evaluate(args)

def train(args):
    # load data
    print("[Loading data with batch size {}...]".format(args['batch_size']))
    train_batch = DataLoader(args['train_file'], args['batch_size'], args, evaluation=False)
    vocab = train_batch.vocab
    args['vocab_size'] = vocab['char'].size
    args['pos_vocab_size'] = vocab['pos'].size
    dev_batch = DataLoader(args['eval_file'], args['batch_size'], args, vocab=vocab, evaluation=True)

    utils.ensure_dir(args['model_dir'])
    model_file = '{}/{}_lemmatizer.pt'.format(args['model_dir'], args['lang'])

    # pred and gold path
    system_pred_file = args['output_file']
    gold_file = args['gold_file']

    utils.print_config(args)

    # skip training if the language does not have training or dev data
    if len(train_batch) == 0 or len(dev_batch) == 0:
        print("[Skip training because no data available...]")
        sys.exit(0)

    # start training
    # train a dictionary-based lemmatizer
    trainer = Trainer(args=args, vocab=vocab, use_cuda=args['cuda'])
    print("[Training dictionary-based lemmatizer...]")
    trainer.train_dict(train_batch.conll.get(['word', 'upos', 'lemma']))
    print("Evaluating on dev set...")
    dev_preds = trainer.predict_dict(dev_batch.conll.get(['word', 'upos']))
    dev_batch.conll.write_conll_with_lemmas(dev_preds, system_pred_file)
    _, _, dev_f = scorer.score(system_pred_file, gold_file)
    print("Dev F1 = {:.2f}".format(dev_f * 100))

    if args.get('dict_only', False):
        # save dictionaries
        trainer.save(model_file)
    else:
        # train a seq2seq model
        print("[Training seq2seq-based lemmatizer...]")
        global_step = 0
        max_steps = len(train_batch) * args['num_epoch']
        dev_score_history = []
        best_dev_preds = []
        current_lr = args['lr']
        global_start_time = time.time()
        format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'

        # start training
        for epoch in range(1, args['num_epoch']+1):
            train_loss = 0
            for i, batch in enumerate(train_batch):
                start_time = time.time()
                global_step += 1
                loss = trainer.update(batch, eval=False) # update step
                train_loss += loss
                if global_step % args['log_step'] == 0:
                    duration = time.time() - start_time
                    print(format_str.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), global_step,\
                            max_steps, epoch, args['num_epoch'], loss, duration, current_lr))

            # eval on dev
            print("Evaluating on dev set...")
            dev_preds = []
            dev_edits = []
            for i, batch in enumerate(dev_batch):
                preds, edits = trainer.predict(batch, args['beam_size'])
                dev_preds += preds
                if edits is not None:
                    dev_edits += edits
            dev_preds = trainer.postprocess(dev_batch.conll.get(['word']), dev_preds, edits=dev_edits)

            # try ensembling with dict if necessary
            if args.get('ensemble_dict', False):
                print("[Ensembling dict with seq2seq model...]")
                dev_preds = trainer.ensemble(dev_batch.conll.get(['word', 'upos']), dev_preds)
            dev_batch.conll.write_conll_with_lemmas(dev_preds, system_pred_file)
            _, _, dev_score = scorer.score(system_pred_file, gold_file)

            train_loss = train_loss / train_batch.num_examples * args['batch_size'] # avg loss per batch
            print("epoch {}: train_loss = {:.6f}, dev_score = {:.4f}".format(epoch, train_loss, dev_score))

            # save best model
            if epoch == 1 or dev_score > max(dev_score_history):
                trainer.save(model_file)
                print("new best model saved.")
                best_dev_preds = dev_preds

            # lr schedule
            if epoch > args['decay_epoch'] and dev_score <= dev_score_history[-1] and \
                    args['optim'] in ['sgd', 'adagrad']:
                current_lr *= args['lr_decay']
                trainer.update_lr(current_lr)

            dev_score_history += [dev_score]
            print("")

        print("Training ended with {} epochs.".format(epoch))

        best_f, best_epoch = max(dev_score_history)*100, np.argmax(dev_score_history)+1
        print("Best dev F1 = {:.2f}, at epoch = {}".format(best_f, best_epoch))

def evaluate(args):
    # file paths
    system_pred_file = args['output_file']
    gold_file = args['gold_file']
    model_file = '{}/{}_lemmatizer.pt'.format(args['model_dir'], args['lang'])

    # load model
    use_cuda = args['cuda'] and not args['cpu']
    trainer = Trainer(model_file=model_file, use_cuda=use_cuda)
    loaded_args, vocab = trainer.args, trainer.vocab

    for k in args:
        if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand']:
            loaded_args[k] = args[k]

    # laod data
    print("Loading data with batch size {}...".format(args['batch_size']))
    batch = DataLoader(args['eval_file'], args['batch_size'], loaded_args, vocab=vocab, evaluation=True)

    # skip eval if dev data does not exist
    if len(batch) == 0:
        print("Skip evaluation because no dev data is available...")
        print("Lemma score:")
        print("{} ".format(args['lang']))
        sys.exit(0)

    dict_preds = trainer.predict_dict(batch.conll.get(['word', 'upos']))

    if loaded_args.get('dict_only', False):
        preds = dict_preds
    else:
        print("Running the seq2seq model...")
        preds = []
        edits = []
        for i, b in enumerate(batch):
            ps, es = trainer.predict(b, args['beam_size'])
            preds += ps
            if es is not None:
                edits += es
        preds = trainer.postprocess(batch.conll.get(['word']), preds, edits=edits)

        if loaded_args.get('ensemble_dict', False):
            print("[Ensembling dict with seq2seq lemmatizer...]")
            preds = trainer.ensemble(batch.conll.get(['word', 'upos']), preds)

    # write to file and score
    batch.conll.write_conll_with_lemmas(preds, system_pred_file)
    if gold_file is not None:
        _, _, score = scorer.score(system_pred_file, gold_file)

        print("Lemma score:")
        print("{} {:.2f}".format(args['lang'], score*100))

if __name__ == '__main__':
    main()
