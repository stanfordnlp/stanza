from bisect import bisect_left
from collections import Counter
from copy import copy
import json
import pickle
import numpy as np
import random
import re
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils.postprocess_vietnamese_tokenizer_data import paras_to_chunks

from models.tokenize.trainer import TokenizerTrainer

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--txt_file', type=str, help="Input plaintext file")
    parser.add_argument('--label_file', type=str, default=None, help="Character-level label file")
    parser.add_argument('--json_file', type=str, default=None, help="JSON file with pre-chunked units")
    parser.add_argument('--mwt_json_file', type=str, default=None, help="JSON file for MWT expansions")
    parser.add_argument('--conll_file', type=str, default=None, help="CoNLL file for output")
    parser.add_argument('--vocab_file', type=str, default=None, help="Vocab file")
    parser.add_argument('--lang', type=str, help="Language")

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])

    parser.add_argument('--emb_dim', type=int, default=30, help="Dimension of unit embeddings")
    parser.add_argument('--hidden_dim', type=int, default=100, help="Dimension of hidden units")
    parser.add_argument('--conv_filters', type=str, default="1,5,9,,1,5,9", help="Configuration of conv filters. ,, separates layers and , separates filter sizes in the same layer.")
    parser.add_argument('--residual', action='store_true', help="Add linear residual connections")
    parser.add_argument('--no-rnn', dest='rnn', action='store_false', help="Use CNN tokenizer")
    parser.add_argument('--input_dropout', action='store_true', help="Dropout input embeddings as well")
    parser.add_argument('--aux_clf', type=float, default=0.0, help="Strength for auxiliary classifiers; default 0 (don't use auxiliary classifiers)")
    parser.add_argument('--merge_aux_clf', action='store_true', help="Merge prediction from auxiliary classifiers with final classifier output")
    parser.add_argument('--conv_res', type=str, default=None, help="Convolutional residual layers for the RNN")
    parser.add_argument('--rnn_layers', type=int, default=1, help="Layers of RNN in the tokenizer")

    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm to clip to")
    parser.add_argument('--anneal', type=float, default=0, help="(Equivalent) frequency to half the learning rate; 0 means no annealing (the default)")
    parser.add_argument('--dropout', type=float, default=0.0, help="Dropout probability")
    parser.add_argument('--unit_dropout', type=float, default=0.0, help="Unit dropout probability")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay")
    parser.add_argument('--max_seqlen', type=int, default=100, help="Maximum sequence length to consider at a time")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size to use")
    parser.add_argument('--epochs', type=int, default=10, help="Total epochs to train the model for")
    parser.add_argument('--steps', type=int, default=None, help="Steps to train the model for, if unspecified use epochs")
    parser.add_argument('--report_steps', type=int, default=20, help="Update step interval to report loss")
    parser.add_argument('--shuffle_steps', type=int, default=0, help="Step interval to shuffle each paragragraph in the generator")
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")
    parser.add_argument('--save_dir', type=str, default='saved_models', help="Directory to save models in")
    parser.add_argument('--no_cuda', dest="cuda", action="store_false")

    args = parser.parse_args()

    args = vars(args)
    args['feat_funcs'] = ['space_before', 'capitalized', 'all_caps', 'numeric']
    args['feat_dim'] = len(args['feat_funcs'])
    args['save_name'] = "{}/{}".format(args['save_dir'], args['save_name']) if args['save_name'] is not None else '{}/{}_tokenizer.pkl'.format(args['save_dir'], args['lang'])
    trainer = TokenizerTrainer(args)

    N = len(trainer.data_generator)
    if args['mode'] == 'train':
        if args['cuda']:
            trainer.model.cuda()
        steps = args['steps'] if args['steps'] is not None else int(N * args['epochs'] / args['batch_size'] + .5)
        lr0 = 2e-3

        for step in range(steps):
            batch = trainer.data_generator.next(trainer.vocab, feat_funcs=trainer.feat_funcs, unit_dropout=args['unit_dropout'])

            loss = trainer.update(batch)
            if step % args['report_steps'] == 0:
                print("Step {:6d}/{:6d} Loss: {:.3f}".format(step, steps, loss))

            if args['shuffle_steps'] > 0 and step % args['shuffle_steps'] == 0:
                trainer.data_generator.shuffle()

            if args['anneal'] > 0:
                trainer.change_lr(lr0 * (.5 ** ((step + 1) / args['anneal'])))

        trainer.save(args['save_name'])
    else:
        trainer.load(args['save_name'])
        if args['cuda']:
            trainer.model.cuda()

        offset = 0
        oov_count = 0

        mwt_dict = None
        if args['mwt_json_file'] is not None:
            with open(args['mwt_json_file'], 'r') as f:
                mwt_dict0 = json.load(f)

            mwt_dict = dict()
            for item in mwt_dict0:
                (key, expansion), count = item

                if key not in mwt_dict or mwt_dict[key][1] < count:
                    mwt_dict[key] = (expansion, count)

        def print_sentence(sentence, f, mwt_dict=None):
            i = 0
            for tok, p in current_sent:
                expansion = None
                if p == 3 and mwt_dict is not None:
                    # MWT found, (attempt to) expand it!
                    if tok in mwt_dict:
                        expansion = mwt_dict[tok][0]
                    elif tok.lower() in mwt_dict:
                        expansion = mwt_dict[tok.lower()][0]
                if expansion is not None:
                    f.write("{}-{}\t{}{}\n".format(i+1, i+len(expansion), tok, "\t_" * 8))
                    for etok in expansion:
                        f.write("{}\t{}{}\t{}{}\n".format(i+1, etok, "\t_" * 4, i, "\t_" * 3))
                        i += 1
                else:
                    if len(tok) <= 0:
                        continue
                    f.write("{}\t{}{}\t{}{}\n".format(i+1, tok, "\t_" * 4, i, "\t_" * 3))
                    i += 1
            f.write('\n')

        with open(args['conll_file'], 'w') as f:
            while True:
                batch = trainer.data_generator.next(trainer.vocab, feat_funcs=trainer.feat_funcs, eval_offset=offset)
                if batch is None:
                    break
                pred = np.argmax(trainer.predict(batch)[0], axis=1)

                current_tok = ''
                current_sent = []

                for t, p in zip(batch[3][0], pred):
                    if t == '<PAD>':
                        break
                    offset += 1
                    if trainer.vocab.unit2id(t) == trainer.vocab.unit2id('<UNK>'):
                        oov_count += 1

                    current_tok += t
                    if p >= 1:
                        current_sent += [(trainer.vocab.normalize_token(current_tok), p)]
                        current_tok = ''
                        if p == 2:
                            print_sentence(current_sent, f, mwt_dict)
                            current_sent = []

                if len(current_tok):
                    current_sent += [(trainer.vocab.normalize_token(current_tok), 2)]

                if len(current_sent):
                    print_sentence(current_sent, f, mwt_dict)

        print("OOV rate: {:6.3f}% ({:6d}/{:6d})".format(oov_count / offset * 100, oov_count, offset))
