"""
Entry point for training and evaluating a neural tokenizer.

This tokenizer treats tokenization and sentence segmentation as a tagging problem, and uses a combination of 
recurrent and convolutional architectures.
For details please refer to paper: https://nlp.stanford.edu/pubs/qi2018universal.pdf.
"""

import random
import argparse
from copy import copy
import numpy as np
import torch

from stanfordnlp.models.common import utils
from stanfordnlp.models.tokenize.trainer import Trainer
from stanfordnlp.models.tokenize.data import DataLoader
from stanfordnlp.models.tokenize.utils import load_mwt_dict, eval_model, output_predictions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_file', type=str, help="Input plaintext file")
    parser.add_argument('--label_file', type=str, default=None, help="Character-level label file")
    parser.add_argument('--json_file', type=str, default=None, help="JSON file with pre-chunked units")
    parser.add_argument('--mwt_json_file', type=str, default=None, help="JSON file for MWT expansions")
    parser.add_argument('--conll_file', type=str, default=None, help="CoNLL file for output")
    parser.add_argument('--dev_txt_file', type=str, help="(Train only) Input plaintext file for the dev set")
    parser.add_argument('--dev_label_file', type=str, default=None, help="(Train only) Character-level label file for the dev set")
    parser.add_argument('--dev_json_file', type=str, default=None, help="(Train only) JSON file with pre-chunked units for the dev set")
    parser.add_argument('--dev_conll_gold', type=str, default=None, help="(Train only) CoNLL-U file for the dev set for early stopping")
    parser.add_argument('--lang', type=str, help="Language")
    parser.add_argument('--shorthand', type=str, help="UD treebank shorthand")

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])

    parser.add_argument('--emb_dim', type=int, default=32, help="Dimension of unit embeddings")
    parser.add_argument('--hidden_dim', type=int, default=64, help="Dimension of hidden units")
    parser.add_argument('--conv_filters', type=str, default="1,9", help="Configuration of conv filters. ,, separates layers and , separates filter sizes in the same layer.")
    parser.add_argument('--no-residual', dest='residual', action='store_false', help="Add linear residual connections")
    parser.add_argument('--no-hierarchical', dest='hierarchical', action='store_false', help="\"Hierarchical\" RNN tokenizer")
    parser.add_argument('--hier_invtemp', type=float, default=0.5, help="Inverse temperature used in propagating tokenization predictions between RNN layers")
    parser.add_argument('--input_dropout', action='store_true', help="Dropout input embeddings as well")
    parser.add_argument('--conv_res', type=str, default=None, help="Convolutional residual layers for the RNN")
    parser.add_argument('--rnn_layers', type=int, default=1, help="Layers of RNN in the tokenizer")

    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm to clip to")
    parser.add_argument('--anneal', type=float, default=.999, help="Anneal the learning rate by this amount when dev performance deteriorate")
    parser.add_argument('--anneal_after', type=int, default=2000, help="Anneal the learning rate no earlier than this step")
    parser.add_argument('--lr0', type=float, default=2e-3, help="Initial learning rate")
    parser.add_argument('--dropout', type=float, default=0.33, help="Dropout probability")
    parser.add_argument('--unit_dropout', type=float, default=0.33, help="Unit dropout probability")
    parser.add_argument('--tok_noise', type=float, default=0.02, help="Probability to induce noise to the input of the higher RNN")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay")
    parser.add_argument('--max_seqlen', type=int, default=100, help="Maximum sequence length to consider at a time")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size to use")
    parser.add_argument('--epochs', type=int, default=10, help="Total epochs to train the model for")
    parser.add_argument('--steps', type=int, default=20000, help="Steps to train the model for, if unspecified use epochs")
    parser.add_argument('--report_steps', type=int, default=20, help="Update step interval to report loss")
    parser.add_argument('--shuffle_steps', type=int, default=100, help="Step interval to shuffle each paragragraph in the generator")
    parser.add_argument('--eval_steps', type=int, default=200, help="Step interval to evaluate the model on the dev set for early stopping")
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")
    parser.add_argument('--load_name', type=str, default=None, help="File name to load a saved model")
    parser.add_argument('--save_dir', type=str, default='saved_models/tokenize', help="Directory to save models in")
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA and run on CPU.')
    parser.add_argument('--seed', type=int, default=1234)
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
    print("Running tokenizer in {} mode".format(args['mode']))

    args['feat_funcs'] = ['space_before', 'capitalized', 'all_caps', 'numeric']
    args['feat_dim'] = len(args['feat_funcs'])
    args['save_name'] = "{}/{}".format(args['save_dir'], args['save_name']) if args['save_name'] is not None \
            else '{}/{}_tokenizer.pt'.format(args['save_dir'], args['shorthand'])
    utils.ensure_dir(args['save_dir'])

    if args['mode'] == 'train':
        train(args)
    else:
        evaluate(args)

def train(args):
    mwt_dict = load_mwt_dict(args['mwt_json_file'])

    train_input_files = {
            'json': args['json_file'],
            'txt': args['txt_file'],
            'label': args['label_file']
            }
    train_batches = DataLoader(args, input_files=train_input_files)
    vocab = train_batches.vocab
    args['vocab_size'] = len(vocab)

    dev_input_files = {
            'json': args['dev_json_file'],
            'txt': args['dev_txt_file'],
            'label': args['dev_label_file']
            }
    dev_batches = DataLoader(args, input_files=dev_input_files, vocab=vocab, evaluation=True)

    trainer = Trainer(args=args, vocab=vocab, use_cuda=args['cuda'])

    if args['load_name'] is not None:
        load_name = "{}/{}".format(args['save_dir'], args['load_name'])
        trainer.load(load_name)
    trainer.change_lr(args['lr0'])

    N = len(train_batches)
    steps = args['steps'] if args['steps'] is not None else int(N * args['epochs'] / args['batch_size'] + .5)
    lr = args['lr0']

    prev_dev_score = -1
    best_dev_score = -1
    best_dev_step = -1

    for step in range(1, steps+1):
        batch = train_batches.next(unit_dropout=args['unit_dropout'])

        loss = trainer.update(batch)
        if step % args['report_steps'] == 0:
            print("Step {:6d}/{:6d} Loss: {:.3f}".format(step, steps, loss))

        if args['shuffle_steps'] > 0 and step % args['shuffle_steps'] == 0:
            train_batches.shuffle()

        if step % args['eval_steps'] == 0:
            dev_score = eval_model(args, trainer, dev_batches, vocab, mwt_dict)
            reports = ['Dev score: {:6.3f}'.format(dev_score * 100)]
            if step >= args['anneal_after'] and dev_score < prev_dev_score:
                reports += ['lr: {:.6f} -> {:.6f}'.format(lr, lr * args['anneal'])]
                lr *= args['anneal']
                trainer.change_lr(lr)

            prev_dev_score = dev_score

            if dev_score > best_dev_score:
                reports += ['New best dev score!']
                best_dev_score = dev_score
                best_dev_step = step
                trainer.save(args['save_name'])
            print('\t'.join(reports))

    print('Best dev score={} at step {}'.format(best_dev_score, best_dev_step))

def evaluate(args):
    mwt_dict = load_mwt_dict(args['mwt_json_file'])
    use_cuda = args['cuda'] and not args['cpu']
    trainer = Trainer(model_file=args['save_name'], use_cuda=use_cuda)
    loaded_args, vocab = trainer.args, trainer.vocab
    for k in loaded_args:
        if not k.endswith('_file') and k not in ['cuda', 'mode', 'save_dir', 'save_name']:
            args[k] = loaded_args[k]

    eval_input_files = {
            'json': args['json_file'],
            'txt': args['txt_file'],
            'label': args['label_file']
            }

    batches = DataLoader(args, input_files=eval_input_files, vocab=vocab, evaluation=True)

    with open(args['conll_file'], 'w') as conll_output_file:
        oov_count, N, _ = output_predictions(conll_output_file, trainer, batches, vocab, mwt_dict, args['max_seqlen'])

    print("OOV rate: {:6.3f}% ({:6d}/{:6d})".format(oov_count / N * 100, oov_count, N))


if __name__ == '__main__':
    main()
