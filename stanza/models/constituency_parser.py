"""
A command line interface to a shift reduce constituency parser.

This follows the work of
Recurrent neural network grammars by Dyer et al
In-Order Transition-based Constituent Parsing by Liu & Zhang

The general outline is:

  Train a model by taking a list of trees, converting them to
    transition sequences, and learning a model which can predict the
    next transition given a current state
  Then, at inference time, repeatedly predict the next transition until parsing is complete

In order to make the runtime a more competitive speed, effort is taken
to batch the transitions and apply multiple transitions at once.  At
train time, batches are groups together by length, and at inference
time, new trees are added to the batch as previous trees on the batch
finish their inference.
"""

import argparse
import logging
import os

import torch

from stanza.models.common import utils
from stanza.models.constituency import trainer

logger = logging.getLogger('stanza')

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data/constituency', help='Directory of constituency data.')

    parser.add_argument('--wordvec_dir', type=str, default='extern_data/wordvec', help='Directory of word vectors')
    parser.add_argument('--wordvec_file', type=str, default='', help='File that contains word vectors')
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')
    parser.add_argument('--pretrain_max_vocab', type=int, default=250000)

    parser.add_argument('--tag_embedding_dim', type=int, default=20, help="Embedding size for a tag")
    parser.add_argument('--delta_embedding_dim', type=int, default=100, help="Embedding size for a delta embedding")

    parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--mode', default='train', choices=['train', 'predict'])

    parser.add_argument('--lang', type=str, help='Language')
    parser.add_argument('--shorthand', type=str, help="Treebank shorthand")

    parser.add_argument('--transition_embedding_dim', type=int, default=20, help="Embedding size for a transition")
    parser.add_argument('--transition_hidden_size', type=int, default=20, help="Embedding size for transition stack")
    parser.add_argument('--hidden_size', type=int, default=100, help="Size of the output layers for constituency stack and word queue")

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--train_batch_size', type=int, default=50, help='How many trees to train before taking an optimizer step')
    parser.add_argument('--eval_batch_size', type=int, default=50, help='How many trees to batch when running eval')

    parser.add_argument('--save_dir', type=str, default='saved_models/constituency', help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

    DEFAULT_LEARNING_RATES = { "adamw": 0.001, "adadelta": 1.0, "sgd": 0.001 }
    parser.add_argument('--learning_rate', default=None, type=float, help='Learning rate for the optimizer.  Reasonable values are 1.0 for adadelta or 0.001 for SGD.  None uses a default for the given optimizer: {}'.format(DEFAULT_LEARNING_RATES))
    # When using adadelta, weight_decay of 0.01 to 0.001 had the best results.
    # 0.1 was very clearly too high. 0.0001 might have been okay.
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay (eg, l2 reg) to use in the optimizer')
    parser.add_argument('--optim', default='Adadelta', help='Optimizer type: SGD, AdamW, or Adadelta')

    parser.add_argument('--word_dropout', default=0.0, type=float, help='Dropout on the word embedding')
    parser.add_argument('--predict_dropout', default=0.0, type=float, help='Dropout on the final prediction layer')

    parser.add_argument('--use_compound_unary', default=False, action='store_true', help='Use compound unaries in the transition sequence')
    parser.add_argument('--use_compound_open', default=False, action='store_true', help='Use compound opens in the transition sequence')

    parser.add_argument('--nonlinearity', default='tanh', choices=['tanh', 'relu'], help='Nonlinearity to use in the model')

    parser.add_argument('--rare_word_unknown_frequency', default=0.02, type=float, help='How often to replace a rare word with UNK when training')
    parser.add_argument('--rare_word_threshold', default=0.02, type=float, help='How many words to consider as rare words as a fraction of the dataset')

    parser.add_argument('--num_lstm_layers', default=2, type=int, help='How many layers to use in the LSTMs')
    parser.add_argument('--num_output_layers', default=2, type=int, help='How many layers to use at the prediction level')

    parser.add_argument('--train_method', default='gold_entire', choices=['early_entire', 'gold_entire'], help='Different training methods to use')

    args = parser.parse_args(args=args)
    if not args.lang and args.shorthand and len(args.shorthand.split("_")) == 2:
        args.lang = args.shorthand.split("_")[0]
    if args.cpu:
        args.cuda = False
    if args.learning_rate is None:
        args.learning_rate = DEFAULT_LEARNING_RATES.get(args.optim.lower(), None)

    args = vars(args)
    return args

def main(args=None):
    args = parse_args(args=args)

    utils.set_random_seed(args['seed'], args['cuda'])

    logger.info("Running constituency parser in {} mode".format(args['mode']))
    logger.debug("Using GPU: {}".format(args['cuda']))

    model_file = args['save_name'] if args['save_name'] else '{}_constituency.pt'.format(args['shorthand'])
    model_file = os.path.join(args['save_dir'], model_file)

    if args['mode'] == 'train':
        trainer.train(args, model_file)
    else:
        trainer.evaluate(args, model_file)

if __name__ == '__main__':
    main()
