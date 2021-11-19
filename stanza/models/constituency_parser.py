"""A command line interface to a shift reduce constituency parser.

This follows the work of
Recurrent neural network grammars by Dyer et al
In-Order Transition-based Constituent Parsing by Liu & Zhang

The general outline is:

  Train a model by taking a list of trees, converting them to
    transition sequences, and learning a model which can predict the
    next transition given a current state
  Then, at inference time, repeatedly predict the next transition until parsing is complete

The "transitions" are variations on shift/reduce as per an
intro-to-compilers class.  The idea is that you can treat all of the
words in a sentence as a buffer of tokens, then either "shift" them to
represent a new constituent, or "reduce" one or more constituents to
form a new constituent.

In order to make the runtime a more competitive speed, effort is taken
to batch the transitions and apply multiple transitions at once.  At
train time, batches are groups together by length, and at inference
time, new trees are added to the batch as previous trees on the batch
finish their inference.

There are two minor differences in the model:
  - The word input is a bi-lstm, not a uni-lstm.
    This gave a small increase in accuracy.
  - The combination of several constituents into one constituent is done
    via a single bi-lstm rather than two separate lstms.  This increases
    speed without a noticeable effect on accuracy.

A couple experiments which have been tried with little noticeable impact:
  - Combining constituents using the method in the paper (only a trained
    vector at the start instead of both ends) did not affect results
    and is a little slower
  - Using multiple layers of LSTM hidden state for the input to the final
    classification layers didn't help
  - Initializing Linear layers with He initialization and a positive bias
    (to avoid dead connections) had no noticeable effect on accuracy
    0.8396 on it_turin with the original initialization
    0.8401 and 0.8427 on two runs with updated initialization
    (so maybe a small improvement...)

The code breakdown is as follows:

  this file: main interface for training or evaluating models
  constituency/trainer.py: contains the training & evaluation code

  constituency/parse_tree.py: a data structure for representing a parse tree and utility methods
  constituency/tree_reader.py: a module which can read trees from a string or input file

  constituency/tree_stack.py: a linked list which can branch in
    different directions, which will be useful when implementing beam
    search or a dynamic oracle

  constituency/parse_transitions.py: transitions and a State data structure to store them
  constituency/transition_sequence.py: turns ParseTree objects into
    the transition sequences needed to make them

  constituency/base_model.py: operates on the transitions to turn them in to constituents,
    eventually forming one final parse tree composed of all of the constituents
  constituency/lstm_model.py: adds LSTM features to the constituents to predict what the
    correct transition to make is, allowing for predictions on previously unseen text

  stanza/pipeline/constituency_processor.py: interface between this model and the Pipeline

Some alternate optimizer methods:
  adabelief: https://github.com/juntang-zhuang/Adabelief-Optimizer
  madgrad: https://github.com/facebookresearch/madgrad
"""

import argparse
import logging
import os

import torch

from stanza import Pipeline
from stanza.models.common import utils
from stanza.models.constituency import trainer
from stanza.models.constituency.lstm_model import SentenceBoundary
from stanza.models.constituency.parse_transitions import TransitionScheme

logger = logging.getLogger('stanza')

def parse_args(args=None):
    """
    Adds the arguments for building the con parser

    For the most part, defaults are set to cross-validated values, at least for WSJ
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data/constituency', help='Directory of constituency data.')

    parser.add_argument('--wordvec_dir', type=str, default='extern_data/wordvec', help='Directory of word vectors')
    parser.add_argument('--wordvec_file', type=str, default='', help='File that contains word vectors')
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')
    parser.add_argument('--pretrain_max_vocab', type=int, default=250000)

    # for whatever reason, this feature was not helpful
    parser.add_argument('--charlm_forward_file', type=str, default=None, help="Exact path to use for forward charlm")
    parser.add_argument('--charlm_backward_file', type=str, default=None, help="Exact path to use for backward charlm")
    # BERT helps a lot and actually doesn't slow things down too much
    # for VI, for example, use vinai/phobert-base
    parser.add_argument('--bert_model', type=str, default=None, help="Use an external bert model (requires the transformers package)")
    parser.add_argument('--no_bert_model', dest='bert_model', action="store_const", const=None, help="Don't use bert")

    parser.add_argument('--tag_embedding_dim', type=int, default=20, help="Embedding size for a tag.  0 turns off the feature")
    # Smaller values also seem to work
    # For example, after 700 iterations:
    #   32: 0.9174
    #   50: 0.9183
    #   72: 0.9176
    #  100: 0.9185
    # not a huge difference regardless
    # (these numbers were without retagging)
    parser.add_argument('--delta_embedding_dim', type=int, default=100, help="Embedding size for a delta embedding")

    parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--mode', default='train', choices=['train', 'predict', 'remove_optimizer'])
    parser.add_argument('--predict_dir', type=str, default=".", help='Where to write the predictions during --mode predict.  Pred and orig files will be written - the orig file will be retagged if that is requested.  The orig file is important as the results will be shuffled')
    parser.add_argument('--predict_file', type=str, default=None, help='Base name for writing predictions')

    parser.add_argument('--lang', type=str, help='Language')
    parser.add_argument('--shorthand', type=str, help="Treebank shorthand")

    parser.add_argument('--transition_embedding_dim', type=int, default=20, help="Embedding size for a transition")
    parser.add_argument('--transition_hidden_size', type=int, default=20, help="Embedding size for transition stack")
    # larger was more effective, up to a point
    parser.add_argument('--hidden_size', type=int, default=128, help="Size of the output layers for constituency stack and word queue")

    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--epoch_size', type=int, default=5000, help="Runs this many trees in an 'epoch' instead of going through the training dataset exactly once.  Set to 0 to do the whole training set")

    # 1 seems to be the most effective, but we should cross-validate
    parser.add_argument('--oracle_initial_epoch', type=int, default=1, help="Epoch where we start using the dynamic oracle to let the parser keep going with wrong decisions")
    parser.add_argument('--oracle_frequency', type=float, default=0.8, help="How often to use the oracle vs how often to force the correct transition")
    parser.add_argument('--oracle_forced_errors', type=float, default=0.001, help="Occasionally have the model randomly walk through the state space to try to learn how to recover")

    # 30 is slightly slower than 50, for example, but seems to train a bit better on WSJ
    # earlier version of the model (less accurate overall) had the following results with adadelta:
    #  30: 0.9085
    #  50: 0.9070
    #  75: 0.9010
    # 150: 0.8985
    # as another data point, running a newer version with better constituency lstm behavior had:
    #  30: 0.9111
    #  50: 0.9094
    # checking smaller batch sizes to see how this works, at 135 epochs, the values are
    #  10: 0.8919
    #  20: 0.9072
    #  30: 0.9121
    # obviously these experiments aren't the complete story, but it
    # looks like 30 trees per batch is the best value for WSJ
    # note that these numbers are for adadelta and might not apply
    # to other optimizers
    # eval batch should generally be faster the bigger the batch,
    # up to a point, as it allows for more batching of the LSTM
    # operations and the prediction step
    parser.add_argument('--train_batch_size', type=int, default=30, help='How many trees to train before taking an optimizer step')
    parser.add_argument('--eval_batch_size', type=int, default=50, help='How many trees to batch when running eval')

    parser.add_argument('--save_dir', type=str, default='saved_models/constituency', help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")
    parser.add_argument('--save_latest_name', type=str, default=None, help="Save the latest model here regardless of score.  Useful for restarting training")

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

    # Numbers are on a VLSP dataset, before adding attn or other improvements
    # baseline is an 80.6 model that occurs when trained using adadelta, lr 1.0
    #
    # adabelief 0.1:      fails horribly
    #           0.02:     converges very low scores
    #           0.01:     very slow learning
    #           0.002:    almost decent
    #           0.001:    close, but about 1 f1 low on IT
    #           0.0005:   79.71
    #           0.0002:   80.11
    #           0.0001:   79.85
    #           0.00005:  80.40
    #           0.00002:  80.02
    #           0.00001:  78.95
    # madgrad   0.005:    fails horribly
    #           0.001:    low scores
    #           0.0005:   still somewhat low
    #           0.0002:   close, but about 1 f1 low on IT
    #           0.0001:   80.04
    #           0.00005:  79.91
    #           0.00002:  80.15
    #           0.00001:  80.44
    #           0.000005: 80.34
    #           0.000002: 80.39
    DEFAULT_LEARNING_RATES = { "adamw": 0.001, "adadelta": 1.0, "sgd": 0.001, "adabelief": 0.01, "madgrad": 0.005 }
    parser.add_argument('--learning_rate', default=None, type=float, help='Learning rate for the optimizer.  Reasonable values are 1.0 for adadelta or 0.001 for SGD.  None uses a default for the given optimizer: {}'.format(DEFAULT_LEARNING_RATES))
    DEFAULT_LEARNING_EPS = { "adabelief": 1e-12 }
    parser.add_argument('--learning_eps', default=None, type=float, help='eps value to use in the optimizer.  None uses a default for the given optimizer: {}'.format(DEFAULT_LEARNING_RATES))
    # When using adadelta, weight_decay of 0.01 to 0.001 had the best results.
    # 0.1 was very clearly too high. 0.0001 might have been okay.
    # weight decay values other than adadelta have not been thoroughly tested.
    DEFAULT_WEIGHT_DECAY = { "adamw": 0.01, "adadelta": 0.01, "sgd": 0.01, "adabelief": 1.2e-6, "madgrad": 1e-6 }
    parser.add_argument('--weight_decay', default=None, type=float, help='Weight decay (eg, l2 reg) to use in the optimizer')
    parser.add_argument('--optim', default='Adadelta', help='Optimizer type: SGD, AdamW, Adadelta, AdaBelief')

    # When using word_dropout and predict_dropout in conjunction with relu, one particular experiment produced the following dev scores after 300 iterations:
    # 0.0: 0.9085
    # 0.2: 0.9165
    # 0.4: 0.9162
    # 0.5: 0.9123
    # Letting 0.2 and 0.4 run for longer, along with 0.3 as another
    # trial, continued to give extremely similar results over time.
    # No attempt has been made to test the different dropouts separately...
    parser.add_argument('--word_dropout', default=0.2, type=float, help='Dropout on the word embedding')
    parser.add_argument('--predict_dropout', default=0.2, type=float, help='Dropout on the final prediction layer')
    # lstm_dropout has not been fully tested yet
    # one experiment after 200 iterations (after retagging, so scores are lower than some other experiments):
    # 0.0: 0.9093
    # 0.1: 0.9094
    # 0.2: 0.9094
    # 0.3: 0.9076
    # 0.4: 0.9077
    parser.add_argument('--lstm_layer_dropout', default=0.0, type=float, help='Dropout in the LSTM layers')
    # one not very conclusive experiment (not long enough) came up with these numbers after ~200 iterations
    # 0.0       0.9091
    # 0.1       0.9095
    # 0.2       0.9118
    # 0.3       0.9123
    # 0.4       0.9080
    parser.add_argument('--lstm_input_dropout', default=0.2, type=float, help='Dropout on the input to an LSTM')

    parser.add_argument('--transition_scheme', default=TransitionScheme.IN_ORDER, type=lambda x: TransitionScheme[x.upper()],
                        help='Transition scheme to use.  {}'.format(", ".join(x.name for x in TransitionScheme)))

    parser.add_argument('--constituency_lstm', default=False, action='store_true', help="Build constituents using the full LSTM instead of just the nodes below the new constituent.  Doesn't match the original papers and might be slightly less effective")

    # combining dummy and open node embeddings might be a slight improvement
    # for example, after 550 iterations, one experiment had
    # True:     0.9154
    # False:    0.9150
    # another (with a different structure) had 850 iterations
    # True:     0.9155
    # False:    0.9149
    parser.add_argument('--combined_dummy_embedding', default=True, action='store_true', help="Use the same embedding for dummy nodes and the vectors used when combining constituents")
    parser.add_argument('--no_combined_dummy_embedding', dest='combined_dummy_embedding', action='store_false', help="Don't use the same embedding for dummy nodes and the vectors used when combining constituents")

    # relu gave at least 1 F1 improvement over tanh in various experiments
    # relu & gelu seem roughly the same, but relu is clearly faster.
    # relu, 496 iterations: 0.9176
    # gelu, 467 iterations: 0.9181
    # after the same clock time on the same hardware.  the two had been
    # trading places in terms of accuracy over those ~500 iterations.
    # leaky_relu was not an improvement - a full run on WSJ led to 0.9181 f1 instead of 0.919
    parser.add_argument('--nonlinearity', default='relu', choices=['tanh', 'relu', 'gelu', 'leaky_relu'], help='Nonlinearity to use in the model.  relu is a noticeable improvement')

    parser.add_argument('--rare_word_unknown_frequency', default=0.02, type=float, help='How often to replace a rare word with UNK when training')
    parser.add_argument('--rare_word_threshold', default=0.02, type=float, help='How many words to consider as rare words as a fraction of the dataset')
    parser.add_argument('--tag_unknown_frequency', default=0.001, type=float, help='How often to replace a tag with UNK when training')

    parser.add_argument('--num_lstm_layers', default=2, type=int, help='How many layers to use in the LSTMs')
    parser.add_argument('--num_output_layers', default=3, type=int, help='How many layers to use at the prediction level')

    parser.add_argument('--sentence_boundary_vectors', default=SentenceBoundary.EVERYTHING, type=lambda x: SentenceBoundary[x.upper()],
                        help='Vectors to learn at the start & end of sentences.  {}'.format(", ".join(x.name for x in SentenceBoundary)))

    parser.add_argument('--finetune', action='store_true', help='Load existing model during `train` mode from `load_name` path')
    parser.add_argument('--maybe_finetune', action='store_true', help='Load existing model during `train` mode from `load_name` path if it exists.  Useful for running in situations where a job is frequently being preempted')
    parser.add_argument('--load_name', type=str, default=None, help='Model to load when finetuning, evaluating, or manipulating an existing file')

    parser.add_argument('--retag_package', default="default", help='Which tagger shortname to use when retagging trees.  None for no retagging.  Retagging is recommended, as gold tags will not be available at pipeline time')
    parser.add_argument('--retag_method', default='xpos', choices=['xpos', 'upos'], help='Which tags to use when retagging')
    parser.add_argument('--no_retag', dest='retag_package', action="store_const", const=None, help="Don't retag the trees")

    args = parser.parse_args(args=args)
    if not args.lang and args.shorthand and len(args.shorthand.split("_")) == 2:
        args.lang = args.shorthand.split("_")[0]
    if args.cpu:
        args.cuda = False
    if args.learning_rate is None:
        args.learning_rate = DEFAULT_LEARNING_RATES.get(args.optim.lower(), None)
    if args.learning_eps is None:
        args.learning_eps = DEFAULT_LEARNING_EPS.get(args.optim.lower(), None)
    if args.weight_decay is None:
        args.weight_decay = DEFAULT_WEIGHT_DECAY.get(args.optim.lower(), None)

    args = vars(args)

    if args['retag_method'] == 'xpos':
        args['retag_xpos'] = True
    elif args['retag_method'] == 'upos':
        args['retag_xpos'] = False
    else:
        raise ValueError("Unknown retag method {}".format(xpos))

    return args

def main(args=None):
    """
    Main function for building con parser

    Processes args, calls the appropriate function for the chosen --mode
    """
    args = parse_args(args=args)

    utils.set_random_seed(args['seed'], args['cuda'])

    logger.info("Running constituency parser in %s mode", args['mode'])
    logger.debug("Using GPU: %s", args['cuda'])

    model_save_file = args['save_name'] if args['save_name'] else '{}_constituency.pt'.format(args['shorthand'])
    model_save_file = os.path.join(args['save_dir'], model_save_file)

    model_save_latest_file = None
    if args['save_latest_name']:
        model_save_latest_file = os.path.join(args['save_dir'], args['save_latest_name'])

    model_load_file = model_save_file
    if args['load_name']:
        model_load_file = os.path.join(args['save_dir'], args['load_name'])
    elif args['mode'] == 'train' and args['save_latest_name']:
        model_load_file = model_save_latest_file

    if args['retag_package'] is not None:
        if '_' in args['retag_package']:
            lang, package = args['retag_package'].split('_', 1)
        else:
            lang = args['lang']
            package = args['retag_package']
        retag_pipeline = Pipeline(lang=lang, processors="tokenize, pos", tokenize_pretokenized=True, pos_package=package, pos_tqdm=True)
    else:
        retag_pipeline = None

    if args['mode'] == 'train':
        trainer.train(args, model_save_file, model_load_file, model_save_latest_file, retag_pipeline)
    elif args['mode'] == 'predict':
        trainer.evaluate(args, model_load_file, retag_pipeline)
    elif args['mode'] == 'remove_optimizer':
        trainer.remove_optimizer(args, model_save_file, model_load_file)

if __name__ == '__main__':
    main()
