import argparse
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stanza.models.common import pretrain
from stanza.models.common import utils
from stanza.models.constituency import base_model
from stanza.models.constituency import lstm_model
from stanza.models.constituency import parse_transitions
from stanza.models.constituency import parse_tree
from stanza.models.constituency import transition_sequence
from stanza.models.constituency import tree_reader
from stanza.server.parser_eval import EvaluateParser

tqdm = utils.get_tqdm()

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
    parser.add_argument('--eval_batch_size', type=int, default=10, help='How many trees to batch when running eval')

    parser.add_argument('--save_dir', type=str, default='saved_models/constituency', help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

    parser.add_argument('--learning_rate', default=1.0, type=float, help='Learning rate for the optimizer.  Reasonable values are 1.0 for adadelta or 0.005 for SGD')
    parser.add_argument('--weight_decay', default=0.001, type=float, help='Weight decay (eg, l2 reg) to use in the optimizer')
    parser.add_argument('--optim', default='Adadelta', help='Optimizer type: SGD or Adadelta')

    parser.add_argument('--word_dropout', default=0.0, type=float, help='Dropout on the word embedding')
    parser.add_argument('--predict_dropout', default=0.0, type=float, help='Dropout on the final prediction layer')

    parser.add_argument('--use_compound_unary', default=False, action='store_true', help='Use compound unaries in the transition sequence')
    parser.add_argument('--use_compound_open', default=False, action='store_true', help='Use compound opens in the transition sequence')

    parser.add_argument('--nonlinearity', default='tanh', choices=['tanh', 'relu'], help='Nonlinearity to use in the model')

    parser.add_argument('--rare_word_unknown_frequency', default=0.02, type=float, help='How often to replace a rare word with UNK when training')
    parser.add_argument('--rare_word_threshold', default=0.02, type=float, help='How many words to consider as rare words as a fraction of the dataset')

    parser.add_argument('--num_lstm_layers', default=1, type=int, help='How many layers to use in the LSTMs')

    parser.add_argument('--train_method', default='gold_entire', choices=['early_entire', 'gold_entire'], help='Different training methods to use')

    args = parser.parse_args(args=args)
    if not args.lang and args.shorthand and len(args.shorthand.split("_")) == 2:
        args.lang = args.shorthand.split("_")[0]
    if args.cpu:
        args.cuda = False
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
        train(args, model_file)
    else:
        evaluate(args, model_file)

def load_pretrain(args):
    pretrain_file = pretrain.find_pretrain_file(args['wordvec_pretrain_file'], args['save_dir'], args['shorthand'], args['lang'])
    if os.path.exists(pretrain_file):
        vec_file = None
    else:
        vec_file = args['wordvec_file'] if args['wordvec_file'] else utils.get_wordvec_file(args['wordvec_dir'], args['shorthand'])
    pt = pretrain.Pretrain(pretrain_file, vec_file, args['pretrain_max_vocab'])
    return pt

def read_treebank(filename):
    """
    Read a treebank and alter the trees to be a simpler format for learning to parse
    """
    logger.info("Reading trees from {}".format(filename))
    trees = tree_reader.read_tree_file(filename)
    trees = [t.prune_none().simplify_labels() for t in trees]
    return trees

def verify_transitions(trees, sequences):
    model = base_model.SimpleModel()
    logger.info("Verifying the transition sequences for {} trees".format(len(trees)))
    for tree, sequence in tqdm(zip(trees, sequences), total=len(trees)):
        state = parse_transitions.initial_state_from_gold_tree(tree, model)
        for trans in sequence:
            state = trans.apply(state, model)
        result = model.get_top_constituent(state.constituents)
        if tree != result:
            raise RuntimeError("Transition sequence did not match for a tree!\nOriginal tree:{}\nTransitions: {}\nResult tree:{}".format(tree, sequence, result))

def evaluate(args, model_file):
    pretrain = load_pretrain(args)
    model = lstm_model.load(model_file, pretrain)

    treebank = read_treebank(args['eval_file'])
    logger.info("Read {} trees for evaluation".format(len(treebank)))

    f1 = run_dev_set(model, treebank, args['eval_batch_size'])
    logger.info("F1 score on {}: {}".format(args['eval_file'], f1))

def build_treebank(trees, args):
    return transition_sequence.build_top_down_treebank(trees, use_compound_unary=args['use_compound_unary'], use_compound_open=args['use_compound_open'])

def get_open_nodes(trees, args):
    if args['use_compound_open']:
        return parse_tree.Tree.get_compound_constituents(trees)
    else:
        return [(x,) for x in parse_tree.Tree.get_unique_constituent_labels(trees)]

def print_args(args):
    """
    For record keeping purposes, print out the arguments when training
    """
    keys = sorted(args.keys())
    log_lines = ['%s: %s' % (k, args[k]) for k in keys]
    logger.info('ARGS USED AT TRAINING TIME:\n%s\n' % '\n'.join(log_lines))

def train(args, model_file):
    print_args(args)

    utils.ensure_dir(args['save_dir'])

    train_trees = read_treebank(args['train_file'])
    logger.info("Read {} trees for the training set".format(len(train_trees)))

    dev_trees = read_treebank(args['eval_file'])
    logger.info("Read {} trees for the dev set".format(len(dev_trees)))

    train_constituents = parse_tree.Tree.get_unique_constituent_labels(train_trees)
    dev_constituents = parse_tree.Tree.get_unique_constituent_labels(dev_trees)
    logger.info("Unique constituents in training set: {}".format(train_constituents))
    for con in dev_constituents:
        if con not in train_constituents:
            raise RuntimeError("Found label {} in the dev set which don't exist in the train set".format(con))

    logger.info("Building training transition sequences")
    train_sequences = build_treebank(tqdm(train_trees), args)
    train_transitions = transition_sequence.all_transitions(train_sequences)

    logger.info("Building dev transition sequences")
    dev_sequences = build_treebank(tqdm(dev_trees), args)
    dev_transitions = transition_sequence.all_transitions(dev_sequences)

    logger.info("Total unique transitions in train set: {}".format(len(train_transitions)))
    for trans in dev_transitions:
        if trans not in train_transitions:
            raise RuntimeError("Found transition {} in the dev set which don't exist in the train set".format(trans))

    verify_transitions(train_trees, train_sequences)
    verify_transitions(dev_trees, dev_sequences)

    root_labels = parse_tree.Tree.get_root_labels(train_trees)
    for root_state in parse_tree.Tree.get_root_labels(dev_trees):
        if root_state not in root_labels:
            raise RuntimeError("Found root state {} in the dev set which is not a ROOT state in the train set".format(root_state))

    tags = parse_tree.Tree.get_unique_tags(train_trees)
    for tag in parse_tree.Tree.get_unique_tags(dev_trees):
        if tag not in tags:
            raise RuntimeError("Found tag {} in the dev set which is not a tag in the train set".format(tag))

    # we don't check against the words in the dev set as it is
    # expected there will be some UNK words
    words = parse_tree.Tree.get_unique_words(train_trees)
    rare_words = parse_tree.Tree.get_rare_words(train_trees, args['rare_word_threshold'])
    # also, it's not actually an error if there is a pattern of
    # compound unary or compound open nodes which doesn't exist in the
    # train set.  it just means we probably won't ever get that right
    open_nodes = get_open_nodes(train_trees, args)

    pretrain = load_pretrain(args)

    # at this point we have:
    # pretrain
    # train_trees, dev_trees
    # lists of transitions, internal nodes, and root states the parser needs to be aware of

    # TODO: instead of train_constituents, create with the proper open tags
    model = lstm_model.LSTMModel(pretrain, train_transitions, train_constituents, tags, words, rare_words, root_labels, open_nodes, args)
    if args['cuda']:
        model.cuda()

    iterate_training(model, train_trees, train_sequences, train_transitions, dev_trees, args, model_file)

def iterate_training(model, train_trees, train_sequences, transitions, dev_trees, args, model_file):
    # TODO: try different loss functions and optimizers
    if args['optim'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args['learning_rate'], momentum=0.9, weight_decay=args['weight_decay'])
    elif args['optim'].lower() == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    else:
        raise ValueError("Unknown optimizer: %s" % args.optim)

    loss_function = nn.CrossEntropyLoss(reduction='sum')
    if args['cuda']:
        loss_function.cuda()

    device = next(model.parameters()).device
    transition_tensors = {x: torch.tensor(y, requires_grad=False, device=device).unsqueeze(0)
                          for (y, x) in enumerate(transitions)}

    model.train()

    train_data = list(zip(train_trees, train_sequences))
    leftover_training_data = []
    best_f1 = 0.0
    for epoch in range(args['epochs']):
        model.train()
        logger.info("Starting epoch {}".format(epoch+1))
        epoch_data = leftover_training_data
        while len(epoch_data) < args['eval_interval']:
            random.shuffle(train_data)
            epoch_data.extend(train_data)
        leftover_training_data = epoch_data[args['eval_interval']:]
        epoch_data = epoch_data[:args['eval_interval']]

        epoch_loss = 0.0
        correct = 0
        incorrect = 0
        for step, (tree, sequence) in enumerate(tqdm(epoch_data)):
            # Currently we do fake batching
            # TODO: do a real batch over the trees to speed things up
            if step % args['train_batch_size'] == 0 and step > 0:
                optimizer.step()
                optimizer.zero_grad()

            state = parse_transitions.initial_state_from_gold_tree(tree, model)
            if args['train_method'] in ('gold_entire', 'early_entire'):
                errors = []
                answers = []
                for gold_transition in sequence:
                    outputs, pred_transition = model.predict((state,))
                    trans_tensor = transition_tensors[gold_transition]
                    errors.append(outputs.squeeze())
                    answers.append(trans_tensor)
                    state = gold_transition.apply(state, model)
                    if pred_transition != gold_transition:
                        incorrect = incorrect + 1
                        if args['train_method'] == 'early_entire':
                            break
                    else:
                        correct = correct + 1

                errors = torch.stack(errors)
                answers = torch.cat(answers)
                tree_loss = loss_function(errors, answers)
                tree_loss.backward()
                epoch_loss += tree_loss.item()

        # there will always be leftover, so call step() one more time
        optimizer.step()
        optimizer.zero_grad()

        # print statistics
        f1 = run_dev_set(model, dev_trees, args['eval_batch_size'])
        if f1 > best_f1:
            logger.info("New best dev score: {} > {}".format(f1, best_f1))
            best_f1 = f1
            lstm_model.save(model_file, model)
        logger.info("Epoch {} finished\nTransitions correct: {}  Transitions incorrect: {}\n  Total loss for epoch: {}\n  Dev score: {}\n  Best dev score: {}".format(epoch+1, correct, incorrect, epoch_loss, f1, best_f1))

def run_dev_set(model, dev_trees, batch_size):
    logger.info("Processing {} dev trees".format(len(dev_trees)))
    model.eval()
    treebank = []

    tree_iterator = iter(tqdm(dev_trees))
    tree_batch = []
    for _ in range(batch_size):
        gold_tree = next(tree_iterator, None)
        if gold_tree is None:
            break
        # TODO: can batch the initial_states
        # tree, # transitions, state
        # TODO: could store the number of transitions in the state...
        tree_batch.append((gold_tree, 0, parse_transitions.initial_state_from_gold_tree(gold_tree, model)))

    while len(tree_batch) > 0:
        states = [tree[2] for tree in tree_batch]
        _, transitions = model.predict(states, is_legal=True)

        remove = set()
        for idx, (state, transition) in enumerate(zip(states, transitions)):
            if not transition:
                logger.error("Got stuck and couldn't find a legal transition on the following gold tree:\n{}\n\nFinal state:\n{}".format(tree_batch[idx][0], state.to_string(model)))
                remove.add(idx)
                continue

            # TODO: batch this as well!
            state = transition.apply(state, model)
            tree_batch[idx] = (tree_batch[idx][0], tree_batch[idx][1] + 1, state)
            if tree_batch[idx][1] >= 1000:
                # too many transitions
                logger.error("Went infinite on the following gold tree:\n{}\n\nFinal state:\n{}".format(tree_batch[idx][0], state.to_string(model)))
                remove.add(idx)
                continue

            if state.finished(model):
                predicted_tree = state.get_tree(model)
                gold_tree = tree_batch[idx][0]
                # TODO: put an actual score here?
                treebank.append((gold_tree, [(predicted_tree, 1.0)]))
                remove.add(idx)

        tree_batch = [tree for idx, tree in enumerate(tree_batch) if idx not in remove]
        for _ in range(batch_size - len(tree_batch)):
            gold_tree = next(tree_iterator, None)
            if gold_tree is None:
                break
            tree_batch.append((gold_tree, 0, parse_transitions.initial_state_from_gold_tree(gold_tree, model)))


    if len(treebank) < len(dev_trees):
        logger.warning("Only evaluating {} trees instead of {}".format(len(treebank), len(dev_trees)))

    with EvaluateParser(classpath="$CLASSPATH") as ep:
        response = ep.process(treebank)
        return response.f1


if __name__ == '__main__':
    main()
