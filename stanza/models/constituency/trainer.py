"""
This file includes a variety of methods needed to train new
constituency parsers.  It also includes a method to load an
already-trained parser.

See the `train` method for the code block which starts from
  raw treebank and returns a new parser.
`evaluate` reads a treebank and gives a score for those trees.
`parse_tagged_words` is useful at Pipeline time -
  it takes words & tags and processes that into trees.
"""

import logging
import random
import os

import torch
from torch import nn
from torch import optim

from stanza.models.common import pretrain
from stanza.models.common import utils
from stanza.models.common.char_model import CharacterLanguageModel
from stanza.models.constituency import base_model
from stanza.models.constituency import parse_transitions
from stanza.models.constituency import parse_tree
from stanza.models.constituency import transition_sequence
from stanza.models.constituency import tree_reader
from stanza.models.constituency.lstm_model import LSTMModel
from stanza.models.constituency.parse_transitions import State, TransitionScheme
from stanza.models.constituency.utils import retag_trees
from stanza.server.parser_eval import EvaluateParser

tqdm = utils.get_tqdm()

logger = logging.getLogger('stanza.constituency.trainer')

class Trainer:
    """
    Stores a constituency model and its optimizer

    Not inheriting from common/trainer.py because there's no concept of change_lr (yet?)
    """
    def __init__(self, model=None, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def save(self, filename, save_optimizer=True):
        """
        Save the model (and by default the optimizer) to the given path
        """
        params = self.model.get_params()
        checkpoint = {
            'params': params,
            'model_type': 'LSTM',
        }
        if save_optimizer and self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(checkpoint, filename, _use_new_zipfile_serialization=False)
        logger.info("Model saved to %s", filename)


    @staticmethod
    def load(filename, pt, forward_charlm, backward_charlm, use_gpu, args=None, load_optimizer=False):
        """
        Load back a model and possibly its optimizer.

        pt: a Pretrain word embedding
        """
        if args is None:
            args = {}

        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            logger.exception("Cannot load model from %s", filename)
            raise
        logger.debug("Loaded model from %s", filename)

        model_type = checkpoint['model_type']
        params = checkpoint.get('params', checkpoint)

        if model_type == 'LSTM':
            model = LSTMModel(pretrain=pt,
                              forward_charlm=forward_charlm,
                              backward_charlm=backward_charlm,
                              transitions=params['transitions'],
                              constituents=params['constituents'],
                              tags=params['tags'],
                              words=params['words'],
                              rare_words=params['rare_words'],
                              root_labels=params['root_labels'],
                              open_nodes=params['open_nodes'],
                              args=params['config'])
        else:
            raise ValueError("Unknown model type {}".format(model_type))
        model.load_state_dict(params['model'], strict=False)

        if use_gpu:
            model.cuda()

        if load_optimizer:
            optimizer_args = dict(params['config'])
            optimizer_args.update(args)
            optimizer = build_optimizer(optimizer_args, model)

            if checkpoint.get('optimizer_state_dict', None) is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                logger.info("Attempted to load optimizer to resume training, but optimizer not saved.  Creating new optimizer")
        else:
            optimizer = None

        logger.debug("-- MODEL CONFIG --")
        for k in model.args.keys():
            logger.debug("  --%s: %s", k, model.args[k])

        return Trainer(model=model, optimizer=optimizer)


def build_optimizer(args, model):
    """
    Build an optimizer based on the arguments given
    """
    if args['optim'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args['learning_rate'], momentum=0.9, weight_decay=args['weight_decay'])
    elif args['optim'].lower() == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    elif args['optim'].lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    else:
        raise ValueError("Unknown optimizer: %s" % args.optim)
    return optimizer

def load_pretrain(args):
    """
    Loads a pretrain based on the paths in the arguments
    """
    pretrain_file = pretrain.find_pretrain_file(args['wordvec_pretrain_file'], args['save_dir'], args['shorthand'], args['lang'])
    if os.path.exists(pretrain_file):
        vec_file = None
    else:
        vec_file = args['wordvec_file'] if args['wordvec_file'] else utils.get_wordvec_file(args['wordvec_dir'], args['shorthand'])
    pt = pretrain.Pretrain(pretrain_file, vec_file, args['pretrain_max_vocab'])
    return pt

def load_charlm(charlm_file):
    if charlm_file:
        logger.debug("Loading charlm from %s", charlm_file)
        return CharacterLanguageModel.load(charlm_file, finetune=False)
    return None

def read_treebank(filename):
    """
    Read a treebank and alter the trees to be a simpler format for learning to parse
    """
    logger.info("Reading trees from %s", filename)
    trees = tree_reader.read_tree_file(filename)
    trees = [t.prune_none().simplify_labels() for t in trees]

    illegal_trees = [t for t in trees if len(t.children) > 1]
    if len(illegal_trees) > 0:
        raise ValueError("Found {} tree(s) which had non-unary transitions at the ROOT.  First illegal tree: {}".format(len(illegal_trees), illegal_trees[0]))

    return trees

def verify_transitions(trees, sequences, transition_scheme):
    """
    Given a list of trees and their transition sequences, verify that the sequences rebuild the trees
    """
    model = base_model.SimpleModel(transition_scheme)
    logger.info("Verifying the transition sequences for %d trees", len(trees))

    data = zip(trees, sequences)
    if logger.getEffectiveLevel() <= logging.INFO:
        data = tqdm(zip(trees, sequences), total=len(trees))

    for tree, sequence in data:
        state = parse_transitions.initial_state_from_gold_trees([tree], model)[0]
        for idx, trans in enumerate(sequence):
            if not trans.is_legal(state, model):
                raise RuntimeError("Transition {}:{} was not legal in a transition sequence:\nOriginal tree: {}\nTransitions: {}".format(idx, trans, tree, sequence))
            state = trans.apply(state, model)
        result = model.get_top_constituent(state.constituents)
        if tree != result:
            raise RuntimeError("Transition sequence did not match for a tree!\nOriginal tree:{}\nTransitions: {}\nResult tree:{}".format(tree, sequence, result))

def evaluate(args, model_file, retag_pipeline):
    """
    Loads the given model file and tests the eval_file treebank.

    May retag the trees using retag_pipeline
    Uses a subprocess to run the Java EvalB code
    """
    pt = load_pretrain(args)
    forward_charlm = load_charlm(args['charlm_forward_file'])
    backward_charlm = load_charlm(args['charlm_backward_file'])
    trainer = Trainer.load(model_file, pt, forward_charlm, backward_charlm, args['cuda'])

    treebank = read_treebank(args['eval_file'])
    logger.info("Read %d trees for evaluation", len(treebank))

    if retag_pipeline is not None:
        logger.info("Retagging trees using the %s tags from the %s package...", args['retag_method'], args['retag_package'])
        treebank = retag_trees(treebank, retag_pipeline, args['retag_xpos'])
        logger.info("Retagging finished")

    f1 = run_dev_set(trainer.model, treebank, args)
    logger.info("F1 score on %s: %f", args['eval_file'], f1)

def build_treebank(trees, transition_scheme):
    """
    Convert a set of trees into the corresponding treebank based on the args

    Currently only supports top-down transitions, but more may be added in the future, especially bottom up
    """
    return transition_sequence.build_treebank(trees, transition_scheme=transition_scheme)

def get_open_nodes(trees, args):
    """
    Return a list of all open nodes in the given dataset.
    Depending on the parameters, may be single or compound open transitions.
    """
    if args['transition_scheme'] is TransitionScheme.TOP_DOWN_COMPOUND:
        return parse_tree.Tree.get_compound_constituents(trees)
    else:
        return [(x,) for x in parse_tree.Tree.get_unique_constituent_labels(trees)]

def print_args(args):
    """
    For record keeping purposes, print out the arguments when training
    """
    keys = sorted(args.keys())
    log_lines = ['%s: %s' % (k, args[k]) for k in keys]
    logger.info('ARGS USED AT TRAINING TIME:\n%s\n', '\n'.join(log_lines))

def remove_optimizer(args, model_save_file, model_load_file):
    """
    A utility method to remove the optimizer from a save file

    Will make the save file a lot smaller
    """
    # TODO: kind of overkill to load in the pretrain rather than
    # change the load/save to work without it, but probably this
    # functionality isn't used that often anyway
    pt = load_pretrain(args)
    forward_charlm = load_charlm(args['charlm_forward_file'])
    backward_charlm = load_charlm(args['charlm_backward_file'])
    trainer = Trainer.load(model_load_file, pt, forward_charlm, backward_charlm, use_gpu=False, load_optimizer=False)
    trainer.save(model_save_file)

def convert_trees_to_sequences(trees, tree_type, transition_scheme):
    logger.info("Building {} transition sequences".format(tree_type))
    if logger.getEffectiveLevel() <= logging.INFO:
        trees = tqdm(trees)
    sequences = build_treebank(trees, transition_scheme)
    transitions = transition_sequence.all_transitions(sequences)
    return sequences, transitions

def build_trainer(args, train_trees, dev_trees, pt, forward_charlm, backward_charlm):
    """
    Builds a Trainer (with model) and the train_sequences and transitions for the given trees.
    """
    train_constituents = parse_tree.Tree.get_unique_constituent_labels(train_trees)
    dev_constituents = parse_tree.Tree.get_unique_constituent_labels(dev_trees)
    logger.info("Unique constituents in training set: %s", train_constituents)
    for con in dev_constituents:
        if con not in train_constituents:
            raise RuntimeError("Found label {} in the dev set which don't exist in the train set".format(con))

    train_sequences, train_transitions = convert_trees_to_sequences(train_trees, "training", args['transition_scheme'])
    dev_sequences, dev_transitions = convert_trees_to_sequences(dev_trees, "dev", args['transition_scheme'])

    logger.info("Total unique transitions in train set: %d", len(train_transitions))
    for trans in dev_transitions:
        if trans not in train_transitions:
            raise RuntimeError("Found transition {} in the dev set which don't exist in the train set".format(trans))

    verify_transitions(train_trees, train_sequences, args['transition_scheme'])
    verify_transitions(dev_trees, dev_sequences, args['transition_scheme'])

    root_labels = parse_tree.Tree.get_root_labels(train_trees)
    for root_state in parse_tree.Tree.get_root_labels(dev_trees):
        if root_state not in root_labels:
            raise RuntimeError("Found root state {} in the dev set which is not a ROOT state in the train set".format(root_state))

    tags = parse_tree.Tree.get_unique_tags(train_trees)
    logger.info("Unique tags in training set: %s", tags)
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

    # at this point we have:
    # pretrain
    # train_trees, dev_trees
    # lists of transitions, internal nodes, and root states the parser needs to be aware of

    if args['finetune'] or (args['maybe_finetune'] and os.path.exists(model_load_file)):
        logger.info("Loading model to continue training from %s", model_load_file)
        trainer = Trainer.load(model_load_file, pt, forward_charlm, backward_charlm, args['cuda'], args, load_optimizer=True)
    else:
        model = LSTMModel(pt, forward_charlm, backward_charlm, train_transitions, train_constituents, tags, words, rare_words, root_labels, open_nodes, args)
        if args['cuda']:
            model.cuda()

        optimizer = build_optimizer(args, model)

        trainer = Trainer(model, optimizer)

    return trainer, train_sequences, train_transitions

def train(args, model_save_file, model_load_file, model_save_latest_file, retag_pipeline):
    """
    Build a model, train it using the requested train & dev files
    """
    print_args(args)

    utils.ensure_dir(args['save_dir'])

    train_trees = read_treebank(args['train_file'])
    logger.info("Read %d trees for the training set", len(train_trees))

    dev_trees = read_treebank(args['eval_file'])
    logger.info("Read %d trees for the dev set", len(dev_trees))

    if retag_pipeline is not None:
        logger.info("Retagging trees using the %s tags from the %s package...", args['retag_method'], args['retag_package'])
        train_trees = retag_trees(train_trees, retag_pipeline, args['retag_xpos'])
        dev_trees = retag_trees(dev_trees, retag_pipeline, args['retag_xpos'])
        logger.info("Retagging finished")

    pt = load_pretrain(args)
    forward_charlm = load_charlm(args['charlm_forward_file'])
    backward_charlm = load_charlm(args['charlm_backward_file'])

    trainer, train_sequences, train_transitions = build_trainer(args, train_trees, dev_trees, pt, forward_charlm, backward_charlm)

    iterate_training(trainer, train_trees, train_sequences, train_transitions, dev_trees, args, model_save_file, model_save_latest_file)


def iterate_training(trainer, train_trees, train_sequences, transitions, dev_trees, args, model_filename, model_latest_filename):
    """
    Given an initialized model, a processed dataset, and a secondary dev dataset, train the model

    The training is iterated in the following loop:
      extract a batch of trees of the same length from the training set
      convert those trees into initial parsing states
      repeat until trees are done:
        batch predict the model's interpretation of the current states
        add the errors to the list of things to backprop
        advance the parsing state for each of the trees

    Currently the only method implemented for advancing the parsing state
    is to use the gold transition.

    TODO: add a dynamic oracle which can adjust the future expected
    parsing decisions after the parser makes an error.  This way,
    the parser will have "experienced" what the correct decision
    to make is when it gets into incorrect states at runtime.
    """
    model = trainer.model
    optimizer = trainer.optimizer

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
    best_epoch = 0
    for epoch in range(1, args['epochs']+1):
        model.train()
        logger.info("Starting epoch %d", epoch)
        epoch_data = leftover_training_data
        while len(epoch_data) < args['eval_interval']:
            random.shuffle(train_data)
            epoch_data.extend(train_data)
        leftover_training_data = epoch_data[args['eval_interval']:]
        epoch_data = epoch_data[:args['eval_interval']]
        epoch_data.sort(key=lambda x: len(x[1]))
        interval_starts = list(range(0, len(epoch_data), args['train_batch_size']))
        random.shuffle(interval_starts)

        epoch_loss = 0.0

        transitions_correct = 0
        transitions_incorrect = 0

        for interval_start in tqdm(interval_starts, postfix="Batch"):
            batch = epoch_data[interval_start:interval_start+args['train_batch_size']]
            # the batch will be empty when all trees from this epoch are trained
            # now we add the state to the trees in the batch
            initial_states = parse_transitions.initial_state_from_gold_trees([tree for tree, _ in batch], model)
            batch = [state._replace(gold_sequence=sequence)
                     for (tree, sequence), state in zip(batch, initial_states)]

            all_errors = []
            all_answers = []

            while len(batch) > 0:
                outputs, pred_transitions = model.predict(batch)
                gold_transitions = [x.gold_sequence[x.num_transitions()] for x in batch]
                trans_tensor = [transition_tensors[gold_transition] for gold_transition in gold_transitions]
                all_errors.append(outputs)
                all_answers.extend(trans_tensor)

                for pred_transition, gold_transition in zip(pred_transitions, gold_transitions):
                    if pred_transition != gold_transition:
                        transitions_incorrect = transitions_incorrect + 1
                    else:
                        transitions_correct = transitions_correct + 1

                # eliminate finished trees, keeping only the transitions we will use
                zipped_batch = [x for x in zip(batch, gold_transitions) if x[0].num_transitions() + 1 < len(x[0].gold_sequence)]
                batch = [x[0] for x in zipped_batch]
                gold_transitions = [x[1] for x in zipped_batch]

                if len(batch) > 0:
                    # bulk update states
                    batch = parse_transitions.bulk_apply(model, batch, gold_transitions, fail=True, max_transitions=None)

            errors = torch.cat(all_errors)
            answers = torch.cat(all_answers)
            tree_loss = loss_function(errors, answers)
            tree_loss.backward()
            epoch_loss += tree_loss.item()

            optimizer.step()
            optimizer.zero_grad()

        # print statistics
        f1 = run_dev_set(model, dev_trees, args)
        if f1 > best_f1:
            logger.info("New best dev score: %.5f > %.5f", f1, best_f1)
            best_f1 = f1
            best_epoch = epoch
            trainer.save(model_filename, save_optimizer=True)
        if model_latest_filename:
            trainer.save(model_latest_filename, save_optimizer=True)
        logger.info("Epoch {} finished\nTransitions correct: {}  Transitions incorrect: {}\n  Total loss for epoch: {}\n  Dev score      ({:5}): {}\n  Best dev score ({:5}): {}".format(epoch, transitions_correct, transitions_incorrect, epoch_loss, epoch, f1, best_epoch, best_f1))

def build_batch_from_trees(batch_size, data_iterator, model):
    """
    Read from the data_iterator batch_size trees and turn them into new parsing states
    """
    tree_batch = []
    for _ in range(batch_size):
        gold_tree = next(data_iterator, None)
        if gold_tree is None:
            break
        tree_batch.append(gold_tree)

    if len(tree_batch) > 0:
        tree_batch = parse_transitions.initial_state_from_gold_trees(tree_batch, model)
    return tree_batch

def build_batch_from_tagged_words(batch_size, data_iterator, model):
    """
    Read from the data_iterator batch_size tagged sentences and turn them into new parsing states
    """
    tree_batch = []
    for _ in range(batch_size):
        sentence = next(data_iterator, None)
        if sentence is None:
            break
        tree_batch.append(sentence)

    if len(tree_batch) > 0:
        tree_batch = parse_transitions.initial_state_from_words(tree_batch, model)
    return tree_batch

def parse_sentences(data_iterator, build_batch_fn, batch_size, model):
    """
    Given an iterator over the data and a method for building batches, returns a bunch of parse trees.

    The data_iterator should be anything which returns the data for a parse task via next()
    build_batch_fn is a function that turns that data into State objects
    This will be called to generate batches of size batch_size until the data is exhausted

    The return is a list of tuples: (gold_tree, [(predicted, score) ...])
    gold_tree will be left blank if the data did not include gold trees
    currently score is always 1.0, but the interface may be expanded to get a score from the result of the parsing
    """
    treebank = []
    tree_batch = build_batch_fn(batch_size, data_iterator, model)
    horizon_iterator = iter([])

    while len(tree_batch) > 0:
        _, transitions = model.predict(tree_batch, is_legal=True)
        tree_batch = parse_transitions.bulk_apply(model, tree_batch, transitions)

        remove = set()
        for idx, tree in enumerate(tree_batch):
            if tree.finished(model):
                predicted_tree = tree.get_tree(model)
                gold_tree = tree.gold_tree
                # TODO: put an actual score here?
                treebank.append((gold_tree, [(predicted_tree, 1.0)]))
                remove.add(idx)

        tree_batch = [tree for idx, tree in enumerate(tree_batch) if idx not in remove]

        for _ in range(batch_size - len(tree_batch)):
            horizon_tree = next(horizon_iterator, None)
            if not horizon_tree:
                horizon_batch = build_batch_fn(batch_size, data_iterator, model)
                if len(horizon_batch) == 0:
                    break
                horizon_iterator = iter(horizon_batch)
                horizon_tree = next(horizon_iterator, None)

            tree_batch.append(horizon_tree)

    return treebank

def parse_tagged_words(model, words, batch_size):
    """
    This parses tagged words and returns a list of trees.

    The tagged words should be represented:
      one list per sentence
        each sentence is a list of (word, tag)
    The return value is a list of ParseTree objects
    """
    logger.debug("Processing %d sentences", len(words))
    model.eval()

    sentence_iterator = iter(words)
    treebank = parse_sentences(sentence_iterator, build_batch_from_tagged_words, batch_size, model)

    results = [t[1][0][0] for t in treebank]
    return results

def run_dev_set(model, dev_trees, args):
    """
    This reparses a treebank and executes the CoreNLP Java EvalB code.

    It only works if CoreNLP 4.3.0 or higher is in the classpath.
    """
    logger.info("Processing %d trees from %s", len(dev_trees), args['eval_file'])
    model.eval()

    tree_iterator = iter(tqdm(dev_trees))
    treebank = parse_sentences(tree_iterator, build_batch_from_trees, args['eval_batch_size'], model)

    if len(treebank) < len(dev_trees):
        logger.warning("Only evaluating %d trees instead of %d", len(treebank), len(dev_trees))

    if args['mode'] == 'predict' and args['predict_file']:
        utils.ensure_dir(args['predict_dir'], verbose=False)
        pred_file = os.path.join(args['predict_dir'], args['predict_file'] + ".pred.mrg")
        orig_file = os.path.join(args['predict_dir'], args['predict_file'] + ".orig.mrg")
        if os.path.exists(pred_file):
            logger.warning("Cowardly refusing to overwrite {}".format(pred_file))
        elif os.path.exists(orig_file):
            logger.warning("Cowardly refusing to overwrite {}".format(orig_file))
        else:
            with open(pred_file, 'w') as fout:
                for tree in treebank:
                    fout.write(str(tree[1][0][0]))
                    fout.write("\n")

            with open(orig_file, 'w') as fout:
                for tree in treebank:
                    fout.write(str(tree[0]))
                    fout.write("\n")

    with EvaluateParser() as evaluator:
        response = evaluator.process(treebank)
        return response.f1
