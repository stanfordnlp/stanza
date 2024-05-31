"""
Collects a few of the conparser utility methods which don't belong elsewhere
"""

from collections import Counter
import logging
import warnings

import torch.nn as nn
from torch import optim

from stanza.models.common.doc import TEXT, Document
from stanza.models.common.utils import get_optimizer
from stanza.models.constituency.base_model import SimpleModel
from stanza.models.constituency.parse_transitions import TransitionScheme
from stanza.models.constituency.parse_tree import Tree
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

DEFAULT_LEARNING_RATES = { "adamw": 0.0002, "adadelta": 1.0, "sgd": 0.001, "adabelief": 0.00005, "madgrad": 0.0000007 , "mirror_madgrad": 0.00005 }
DEFAULT_LEARNING_EPS = { "adabelief": 1e-12, "adadelta": 1e-6, "adamw": 1e-8 }
DEFAULT_LEARNING_RHO = 0.9
DEFAULT_MOMENTUM = { "madgrad": 0.9, "mirror_madgrad": 0.9, "sgd": 0.9 }

tlogger = logging.getLogger('stanza.constituency.trainer')

# madgrad experiment for weight decay
# with learning_rate set to 0.0000007 and momentum 0.9
# on en_wsj, with a baseline model trained on adadela for 200,
# then madgrad used to further improve that model
#  0.00000002.out: 0.9590347746438835
#  0.00000005.out: 0.9591378819960182
#  0.0000001.out: 0.9595450596319405
#  0.0000002.out: 0.9594603134479271
#  0.0000005.out: 0.9591317672706594
#  0.000001.out: 0.9592548741021389
#  0.000002.out: 0.9598395477013945
#  0.000003.out: 0.9594974271553495
#  0.000004.out: 0.9596665982603754
#  0.000005.out: 0.9591620720706487
DEFAULT_WEIGHT_DECAY = { "adamw": 0.05, "adadelta": 0.02, "sgd": 0.01, "adabelief": 1.2e-6, "madgrad": 2e-6, "mirror_madgrad": 2e-6 }

def retag_tags(doc, pipelines, xpos):
    """
    Returns a list of list of tags for the items in doc

    doc can be anything which feeds into the pipeline(s)
    pipelines are a list of 1 or more retag pipelines
    if multiple pipelines are given, majority vote wins
    """
    tag_lists = []
    for pipeline in pipelines:
        doc = pipeline(doc)
        tag_lists.append([[x.xpos if xpos else x.upos for x in sentence.words] for sentence in doc.sentences])
    # tag_lists: for N pipeline, S sentences
    # we now have N lists of S sentences each
    # for sentence in zip(*tag_lists): N lists of |s| tags for this given sentence s
    # for tag in zip(*sentence): N predicted tags.
    # most common one in the Counter will be chosen
    tag_lists = [[Counter(tag).most_common(1)[0][0] for tag in zip(*sentence)]
                 for sentence in zip(*tag_lists)]
    return tag_lists

def retag_trees(trees, pipelines, xpos=True):
    """
    Retag all of the trees using the given processor

    Returns a list of new trees
    """
    if len(trees) == 0:
        return trees

    new_trees = []
    chunk_size = 1000
    with tqdm(total=len(trees)) as pbar:
        for chunk_start in range(0, len(trees), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(trees))
            chunk = trees[chunk_start:chunk_end]
            sentences = []
            try:
                for idx, tree in enumerate(chunk):
                    tokens = [{TEXT: pt.children[0].label} for pt in tree.yield_preterminals()]
                    sentences.append(tokens)
            except ValueError as e:
                raise ValueError("Unable to process tree %d" % (idx + chunk_start)) from e

            doc = Document(sentences)
            tag_lists = retag_tags(doc, pipelines, xpos)

            for tree_idx, (tree, tags) in enumerate(zip(chunk, tag_lists)):
                try:
                    if any(tag is None for tag in tags):
                        raise RuntimeError("Tagged tree #{} with a None tag!\n{}\n{}".format(tree_idx, tree, tags))
                    new_tree = tree.replace_tags(tags)
                    new_trees.append(new_tree)
                    pbar.update(1)
                except ValueError as e:
                    raise ValueError("Failed to properly retag tree #{}: {}".format(tree_idx, tree)) from e
    if len(new_trees) != len(trees):
        raise AssertionError("Retagged tree counts did not match: {} vs {}".format(len(new_trees), len(trees)))
    return new_trees


# experimental results on nonlinearities
# this is on a VI dataset, VLSP_22, using 1/10th of the data as a dev set
# (no released test set at the time of the experiment)
# original non-Bert tagger, with 1 iteration each instead of averaged over 5
# considering the number of experiments and the length of time they would take
#
# Gelu had the highest score, which tracks with other experiments run.
# Note that publicly released models have typically used Relu
# on account of the runtime speed improvement
#
# Anyway, a larger experiment of 5x models on gelu or relu, using the
# Roberta POS tagger and a corpus of silver trees, resulted in 0.8270
# for relu and 0.8248 for gelu.  So it is not even clear that
# switching to gelu would be an accuracy improvement.
#
# Gelu: 82.32
# Relu: 82.14
# Mish: 81.95
# Relu6: 81.91
# Silu: 81.90
# ELU: 81.73
# Hardswish: 81.67
# Softsign: 81.63
# Hardtanh: 81.44
# Celu: 81.43
# Selu: 81.17
#   TODO: need to redo the prelu experiment with
#         possibly different numbers of parameters
#         and proper weight decay
# Prelu: 80.95 (terminated early)
# Softplus: 80.94
# Logsigmoid: 80.91
# Hardsigmoid: 79.03
# RReLU: 77.00
# Hardshrink: failed
# Softshrink: failed
NONLINEARITY = {
    'celu':       nn.CELU,
    'elu':        nn.ELU,
    'gelu':       nn.GELU,
    'hardshrink': nn.Hardshrink,
    'hardtanh':   nn.Hardtanh,
    'leaky_relu': nn.LeakyReLU,
    'logsigmoid': nn.LogSigmoid,
    'prelu':      nn.PReLU,
    'relu':       nn.ReLU,
    'relu6':      nn.ReLU6,
    'rrelu':      nn.RReLU,
    'selu':       nn.SELU,
    'softplus':   nn.Softplus,
    'softshrink': nn.Softshrink,
    'softsign':   nn.Softsign,
    'tanhshrink': nn.Tanhshrink,
    'tanh':       nn.Tanh,
}

# separating these out allows for backwards compatibility with earlier versions of pytorch
# NOTE torch compatibility: if we ever *release* models with these
# activation functions, we will need to break that compatibility

nonlinearity_list = [
    'GLU',
    'Hardsigmoid',
    'Hardswish',
    'Mish',
    'SiLU',
]

for nonlinearity in nonlinearity_list:
    if hasattr(nn, nonlinearity):
        NONLINEARITY[nonlinearity.lower()] = getattr(nn, nonlinearity)

def build_nonlinearity(nonlinearity):
    """
    Look up "nonlinearity" in a map from function name to function, build the appropriate layer.
    """
    if nonlinearity in NONLINEARITY:
        return NONLINEARITY[nonlinearity]()
    raise ValueError('Chosen value of nonlinearity, "%s", not handled' % nonlinearity)

def build_optimizer(args, model, build_simple_adadelta=False):
    """
    Build an optimizer based on the arguments given

    If we are "multistage" training and epochs_trained < epochs // 2,
    we build an AdaDelta optimizer instead of whatever was requested
    The build_simple_adadelta parameter controls this
    """
    bert_learning_rate = 0.0
    bert_weight_decay = args['bert_weight_decay']
    if build_simple_adadelta:
        optim_type = 'adadelta'
        bert_finetune = args.get('stage1_bert_finetune', False)
        if bert_finetune:
            bert_learning_rate = args['stage1_bert_learning_rate']
        learning_beta2 = 0.999   # doesn't matter for AdaDelta
        learning_eps = DEFAULT_LEARNING_EPS['adadelta']
        learning_rate = args['stage1_learning_rate']
        learning_rho = DEFAULT_LEARNING_RHO
        momentum = None    # also doesn't matter for AdaDelta
        weight_decay = DEFAULT_WEIGHT_DECAY['adadelta']
    else:
        optim_type = args['optim'].lower()
        bert_finetune = args.get('bert_finetune', False)
        if bert_finetune:
            bert_learning_rate = args['bert_learning_rate']
        learning_beta2 = args['learning_beta2']
        learning_eps = args['learning_eps']
        learning_rate = args['learning_rate']
        learning_rho = args['learning_rho']
        momentum = args['learning_momentum']
        weight_decay = args['learning_weight_decay']

    # TODO: allow rho as an arg for AdaDelta
    return get_optimizer(name=optim_type,
                         model=model,
                         lr=learning_rate,
                         betas=(0.9, learning_beta2),
                         eps=learning_eps,
                         momentum=momentum,
                         weight_decay=weight_decay,
                         bert_learning_rate=bert_learning_rate,
                         bert_weight_decay=weight_decay*bert_weight_decay,
                         is_peft=args.get('use_peft', False),
                         bert_finetune_layers=args['bert_finetune_layers'],
                         opt_logger=tlogger)

def build_scheduler(args, optimizer, first_optimizer=False):
    """
    Build the scheduler for the conparser based on its args

    Used to use a warmup for learning rate, but that wasn't working very well
    Now, we just use a ReduceLROnPlateau, which does quite well
    """
    #if args.get('learning_rate_warmup', 0) <= 0:
    #    # TODO: is there an easier way to make an empty scheduler?
    #    lr_lambda = lambda x: 1.0
    #else:
    #    warmup_end = args['learning_rate_warmup']
    #    def lr_lambda(x):
    #        if x >= warmup_end:
    #            return 1.0
    #        return x / warmup_end

    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if first_optimizer:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args['learning_rate_factor'], patience=args['learning_rate_patience'], cooldown=args['learning_rate_cooldown'], min_lr=args['stage1_learning_rate_min_lr'])
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args['learning_rate_factor'], patience=args['learning_rate_patience'], cooldown=args['learning_rate_cooldown'], min_lr=args['learning_rate_min_lr'])
    return scheduler

def initialize_linear(linear, nonlinearity, bias):
    """
    Initializes the bias to a positive value, hopefully preventing dead neurons
    """
    if nonlinearity in ('relu', 'leaky_relu'):
        nn.init.kaiming_normal_(linear.weight, nonlinearity=nonlinearity)
        nn.init.uniform_(linear.bias, 0, 1 / (bias * 2) ** 0.5)

def add_predict_output_args(parser):
    """
    Args specifically for the output location of data
    """
    parser.add_argument('--predict_dir', type=str, default=".", help='Where to write the predictions during --mode predict.  Pred and orig files will be written - the orig file will be retagged if that is requested.  Writing the orig file is useful for removing None and retagging')
    parser.add_argument('--predict_file', type=str, default=None, help='Base name for writing predictions')
    parser.add_argument('--predict_format', type=str, default="{:_O}", help='Format to use when writing predictions')

    parser.add_argument('--predict_output_gold_tags', default=False, action='store_true', help='Output gold tags as part of the evaluation - useful for putting the trees through EvalB')

def postprocess_predict_output_args(args):
    if len(args['predict_format']) <= 2 or (len(args['predict_format']) <= 3 and args['predict_format'].endswith("Vi")):
        args['predict_format'] = "{:" + args['predict_format'] + "}"


def get_open_nodes(trees, transition_scheme):
    """
    Return a list of all open nodes in the given dataset.
    Depending on the parameters, may be single or compound open transitions.
    """
    if transition_scheme is TransitionScheme.TOP_DOWN_COMPOUND:
        return Tree.get_compound_constituents(trees)
    elif transition_scheme is TransitionScheme.IN_ORDER_COMPOUND:
        return Tree.get_compound_constituents(trees, separate_root=True)
    else:
        return [(x,) for x in Tree.get_unique_constituent_labels(trees)]


def verify_transitions(trees, sequences, transition_scheme, unary_limit, reverse, name, root_labels):
    """
    Given a list of trees and their transition sequences, verify that the sequences rebuild the trees
    """
    model = SimpleModel(transition_scheme, unary_limit, reverse, root_labels)
    tlogger.info("Verifying the transition sequences for %d trees", len(trees))

    data = zip(trees, sequences)
    if tlogger.getEffectiveLevel() <= logging.INFO:
        data = tqdm(zip(trees, sequences), total=len(trees))

    for tree_idx, (tree, sequence) in enumerate(data):
        # TODO: make the SimpleModel have a parse operation?
        state = model.initial_state_from_gold_trees([tree])[0]
        for idx, trans in enumerate(sequence):
            if not trans.is_legal(state, model):
                raise RuntimeError("Tree {} of {} failed: transition {}:{} was not legal in a transition sequence:\nOriginal tree: {}\nTransitions: {}".format(tree_idx, name, idx, trans, tree, sequence))
            state = trans.apply(state, model)
        result = model.get_top_constituent(state.constituents)
        if reverse:
            result = result.reverse()
        if tree != result:
            raise RuntimeError("Tree {} of {} failed: transition sequence did not match for a tree!\nOriginal tree:{}\nTransitions: {}\nResult tree:{}".format(tree_idx, name, tree, sequence, result))

def check_constituents(train_constituents, trees, treebank_name, fail=True):
    """
    Check that all the constituents in the other dataset are known in the train set
    """
    constituents = Tree.get_unique_constituent_labels(trees)
    for con in constituents:
        if con not in train_constituents:
            first_error = None
            num_errors = 0
            for tree_idx, tree in enumerate(trees):
                constituents = Tree.get_unique_constituent_labels(tree)
                if con in constituents:
                    num_errors += 1
                    if first_error is None:
                        first_error = tree_idx
            error = "Found constituent label {} in the {} set which don't exist in the train set.  This constituent label occured in {} trees, with the first tree index at {} counting from 1\nThe error tree (which may have POS tags changed from the retagger and may be missing functional tags or empty nodes) is:\n{:P}".format(con, treebank_name, num_errors, (first_error+1), trees[first_error])
            if fail:
                raise RuntimeError(error)
            else:
                warnings.warn(error)

def check_root_labels(root_labels, other_trees, treebank_name):
    """
    Check that all the root states in the other dataset are known in the train set
    """
    for root_state in Tree.get_root_labels(other_trees):
        if root_state not in root_labels:
            raise RuntimeError("Found root state {} in the {} set which is not a ROOT state in the train set".format(root_state, treebank_name))

def remove_duplicate_trees(trees, treebank_name):
    """
    Filter duplicates from the given dataset
    """
    new_trees = []
    known_trees = set()
    for tree in trees:
        tree_str = "{}".format(tree)
        if tree_str in known_trees:
            continue
        known_trees.add(tree_str)
        new_trees.append(tree)
    if len(new_trees) < len(trees):
        tlogger.info("Filtered %d duplicates from %s dataset", (len(trees) - len(new_trees)), treebank_name)
    return new_trees

def remove_singleton_trees(trees):
    """
    remove trees which are just a root and a single word

    TODO: remove these trees in the conversion instead of here
    """
    new_trees = [x for x in trees if
                 len(x.children) > 1 or
                 (len(x.children) == 1 and len(x.children[0].children) > 1) or
                 (len(x.children) == 1 and len(x.children[0].children) == 1 and len(x.children[0].children[0].children) >= 1)]
    if len(trees) - len(new_trees) > 0:
        tlogger.info("Eliminated %d trees with missing structure", (len(trees) - len(new_trees)))
    return new_trees

