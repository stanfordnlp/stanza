"""
Collects a few of the conparser utility methods which don't belong elsewhere
"""

from collections import deque
import copy

import torch.nn as nn
from torch import optim

from stanza.models.common.doc import TEXT, Document

DEFAULT_LEARNING_RATES = { "adamw": 0.0002, "adadelta": 1.0, "sgd": 0.001, "adabelief": 0.00005, "madgrad": 0.00001 }
DEFAULT_LEARNING_EPS = { "adabelief": 1e-12, "adadelta": 1e-6, "adamw": 1e-8 }
DEFAULT_WEIGHT_DECAY = { "adamw": 0.05, "adadelta": 0.02, "sgd": 0.01, "adabelief": 1.2e-6, "madgrad": 1e-6 }
DEFAULT_LEARNING_RHO = 0.9

class TextTooLongError(ValueError):
    """
    A text was too long for the underlying model (possibly BERT)
    """
    def __init__(self, length, max_len, line_num, text):
        super().__init__("Found a text of length %d (possibly after tokenizing).  Maximum handled length is %d  Error occurred at line %d" % (length, max_len, line_num))
        self.line_num = line_num
        self.text = text


def replace_tags(tree, tags):
    if tree.is_leaf():
        raise ValueError("Must call replace_tags with non-leaf")

    tag_iterator = iter(tags)

    new_tree = copy.deepcopy(tree)
    queue = deque()
    queue.append(new_tree)
    while len(queue) > 0:
        next_node = queue.pop()
        if next_node.is_preterminal():
            try:
                label = next(tag_iterator)
            except StopIteration:
                raise ValueError("Not enough tags in sentence for given tree")
            next_node.label = label
        elif next_node.is_leaf():
            raise ValueError("Got a badly structured tree: {}".format(tree))
        else:
            queue.extend(reversed(next_node.children))

    if any(True for _ in tag_iterator):
        raise ValueError("Too many tags for the given tree")

    return new_tree


def retag_trees(trees, pipeline, xpos=True):
    """
    Retag all of the trees using the given processor

    Returns a list of new trees
    """
    sentences = []
    try:
        for idx, tree in enumerate(trees):
            tokens = [{TEXT: pt.children[0].label} for pt in tree.yield_preterminals()]
            sentences.append(tokens)
    except ValueError as e:
        raise ValueError("Unable to process tree %d" % idx) from e

    doc = Document(sentences)
    doc = pipeline(doc)
    if xpos:
        tag_lists = [[x.xpos for x in sentence.words] for sentence in doc.sentences]
    else:
        tag_lists = [[x.upos for x in sentence.words] for sentence in doc.sentences]

    new_trees = []
    for tree_idx, (tree, tags) in enumerate(zip(trees, tag_lists)):
        try:
            new_tree = replace_tags(tree, tags)
            new_trees.append(new_tree)
        except ValueError as e:
            raise ValueError("Failed to properly retag tree #{}: {}".format(tree_idx, tree)) from e
    return new_trees

NONLINEARITY = {
    'tanh':       nn.Tanh,
    'relu':       nn.ReLU,
    'gelu':       nn.GELU,
    'leaky_relu': nn.LeakyReLU,
    'silu':       nn.SiLU,
    'mish':       nn.Mish,
}

def build_nonlinearity(nonlinearity):
    """
    Look up "nonlinearity" in a map from function name to function, build the appropriate layer.
    """
    if nonlinearity in NONLINEARITY:
        return NONLINEARITY[nonlinearity]()
    raise ValueError('Chosen value of nonlinearity, "%s", not handled' % nonlinearity)

def build_optimizer(args, model):
    """
    Build an optimizer based on the arguments given
    """
    parameters = [param for name, param in model.named_parameters() if not model.is_unsaved_module(name)]
    if args['optim'].lower() == 'sgd':
        optimizer = optim.SGD(parameters, lr=args['learning_rate'], momentum=0.9, weight_decay=args['weight_decay'])
    elif args['optim'].lower() == 'adadelta':
        optimizer = optim.Adadelta(parameters, lr=args['learning_rate'], eps=args['learning_eps'], weight_decay=args['weight_decay'], rho=args['learning_rho'])
    elif args['optim'].lower() == 'adamw':
        optimizer = optim.AdamW(parameters, lr=args['learning_rate'], betas=(0.9, args['learning_beta2']), eps=args['learning_eps'], weight_decay=args['weight_decay'])
    elif args['optim'].lower() == 'adabelief':
        try:
            from adabelief_pytorch import AdaBelief
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Could not create adabelief optimizer.  Perhaps the adabelief-pytorch package is not installed") from e
        # TODO: make these args
        optimizer = AdaBelief(parameters, lr=args['learning_rate'], eps=args['learning_eps'], weight_decay=args['weight_decay'], weight_decouple=False, rectify=False)
    elif args['optim'].lower() == 'madgrad':
        try:
            import madgrad
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Could not create madgrad optimizer.  Perhaps the madgrad package is not installed") from e
        optimizer = madgrad.MADGRAD(parameters, lr=args['learning_rate'], weight_decay=args['weight_decay'])
    else:
        raise ValueError("Unknown optimizer: %s" % args['optim'])
    return optimizer

def build_scheduler(args, optimizer):
    if args.get('learning_rate_warmup', 0) <= 0:
        # TODO: is there an easier way to make an empty scheduler?
        lr_lambda = lambda x: 1.0
    else:
        warmup_end = args['learning_rate_warmup']
        def lr_lambda(x):
            if x >= warmup_end:
                return 1.0
            return x / warmup_end

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

def initialize_linear(linear, nonlinearity, bias):
    """
    Initializes the bias to a positive value, hopefully preventing dead neurons
    """
    if nonlinearity in ('relu', 'leaky_relu'):
        nn.init.kaiming_normal_(linear.weight, nonlinearity=nonlinearity)
        nn.init.uniform_(linear.bias, 0, 1 / (bias * 2) ** 0.5)
