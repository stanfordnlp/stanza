"""
Collects a few of the conparser utility methods which don't belong elsewhere
"""

from collections import deque
import copy

import torch.nn as nn
from torch import optim

from stanza.models.common.doc import TEXT, Document

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

def build_nonlinearity(nonlinearity):
    if nonlinearity == 'tanh':
        return nn.Tanh()
    elif nonlinearity == 'relu':
        return nn.ReLU()
    elif nonlinearity == 'gelu':
        return nn.GELU()
    elif nonlinearity == 'leaky_relu':
        return nn.LeakyReLU()
    else:
        raise ValueError('Chosen value of nonlinearity, "%s", not handled' % nonlinearity)

def build_optimizer(args, model):
    """
    Build an optimizer based on the arguments given
    """
    if args['optim'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args['learning_rate'], momentum=0.9, weight_decay=args['weight_decay'])
    elif args['optim'].lower() == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args['learning_rate'], eps=args['learning_eps'], weight_decay=args['weight_decay'], rho=args['learning_rho'])
    elif args['optim'].lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    elif args['optim'].lower() == 'adabelief':
        try:
            from adabelief_pytorch import AdaBelief
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Could not create adabelief optimizer.  Perhaps the adabelief-pytorch package is not installed") from e
        # TODO: make these args
        optimizer = AdaBelief(model.parameters(), lr=args['learning_rate'], eps=args['learning_eps'], weight_decay=args['weight_decay'], weight_decouple=False, rectify=False)
    elif args['optim'].lower() == 'madgrad':
        try:
            import madgrad
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Could not create madgrad optimizer.  Perhaps the madgrad package is not installed") from e
        optimizer = madgrad.MADGRAD(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    else:
        raise ValueError("Unknown optimizer: %s" % args.optim)
    return optimizer

def initialize_linear(linear, nonlinearity, bias):
    """
    Initializes the bias to a positive value, hopefully preventing dead neurons
    """
    if nonlinearity in ('relu', 'leaky_relu'):
        nn.init.kaiming_normal_(linear.weight, nonlinearity=nonlinearity)
        nn.init.uniform_(linear.bias, 0, 1 / (bias * 2) ** 0.5)
