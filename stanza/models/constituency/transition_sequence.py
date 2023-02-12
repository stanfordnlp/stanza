"""
Build a transition sequence from parse trees.

Supports multiple transition schemes - TOP_DOWN and variants, IN_ORDER
"""

import logging

from stanza.models.common import utils
from stanza.models.constituency.parse_transitions import Shift, CompoundUnary, OpenConstituent, CloseConstituent, TransitionScheme, Finalize
from stanza.models.constituency.tree_reader import read_trees
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

logger = logging.getLogger('stanza.constituency.trainer')

def yield_top_down_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN_UNARY):
    """
    For tree (X A B C D), yield Open(X) A B C D Close

    The details are in how to treat unary transitions
    Three possibilities handled by this method:
      TOP_DOWN_UNARY:    (Y (X ...)) -> Open(X) ... Close Unary(Y)
      TOP_DOWN_COMPOUND: (Y (X ...)) -> Open(Y, X) ... Close
      TOP_DOWN:          (Y (X ...)) -> Open(Y) Open(X) ... Close Close
    """
    if tree.is_preterminal():
        yield Shift()
        return

    if tree.is_leaf():
        return

    if transition_scheme is TransitionScheme.TOP_DOWN_UNARY:
        if len(tree.children) == 1:
            labels = []
            while not tree.is_preterminal() and len(tree.children) == 1:
                labels.append(tree.label)
                tree = tree.children[0]
            for transition in yield_top_down_sequence(tree, transition_scheme):
                yield transition
            yield CompoundUnary(*labels)
            return

    if transition_scheme is TransitionScheme.TOP_DOWN_COMPOUND:
        labels = [tree.label]
        while len(tree.children) == 1 and not tree.children[0].is_preterminal():
            tree = tree.children[0]
            labels.append(tree.label)
        yield OpenConstituent(*labels)
    else:
        yield OpenConstituent(tree.label)
    for child in tree.children:
        for transition in yield_top_down_sequence(child, transition_scheme):
            yield transition
    yield CloseConstituent()

def yield_in_order_sequence(tree):
    """
    For tree (X A B C D), yield A Open(X) B C D Close
    """
    if tree.is_preterminal():
        yield Shift()
        return

    if tree.is_leaf():
        return

    for transition in yield_in_order_sequence(tree.children[0]):
        yield transition

    yield OpenConstituent(tree.label)

    for child in tree.children[1:]:
        for transition in yield_in_order_sequence(child):
            yield transition

    yield CloseConstituent()



def yield_in_order_compound_sequence(tree, transition_scheme):
    def helper(tree):
        if tree.is_leaf():
            return

        labels = []
        while len(tree.children) == 1 and not tree.is_preterminal():
            labels.append(tree.label)
            tree = tree.children[0]

        if tree.is_preterminal():
            yield Shift()
            if len(labels) > 0:
                yield CompoundUnary(*labels)
            return

        for transition in helper(tree.children[0]):
            yield transition

        if transition_scheme is TransitionScheme.IN_ORDER_UNARY:
            yield OpenConstituent(tree.label)
        else:
            labels.append(tree.label)
            yield OpenConstituent(*labels)

        for child in tree.children[1:]:
            for transition in helper(child):
                yield transition

        yield CloseConstituent()

        if transition_scheme is TransitionScheme.IN_ORDER_UNARY and len(labels) > 0:
            yield CompoundUnary(*labels)

    if len(tree.children) == 0:
        raise ValueError("Cannot build {} on an empty tree".format(transition_scheme))
    if len(tree.children) != 1:
        raise ValueError("Cannot build {} with a tree that has two top level nodes: {}".format(transition_scheme, tree))

    for t in helper(tree.children[0]):
        yield t

    yield Finalize(tree.label)

def build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN_UNARY):
    """
    Turn a single tree into a list of transitions based on the TransitionScheme
    """
    if transition_scheme is TransitionScheme.IN_ORDER:
        return list(yield_in_order_sequence(tree))
    elif (transition_scheme is TransitionScheme.IN_ORDER_COMPOUND or
          transition_scheme is TransitionScheme.IN_ORDER_UNARY):
        return list(yield_in_order_compound_sequence(tree, transition_scheme))
    else:
        return list(yield_top_down_sequence(tree, transition_scheme))

def build_treebank(trees, transition_scheme=TransitionScheme.TOP_DOWN_UNARY, reverse=False):
    """
    Turn each of the trees in the treebank into a list of transitions based on the TransitionScheme
    """
    if reverse:
        return [build_sequence(tree.reverse(), transition_scheme) for tree in trees]
    else:
        return [build_sequence(tree, transition_scheme) for tree in trees]

def all_transitions(transition_lists):
    """
    Given a list of transition lists, combine them all into a list of unique transitions.
    """
    transitions = set()
    for trans_list in transition_lists:
        transitions.update(trans_list)
    return sorted(transitions)

def convert_trees_to_sequences(trees, treebank_name, transition_scheme, reverse=False):
    """
    Wrap both build_treebank and all_transitions, possibly with a tqdm

    Converts trees to a list of sequences, then returns the list of known transitions
    """
    logger.info("Building {} transition sequences".format(treebank_name))
    if logger.getEffectiveLevel() <= logging.INFO:
        trees = tqdm(trees)
    sequences = build_treebank(trees, transition_scheme, reverse)
    transitions = all_transitions(sequences)
    return sequences, transitions

def main():
    """
    Convert a sample tree and print its transitions
    """
    text="( (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"
    #text = "(WP Who)"

    tree = read_trees(text)[0]

    print(tree)
    transitions = build_sequence(tree)
    print(transitions)

if __name__ == '__main__':
    main()
