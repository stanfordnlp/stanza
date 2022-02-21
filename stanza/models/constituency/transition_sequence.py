"""
Build a transition sequence from parse trees.

Supports multiple transition schemes - TOP_DOWN and variants, IN_ORDER
"""

from stanza.models.constituency.parse_transitions import Shift, CompoundUnary, OpenConstituent, CloseConstituent, TransitionScheme
from stanza.models.constituency.tree_reader import read_trees

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
            yield CompoundUnary(labels)
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

def build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN_UNARY):
    """
    Turn a single tree into a list of transitions based on the TransitionScheme
    """
    if transition_scheme is TransitionScheme.IN_ORDER:
        return list(yield_in_order_sequence(tree))
    else:
        return list(yield_top_down_sequence(tree, transition_scheme))

def build_treebank(trees, transition_scheme=TransitionScheme.TOP_DOWN_UNARY):
    """
    Turn each of the trees in the treebank into a list of transitions based on the TransitionScheme
    """
    return [build_sequence(tree, transition_scheme) for tree in trees]

def all_transitions(transition_lists):
    """
    Given a list of transition lists, combine them all into a list of unique transitions.
    """
    transitions = set()
    for trans_list in transition_lists:
        transitions.update(trans_list)
    return sorted(transitions)

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
