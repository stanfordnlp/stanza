

from stanza.models.constituency.parse_transitions import *
from stanza.models.constituency.tree_reader import read_trees

def yield_top_down_sequence(tree):
    if tree.is_preterminal():
        yield Shift()
        return

    if tree.is_leaf():
        return

    if len(tree.children) == 1:
        labels = []
        while not tree.is_preterminal() and len(tree.children) == 1:
            labels.append(tree.label)
            tree = tree.children[0]
        for transition in yield_top_down_sequence(tree):
            yield transition
        yield CompoundUnary(labels)
        return

    yield OpenConstituent(tree.label)
    for child in tree.children:
        for transition in yield_top_down_sequence(child):
            yield transition
    yield CloseConstituent()

def build_top_down_sequence(tree):
    return [t for t in yield_top_down_sequence(tree)]

def build_top_down_treebank(trees):
    return [build_top_down_sequence(tree) for tree in trees]

def all_transitions(transition_lists):
    """
    Given a list of transition lists, combine them all into a set of unique transitions.
    """
    transitions = set()
    for tl in transition_lists:
        for t in tl:
            transitions.add(t)
    return sorted(transitions)

if __name__ == '__main__':
    text="( (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"
    #text = "(WP Who)"

    tree = read_trees(text)[0]

    print(tree)
    transitions = build_top_down_sequence(tree)
    print(transitions)
