

from stanza.models.constituency.parse_transitions import *
from stanza.models.constituency.tree_reader import read_trees

def yield_top_down_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN_UNARY):
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

def build_top_down_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN_UNARY):
    return list(yield_top_down_sequence(tree, transition_scheme))

def build_top_down_treebank(trees, transition_scheme=TransitionScheme.TOP_DOWN_UNARY):
    return [build_top_down_sequence(tree, transition_scheme) for tree in trees]

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
