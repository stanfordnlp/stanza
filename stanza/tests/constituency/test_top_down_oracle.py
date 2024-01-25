import pytest

from stanza.models.constituency.parse_transitions import Shift, OpenConstituent, CloseConstituent, TransitionScheme
from stanza.models.constituency.top_down_oracle import *
from stanza.models.constituency.transition_sequence import build_sequence
from stanza.models.constituency.tree_reader import read_trees

from stanza.tests.constituency.test_transition_sequence import reconstruct_tree

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

OPEN_SHIFT_EXAMPLE_TREE = """
( (S
     (NP (NNP Jennifer) (NNP Sh\'reyan))
     (VP (VBZ has)
         (NP (RB nice) (NNS antennae)))))
"""

OPEN_SHIFT_PROBLEM_TREE = """
(ROOT (S (NP (NP (NP (DT The) (`` ``) (JJ Thin) (NNP Man) ('' '') (NN series)) (PP (IN of) (NP (NNS movies)))) (, ,) (CONJP (RB as) (RB well) (IN as)) (NP (JJ many) (NNS others)) (, ,)) (VP (VBD based) (NP (PRP$ their) (JJ entire) (JJ comedic) (NN appeal)) (PP (IN on) (NP (NP (DT the) (NN star) (NNS detectives) (POS ')) (JJ witty) (NNS quips) (CC and) (NNS puns))) (SBAR (IN as) (S (NP (NP (JJ other) (NNS characters)) (PP (IN in) (NP (DT the) (NNS movies)))) (VP (VBD were) (VP (VBN murdered)))))) (. .)))
"""

ROOT_LABELS = ["ROOT"]

def test_fix_open_shift():
    trees = read_trees(OPEN_SHIFT_EXAMPLE_TREE)
    assert len(trees) == 1
    tree = trees[0]

    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)
    EXPECTED_ORIG = [OpenConstituent('ROOT'), OpenConstituent('S'), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), OpenConstituent('VP'), Shift(), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent(), CloseConstituent()]
    EXPECTED_FIX_EARLY = [OpenConstituent('ROOT'), OpenConstituent('S'), Shift(), Shift(), OpenConstituent('VP'), Shift(), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent(), CloseConstituent()]
    EXPECTED_FIX_LATE = [OpenConstituent('ROOT'), OpenConstituent('S'), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), OpenConstituent('VP'), Shift(), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent()]

    assert transitions == EXPECTED_ORIG

    new_transitions = fix_one_open_shift(OpenConstituent('NP'), Shift(), transitions, 2, ROOT_LABELS)
    assert new_transitions == EXPECTED_FIX_EARLY

    new_transitions = fix_one_open_shift(OpenConstituent('NP'), Shift(), transitions, 8, ROOT_LABELS)
    assert new_transitions == EXPECTED_FIX_LATE

def test_fix_open_shift_observed_error():
    """
    Ran into an error on this tree, need to fix it

    The problem is the multiple Open in a row all need to be removed when a Shift happens
    """
    trees = read_trees(OPEN_SHIFT_PROBLEM_TREE)
    assert len(trees) == 1
    tree = trees[0]

    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)
    new_transitions = fix_one_open_shift(OpenConstituent('NP'), Shift(), transitions, 2, ROOT_LABELS)
    assert new_transitions is None

    new_transitions = fix_multiple_open_shift(OpenConstituent('NP'), Shift(), transitions, 2, ROOT_LABELS)

    # Can break the expected transitions down like this:
    # [OpenConstituent(('ROOT',)), OpenConstituent(('S',)),
    # all gone: OpenConstituent(('NP',)), OpenConstituent(('NP',)), OpenConstituent(('NP',)),
    # Shift, Shift, Shift, Shift, Shift, Shift,
    # gone: CloseConstituent,
    # OpenConstituent(('PP',)), Shift, OpenConstituent(('NP',)), Shift, CloseConstituent, CloseConstituent,
    # gone: CloseConstituent,
    # Shift, OpenConstituent(('CONJP',)), Shift, Shift, Shift, CloseConstituent, OpenConstituent(('NP',)), Shift, Shift, CloseConstituent, Shift,
    # gone: CloseConstituent,
    # and then the rest:
    # OpenConstituent(('VP',)), Shift, OpenConstituent(('NP',)),
    # Shift, Shift, Shift, Shift, CloseConstituent,
    # OpenConstituent(('PP',)), Shift, OpenConstituent(('NP',)),
    # OpenConstituent(('NP',)), Shift, Shift, Shift, Shift,
    # CloseConstituent, Shift, Shift, Shift, Shift, CloseConstituent,
    # CloseConstituent, OpenConstituent(('SBAR',)), Shift,
    # OpenConstituent(('S',)), OpenConstituent(('NP',)),
    # OpenConstituent(('NP',)), Shift, Shift, CloseConstituent,
    # OpenConstituent(('PP',)), Shift, OpenConstituent(('NP',)),
    # Shift, Shift, CloseConstituent, CloseConstituent,
    # CloseConstituent, OpenConstituent(('VP',)), Shift,
    # OpenConstituent(('VP',)), Shift, CloseConstituent,
    # CloseConstituent, CloseConstituent, CloseConstituent,
    # CloseConstituent, Shift, CloseConstituent, CloseConstituent]
    expected_transitions = [OpenConstituent('ROOT'), OpenConstituent('S'), Shift(), Shift(), Shift(), Shift(), Shift(), Shift(), OpenConstituent('PP'), Shift(), OpenConstituent('NP'), Shift(), CloseConstituent(), CloseConstituent(), Shift(), OpenConstituent('CONJP'), Shift(), Shift(), Shift(), CloseConstituent(), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), Shift(), OpenConstituent('VP'), Shift(), OpenConstituent('NP'), Shift(), Shift(), Shift(), Shift(), CloseConstituent(), OpenConstituent('PP'), Shift(), OpenConstituent('NP'), OpenConstituent('NP'), Shift(), Shift(), Shift(), Shift(), CloseConstituent(), Shift(), Shift(), Shift(), Shift(), CloseConstituent(), CloseConstituent(), OpenConstituent('SBAR'), Shift(), OpenConstituent('S'), OpenConstituent('NP'), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), OpenConstituent('PP'), Shift(), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent(), OpenConstituent('VP'), Shift(), OpenConstituent('VP'), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent(), CloseConstituent(), CloseConstituent(), Shift(), CloseConstituent(), CloseConstituent()]

    assert new_transitions == expected_transitions

def test_open_open_ambiguous_unary_fix():
    trees = read_trees(OPEN_SHIFT_EXAMPLE_TREE)
    assert len(trees) == 1
    tree = trees[0]

    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)
    EXPECTED_ORIG = [OpenConstituent('ROOT'), OpenConstituent('S'), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), OpenConstituent('VP'), Shift(), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent(), CloseConstituent()]
    EXPECTED_FIX = [OpenConstituent('ROOT'), OpenConstituent('S'), OpenConstituent('VP'), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), CloseConstituent(), OpenConstituent('VP'), Shift(), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent(), CloseConstituent()]
    assert transitions == EXPECTED_ORIG
    new_transitions = fix_open_open_ambiguous_unary(OpenConstituent('NP'), OpenConstituent('VP'), transitions, 2, ROOT_LABELS)
    assert new_transitions == EXPECTED_FIX


def test_open_open_ambiguous_later_fix():
    trees = read_trees(OPEN_SHIFT_EXAMPLE_TREE)
    assert len(trees) == 1
    tree = trees[0]

    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)
    EXPECTED_ORIG = [OpenConstituent('ROOT'), OpenConstituent('S'), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), OpenConstituent('VP'), Shift(), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent(), CloseConstituent()]
    EXPECTED_FIX = [OpenConstituent('ROOT'), OpenConstituent('S'), OpenConstituent('VP'), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), OpenConstituent('VP'), Shift(), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent(), CloseConstituent(), CloseConstituent()]
    assert transitions == EXPECTED_ORIG
    new_transitions = fix_open_open_ambiguous_later(OpenConstituent('NP'), OpenConstituent('VP'), transitions, 2, ROOT_LABELS)
    assert new_transitions == EXPECTED_FIX


CLOSE_SHIFT_EXAMPLE_TREE = """
( (NP (DT a)
   (ADJP (NN stock) (HYPH -) (VBG picking))
   (NN tool)))
"""

# not intended to be a correct tree
CLOSE_SHIFT_DEEP_EXAMPLE_TREE = """
( (NP (DT a)
   (VP (ADJP (NN stock) (HYPH -) (VBG picking)))
   (NN tool)))
"""

# not intended to be a correct tree
CLOSE_SHIFT_OPEN_EXAMPLE_TREE = """
( (NP (DT a)
   (ADJP (NN stock) (HYPH -) (VBG picking))
   (NP (NN tool))))
"""

CLOSE_SHIFT_AMBIGUOUS_TREE = """
( (NP (DT a)
   (ADJP (NN stock) (HYPH -) (VBG picking))
   (NN tool)
   (NN foo)))
"""

def test_fix_close_shift_ambiguous_immediate():
    """
    Test the result when a close/shift error occurs and we want to close the new, incorrect constituent immediately
    """
    trees = read_trees(CLOSE_SHIFT_AMBIGUOUS_TREE)
    assert len(trees) == 1
    tree = trees[0]

    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)
    new_sequence = fix_close_shift_ambiguous_later(gold_transition=transitions[7],
                                                   pred_transition=transitions[8],
                                                   gold_sequence=transitions,
                                                   gold_index=7,
                                                   root_labels=ROOT_LABELS)
    expected_original = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), CloseConstituent(), Shift(), Shift(), CloseConstituent(), CloseConstituent()]
    expected_update = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent()]
    assert transitions == expected_original
    assert new_sequence == expected_update

def test_fix_close_shift_ambiguous_later():
    # test that the one with two shifts, which is ambiguous, gets rejected
    trees = read_trees(CLOSE_SHIFT_AMBIGUOUS_TREE)
    assert len(trees) == 1
    tree = trees[0]

    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)
    new_sequence = fix_close_shift_ambiguous_immediate(gold_transition=transitions[7],
                                                       pred_transition=transitions[8],
                                                       gold_sequence=transitions,
                                                       gold_index=7,
                                                       root_labels=ROOT_LABELS)
    expected_original = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), CloseConstituent(), Shift(), Shift(), CloseConstituent(), CloseConstituent()]
    expected_update = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), Shift(), CloseConstituent(), Shift(), CloseConstituent(), CloseConstituent()]
    assert transitions == expected_original
    assert new_sequence == expected_update

def test_oracle_with_optional_level():
    gold_sequence = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), CloseConstituent(), Shift(), Shift(), CloseConstituent(), CloseConstituent()]
    expected_update = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), Shift(), CloseConstituent(), Shift(), CloseConstituent(), CloseConstituent()]

    oracle = TopDownOracle(ROOT_LABELS, 1, "")
    fix, new_sequence = oracle.fix_error(gold_transition=gold_sequence[7],
                                         pred_transition=gold_sequence[8],
                                         gold_sequence=gold_sequence,
                                         gold_index=7)
    assert fix is RepairType.UNKNOWN
    assert new_sequence is None

    oracle = TopDownOracle(ROOT_LABELS, 1, "CLOSE_SHIFT_AMBIGUOUS_IMMEDIATE_ERROR")
    fix, new_sequence = oracle.fix_error(gold_transition=gold_sequence[7],
                                         pred_transition=gold_sequence[8],
                                         gold_sequence=gold_sequence,
                                         gold_index=7)
    assert fix is RepairType.CLOSE_SHIFT_AMBIGUOUS_IMMEDIATE_ERROR
    assert new_sequence == expected_update


def test_fix_close_shift():
    """
    Test a tree of the kind we expect the close/shift to be able to get right
    """
    trees = read_trees(CLOSE_SHIFT_EXAMPLE_TREE)
    assert len(trees) == 1
    tree = trees[0]

    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)

    new_sequence = fix_close_shift(gold_transition=transitions[7],
                                   pred_transition=transitions[8],
                                   gold_sequence=transitions,
                                   gold_index=7,
                                   root_labels=ROOT_LABELS)

    expected_original = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), CloseConstituent(), Shift(), CloseConstituent(), CloseConstituent()]
    expected_update   = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent()]
    assert transitions == expected_original
    assert new_sequence == expected_update

    # test that the one with two shifts, which is ambiguous, gets rejected
    trees = read_trees(CLOSE_SHIFT_AMBIGUOUS_TREE)
    assert len(trees) == 1
    tree = trees[0]

    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)
    new_sequence = fix_close_shift(gold_transition=transitions[7],
                                   pred_transition=transitions[8],
                                   gold_sequence=transitions,
                                   gold_index=7,
                                   root_labels=ROOT_LABELS)
    assert new_sequence is None

def test_fix_close_shift_deeper_tree():
    """
    Test a tree of the kind we expect the close/shift to be able to get right
    """
    trees = read_trees(CLOSE_SHIFT_DEEP_EXAMPLE_TREE)
    assert len(trees) == 1
    tree = trees[0]

    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)

    for count_opens in [True, False]:
        new_sequence = fix_close_shift(gold_transition=transitions[8],
                                       pred_transition=transitions[10],
                                       gold_sequence=transitions,
                                       gold_index=8,
                                       root_labels=ROOT_LABELS,
                                       count_opens=count_opens)

        expected_original = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('VP'), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), CloseConstituent(), CloseConstituent(), Shift(), CloseConstituent(), CloseConstituent()]
        expected_update   = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('VP'), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent(), CloseConstituent()]
        assert transitions == expected_original
        assert new_sequence == expected_update

def test_fix_close_shift_open_tree():
    """
    We would like the close/shift to get this case right as well
    """
    trees = read_trees(CLOSE_SHIFT_OPEN_EXAMPLE_TREE)
    assert len(trees) == 1
    tree = trees[0]

    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)

    assert fix_close_shift(gold_transition=transitions[7],
                           pred_transition=transitions[9],
                           gold_sequence=transitions,
                           gold_index=7,
                           root_labels=ROOT_LABELS,
                           count_opens=False) == None

    new_sequence = fix_close_shift_with_opens(gold_transition=transitions[7],
                                              pred_transition=transitions[9],
                                              gold_sequence=transitions,
                                              gold_index=7,
                                              root_labels=ROOT_LABELS)

    expected_original = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), CloseConstituent(), OpenConstituent('NP'), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent()]
    expected_update   = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent()]
    assert transitions == expected_original
    assert new_sequence == expected_update

    assert fix_close_shift_with_opens(transitions[7], transitions[9], transitions, 7, ROOT_LABELS) == expected_update

CLOSE_OPEN_EXAMPLE_TREE = """
( (VP (VBZ eat)
   (NP (NN spaghetti))
   (PP (IN with) (DT a) (NN fork))))
"""

CLOSE_OPEN_DIFFERENT_LABEL_TREE = """
( (VP (VBZ eat)
   (NP (NN spaghetti))
   (NP (DT a) (NN fork))))
"""

CLOSE_OPEN_TWO_LABELS_TREE = """
( (VP (VBZ eat)
   (NP (NN spaghetti))
   (PP (IN with) (DT a) (NN fork))
   (PP (IN in) (DT a) (NN restaurant))))
"""

def test_fix_close_open():
    trees = read_trees(CLOSE_OPEN_EXAMPLE_TREE)
    assert len(trees) == 1
    tree = trees[0]

    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)

    assert isinstance(transitions[5], CloseConstituent)
    assert transitions[6] == OpenConstituent("PP")

    new_transitions = fix_close_open_correct_open(transitions[5], transitions[6], transitions, 5, ROOT_LABELS)

    expected_original = [OpenConstituent('ROOT'), OpenConstituent('VP'), Shift(), OpenConstituent('NP'), Shift(), CloseConstituent(), OpenConstituent('PP'), Shift(), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent()]
    expected_update   = [OpenConstituent('ROOT'), OpenConstituent('VP'), Shift(), OpenConstituent('NP'), Shift(), OpenConstituent('PP'), Shift(), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent(), CloseConstituent()]

    assert transitions == expected_original
    assert new_transitions == expected_update

def test_fix_close_open_invalid():
    for TREE in (CLOSE_OPEN_DIFFERENT_LABEL_TREE, CLOSE_OPEN_TWO_LABELS_TREE):
        trees = read_trees(TREE)
        assert len(trees) == 1
        tree = trees[0]

        transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)

        assert isinstance(transitions[5], CloseConstituent)
        assert isinstance(transitions[6], OpenConstituent)

        new_transitions = fix_close_open_correct_open(transitions[5], OpenConstituent("PP"), transitions, 5, ROOT_LABELS)
        assert new_transitions is None

def test_fix_close_open_ambiguous_immediate():
    """
    Test that a fix for an ambiguous close/open works as expected
    """
    trees = read_trees(CLOSE_OPEN_TWO_LABELS_TREE)
    assert len(trees) == 1
    tree = trees[0]

    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)
    assert isinstance(transitions[5], CloseConstituent)
    assert isinstance(transitions[6], OpenConstituent)

    reconstructed = reconstruct_tree(tree, transitions, transition_scheme=TransitionScheme.TOP_DOWN)
    assert tree == reconstructed

    new_transitions = fix_close_open_correct_open(transitions[5], OpenConstituent("PP"), transitions, 5, ROOT_LABELS, check_close=False)
    reconstructed = reconstruct_tree(tree, new_transitions, transition_scheme=TransitionScheme.TOP_DOWN)

    expected = """
    ( (VP (VBZ eat)
        (NP (NN spaghetti)
          (PP (IN with) (DT a) (NN fork)))
        (PP (IN in) (DT a) (NN restaurant))))
    """
    expected = read_trees(expected)[0]
    assert reconstructed == expected

def test_fix_close_open_ambiguous_later():
    """
    Test that a fix for an ambiguous close/open works as expected
    """
    trees = read_trees(CLOSE_OPEN_TWO_LABELS_TREE)
    assert len(trees) == 1
    tree = trees[0]

    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)
    assert isinstance(transitions[5], CloseConstituent)
    assert isinstance(transitions[6], OpenConstituent)

    reconstructed = reconstruct_tree(tree, transitions, transition_scheme=TransitionScheme.TOP_DOWN)
    assert tree == reconstructed

    new_transitions = fix_close_open_correct_open_ambiguous_later(transitions[5], OpenConstituent("PP"), transitions, 5, ROOT_LABELS, check_close=False)
    reconstructed = reconstruct_tree(tree, new_transitions, transition_scheme=TransitionScheme.TOP_DOWN)

    expected = """
    ( (VP (VBZ eat)
        (NP (NN spaghetti)
          (PP (IN with) (DT a) (NN fork))
          (PP (IN in) (DT a) (NN restaurant)))))
    """
    expected = read_trees(expected)[0]
    assert reconstructed == expected


SHIFT_CLOSE_EXAMPLES = [
    ("((S (NP (DT an) (NML (NNP Oct) (CD 19)) (NN review))))", "((S (NP (DT an) (NML (NNP Oct) (CD 19))) (NN review)))", 8),
    ("((S (NP (` `) (NP (DT The) (NN Misanthrope)) (` `) (PP (IN at) (NP (NNP Goodman) (NNP Theatre))))))",
     "((S (NP (` `) (NP (DT The)) (NN Misanthrope) (` `) (PP (IN at) (NP (NNP Goodman) (NNP Theatre))))))", 6),
    ("((S (NP (` `) (NP (DT The) (NN Misanthrope)) (` `) (PP (IN at) (NP (NNP Goodman) (NNP Theatre))))))",
     "((S (NP (` `) (NP (DT The) (NN Misanthrope))) (` `) (PP (IN at) (NP (NNP Goodman) (NNP Theatre)))))", 8),
    ("((S (NP (` `) (NP (DT The) (NN Misanthrope)) (` `) (PP (IN at) (NP (NNP Goodman) (NNP Theatre))))))",
     "((S (NP (` `) (NP (DT The) (NN Misanthrope)) (` `) (PP (IN at) (NP (NNP Goodman)) (NNP Theatre)))))", 13),
]

def test_shift_close():
    for idx, (orig_tree, expected_tree, shift_position) in enumerate(SHIFT_CLOSE_EXAMPLES):
        trees = read_trees(orig_tree)
        assert len(trees) == 1
        tree = trees[0]

        transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)
        if shift_position is None:
            print(transitions)
            continue

        assert isinstance(transitions[shift_position], Shift)
        new_transitions = fix_shift_close(Shift(), CloseConstituent(), transitions, shift_position, ROOT_LABELS)
        reconstructed = reconstruct_tree(tree, new_transitions, transition_scheme=TransitionScheme.TOP_DOWN)
        if expected_tree is None:
            print(transitions)
            print(new_transitions)

            print("{:P}".format(reconstructed))
        else:
            expected_tree = read_trees(expected_tree)
            assert len(expected_tree) == 1
            expected_tree = expected_tree[0]

            assert reconstructed == expected_tree

def test_shift_open_ambiguous_unary():
    """
    Test what happens if a Shift is turned into an Open in an ambiguous manner
    """
    trees = read_trees(CLOSE_SHIFT_AMBIGUOUS_TREE)
    assert len(trees) == 1
    tree = trees[0]

    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)
    expected_original = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), CloseConstituent(), Shift(), Shift(), CloseConstituent(), CloseConstituent()]
    assert transitions == expected_original

    new_sequence = fix_shift_open_ambiguous_unary(gold_transition=transitions[4],
                                                  pred_transition=OpenConstituent("ZZ"),
                                                  gold_sequence=transitions,
                                                  gold_index=4,
                                                  root_labels=ROOT_LABELS)
    expected_updated = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), OpenConstituent('ZZ'), Shift(), CloseConstituent(), Shift(), Shift(), CloseConstituent(), Shift(), Shift(), CloseConstituent(), CloseConstituent()]
    assert new_sequence == expected_updated

def test_shift_open_ambiguous_later():
    """
    Test what happens if a Shift is turned into an Open in an ambiguous manner
    """
    trees = read_trees(CLOSE_SHIFT_AMBIGUOUS_TREE)
    assert len(trees) == 1
    tree = trees[0]

    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)
    expected_original = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), CloseConstituent(), Shift(), Shift(), CloseConstituent(), CloseConstituent()]
    assert transitions == expected_original

    new_sequence = fix_shift_open_ambiguous_later(gold_transition=transitions[4],
                                                  pred_transition=OpenConstituent("ZZ"),
                                                  gold_sequence=transitions,
                                                  gold_index=4,
                                                  root_labels=ROOT_LABELS)
    expected_updated = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), OpenConstituent('ZZ'), Shift(), Shift(), Shift(), CloseConstituent(), CloseConstituent(), Shift(), Shift(), CloseConstituent(), CloseConstituent()]
    assert new_sequence == expected_updated
