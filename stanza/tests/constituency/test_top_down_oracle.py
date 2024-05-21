import pytest

from stanza.models.constituency.base_model import SimpleModel
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

def get_single_repair(gold_sequence, wrong_transition, repair_fn, idx, *args, **kwargs):
    return repair_fn(gold_sequence[idx], wrong_transition, gold_sequence, idx, ROOT_LABELS, None, None, *args, **kwargs)

def build_state(model, tree, num_transitions):
    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)
    states = model.initial_state_from_gold_trees([tree], [transitions])
    for idx, t in enumerate(transitions[:num_transitions]):
        assert t.is_legal(states[0], model), "Transition {} not legal at step {} in sequence {}".format(t, idx, sequence)
        states = model.bulk_apply(states, [t])
    state = states[0]
    return state

def test_fix_open_shift():
    trees = read_trees(OPEN_SHIFT_EXAMPLE_TREE)
    assert len(trees) == 1
    tree = trees[0]

    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)
    EXPECTED_ORIG = [OpenConstituent('ROOT'), OpenConstituent('S'), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), OpenConstituent('VP'), Shift(), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent(), CloseConstituent()]
    EXPECTED_FIX_EARLY = [OpenConstituent('ROOT'), OpenConstituent('S'), Shift(), Shift(), OpenConstituent('VP'), Shift(), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent(), CloseConstituent()]
    EXPECTED_FIX_LATE = [OpenConstituent('ROOT'), OpenConstituent('S'), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), OpenConstituent('VP'), Shift(), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent()]

    assert transitions == EXPECTED_ORIG

    new_transitions = get_single_repair(transitions, Shift(), fix_one_open_shift, 2)
    assert new_transitions == EXPECTED_FIX_EARLY

    new_transitions = get_single_repair(transitions, Shift(), fix_one_open_shift, 8)
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
    new_transitions = get_single_repair(transitions, Shift(), fix_one_open_shift, 2)
    assert new_transitions is None

    new_transitions = get_single_repair(transitions, Shift(), fix_multiple_open_shift, 2)

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
    new_transitions = get_single_repair(transitions, OpenConstituent('VP'), fix_open_open_ambiguous_unary, 2)
    assert new_transitions == EXPECTED_FIX


def test_open_open_ambiguous_later_fix():
    trees = read_trees(OPEN_SHIFT_EXAMPLE_TREE)
    assert len(trees) == 1
    tree = trees[0]

    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)
    EXPECTED_ORIG = [OpenConstituent('ROOT'), OpenConstituent('S'), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), OpenConstituent('VP'), Shift(), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent(), CloseConstituent()]
    EXPECTED_FIX = [OpenConstituent('ROOT'), OpenConstituent('S'), OpenConstituent('VP'), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), OpenConstituent('VP'), Shift(), OpenConstituent('NP'), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent(), CloseConstituent(), CloseConstituent()]
    assert transitions == EXPECTED_ORIG
    new_transitions = get_single_repair(transitions, OpenConstituent('VP'), fix_open_open_ambiguous_later, 2)
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
    new_sequence = get_single_repair(transitions, transitions[8], fix_close_shift_ambiguous_later, 7)
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
    new_sequence = get_single_repair(transitions, transitions[8], fix_close_shift_ambiguous_immediate, 7)
    expected_original = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), CloseConstituent(), Shift(), Shift(), CloseConstituent(), CloseConstituent()]
    expected_update = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), Shift(), CloseConstituent(), Shift(), CloseConstituent(), CloseConstituent()]
    assert transitions == expected_original
    assert new_sequence == expected_update

def test_oracle_with_optional_level():
    tree = read_trees(CLOSE_SHIFT_AMBIGUOUS_TREE)[0]
    gold_sequence = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), CloseConstituent(), Shift(), Shift(), CloseConstituent(), CloseConstituent()]
    expected_update = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), Shift(), CloseConstituent(), Shift(), CloseConstituent(), CloseConstituent()]

    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)
    assert transitions == gold_sequence

    oracle = TopDownOracle(ROOT_LABELS, 1, "", "")

    model = SimpleModel(transition_scheme=TransitionScheme.TOP_DOWN_UNARY, root_labels=ROOT_LABELS)
    state = build_state(model, tree, 7)
    fix, new_sequence = oracle.fix_error(pred_transition=gold_sequence[8],
                                         model=model,
                                         state=state)
    assert fix is RepairType.OTHER_CLOSE_SHIFT
    assert new_sequence is None

    oracle = TopDownOracle(ROOT_LABELS, 1, "CLOSE_SHIFT_AMBIGUOUS_IMMEDIATE_ERROR", "")
    fix, new_sequence = oracle.fix_error(pred_transition=gold_sequence[8],
                                         model=model,
                                         state=state)
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

    new_sequence = get_single_repair(transitions, transitions[8], fix_close_shift, 7)

    expected_original = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), CloseConstituent(), Shift(), CloseConstituent(), CloseConstituent()]
    expected_update   = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent()]
    assert transitions == expected_original
    assert new_sequence == expected_update

    # test that the one with two shifts, which is ambiguous, gets rejected
    trees = read_trees(CLOSE_SHIFT_AMBIGUOUS_TREE)
    assert len(trees) == 1
    tree = trees[0]

    transitions = build_sequence(tree, transition_scheme=TransitionScheme.TOP_DOWN)
    new_sequence = get_single_repair(transitions, transitions[8], fix_close_shift, 7)
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
        new_sequence = get_single_repair(transitions, transitions[10], fix_close_shift, 8, count_opens=count_opens)

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

    new_sequence = get_single_repair(transitions, transitions[9], fix_close_shift, 7, count_opens=False)
    assert new_sequence is None

    new_sequence = get_single_repair(transitions, transitions[9], fix_close_shift_with_opens, 7)

    expected_original = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), CloseConstituent(), OpenConstituent('NP'), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent()]
    expected_update   = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), Shift(), Shift(), Shift(), Shift(), CloseConstituent(), CloseConstituent(), CloseConstituent()]
    assert transitions == expected_original
    assert new_sequence == expected_update

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

    new_transitions = get_single_repair(transitions, transitions[6], fix_close_open_correct_open, 5)

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

        new_transitions = get_single_repair(transitions, OpenConstituent("PP"), fix_close_open_correct_open, 5)
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

    new_transitions = get_single_repair(transitions, OpenConstituent("PP"), fix_close_open_correct_open, 5, check_close=False)
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

    new_transitions = get_single_repair(transitions, OpenConstituent("PP"), fix_close_open_correct_open_ambiguous_later, 5, check_close=False)
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
        new_transitions = get_single_repair(transitions, CloseConstituent(), fix_shift_close, shift_position)
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

    new_sequence = get_single_repair(transitions, OpenConstituent("ZZ"), fix_shift_open_ambiguous_unary, 4)
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

    new_sequence = get_single_repair(transitions, OpenConstituent("ZZ"), fix_shift_open_ambiguous_later, 4)
    expected_updated = [OpenConstituent('ROOT'), OpenConstituent('NP'), Shift(), OpenConstituent('ADJP'), OpenConstituent('ZZ'), Shift(), Shift(), Shift(), CloseConstituent(), CloseConstituent(), Shift(), Shift(), CloseConstituent(), CloseConstituent()]
    assert new_sequence == expected_updated
