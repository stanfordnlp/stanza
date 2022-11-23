import itertools
import pytest

from stanza.models.constituency import dynamic_oracle
from stanza.models.constituency import parse_transitions
from stanza.models.constituency import tree_reader
from stanza.models.constituency.base_model import SimpleModel
from stanza.models.constituency.dynamic_oracle import *
from stanza.models.constituency.parse_transitions import CloseConstituent, OpenConstituent, Shift, TransitionScheme
from stanza.models.constituency.transition_sequence import build_treebank

from stanza.tests import *
from stanza.tests.constituency.test_transition_sequence import reconstruct_tree

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

# A sample tree from PTB with a single unary transition (at a location other than root)
SINGLE_UNARY_TREE = """
( (S
    (NP-SBJ-1 (DT A) (NN record) (NN date) )
    (VP (VBZ has) (RB n't)
      (VP (VBN been)
        (VP (VBN set)
          (NP (-NONE- *-1) ))))
    (. .) ))
"""

#  [Shift, OpenConstituent(('NP-SBJ-1',)), Shift, Shift, CloseConstituent, OpenConstituent(('S',)), Shift, OpenConstituent(('VP',)), Shift, Shift, OpenConstituent(('VP',)), Shift, OpenConstituent(('VP',)), Shift, OpenConstituent(('NP',)), CloseConstituent, CloseConstituent, CloseConstituent, CloseConstituent, Shift, CloseConstituent, OpenConstituent(('ROOT',)), CloseConstituent]

# A sample tree from PTB with a double unary transition (at a location other than root)
DOUBLE_UNARY_TREE = """
( (S
    (NP-SBJ
      (NP (RB Not) (PDT all) (DT those) )
      (SBAR
        (WHNP-3 (WP who) )
        (S
          (NP-SBJ (-NONE- *T*-3) )
          (VP (VBD wrote) ))))
    (VP (VBP oppose)
      (NP (DT the) (NNS changes) ))
    (. .) ))
"""

# A sample tree from PTB with a triple unary transition (at a location other than root)
# The triple unary is at the START of the next bracket, which affects how the
# dynamic oracle repairs the transition sequence
TRIPLE_UNARY_START_TREE = """
( (S
    (PRN
      (S
        (NP-SBJ (-NONE- *) )
        (VP (VB See) )))
    (, ,)
    (NP-SBJ
      (NP (DT the) (JJ other) (NN rule) )
      (PP (IN of)
        (NP (NN thumb) ))
      (PP (IN about)
        (NP (NN ballooning) )))))
"""

# A sample tree from PTB with a triple unary transition (at a location other than root)
# The triple unary is at the END of the next bracket, which affects how the
# dynamic oracle repairs the transition sequence
TRIPLE_UNARY_END_TREE = """
( (S
    (NP (NNS optimists) )
    (VP (VBP expect) 
      (S 
        (NP-SBJ-4 (NNP Hong) (NNP Kong) )
        (VP (TO to) 
          (VP (VB hum) 
            (ADVP-CLR (RB along) )
            (SBAR-MNR (RB as) 
              (S 
                (NP-SBJ (-NONE- *-4) )
                (VP (-NONE- *?*) 
                  (ADVP-TMP (IN before) ))))))))))
"""

TREES = [SINGLE_UNARY_TREE, DOUBLE_UNARY_TREE, TRIPLE_UNARY_START_TREE, TRIPLE_UNARY_END_TREE]
TREEBANK = "\n".join(TREES)

NOUN_PHRASE_TREE = """
( (NP
    (NP (NNP Chicago) (POS 's))
    (NNP Goodman)
    (NNP Theatre)))
"""

ROOT_LABELS = ["ROOT"]

def get_repairs(gold_sequence, wrong_transition, repair_fn):
    """
    Use the repair function and the wrong transition to iterate over the gold sequence

    Returns a list of possible repairs, one for each position in the sequence
    Repairs are tuples, (idx, seq)
    """
    repairs = [(idx, repair_fn(gold_transition, wrong_transition, gold_sequence, idx, ROOT_LABELS))
               for idx, gold_transition in enumerate(gold_sequence)]
    repairs = [x for x in repairs if x[1] is not None]
    return repairs

@pytest.fixture(scope="module")
def unary_trees():
    trees = tree_reader.read_trees(TREEBANK)
    trees = [t.prune_none().simplify_labels() for t in trees]
    assert len(trees) == len(TREES)

    return trees

@pytest.fixture(scope="module")
def gold_sequences(unary_trees):
    gold_sequences = build_treebank(unary_trees, TransitionScheme.IN_ORDER)
    return gold_sequences

def test_wrong_open_root(gold_sequences):
    """
    Test the results of the dynamic oracle on a few trees if the ROOT is mishandled.
    """
    wrong_transition = OpenConstituent("S")
    gold_transition = OpenConstituent("ROOT")
    close_transition = CloseConstituent()

    for gold_sequence in gold_sequences:
        # each of the sequences should be ended with ROOT, Close
        assert gold_sequence[-2] == gold_transition

        repairs = get_repairs(gold_sequence, wrong_transition, fix_wrong_open_root_error)
        # there is only spot in the sequence with a ROOT, so there should
        # be exactly one location which affords a S/ROOT replacement
        assert len(repairs) == 1
        repair = repairs[0]

        # the repair should occur at the -2 position, which is where ROOT is
        assert repair[0] == len(gold_sequence) - 2
        # and the resulting list should have the wrong transition followed by a Close
        # to give the model another chance to close the tree
        expected = gold_sequence[:-2] + [wrong_transition, close_transition] + gold_sequence[-2:]
        assert repair[1] == expected

def test_missed_unary(gold_sequences):
    """
    Test the repairs of an open/open error if it is effectively a skipped unary transition
    """
    wrong_transition = OpenConstituent("S")

    repairs = get_repairs(gold_sequences[0], wrong_transition, fix_wrong_open_unary_chain)
    assert len(repairs) == 0

    # here we are simulating picking NT-S instead of NT-VP
    # the DOUBLE_UNARY tree has one location where this is relevant, index 11
    repairs = get_repairs(gold_sequences[1], wrong_transition, fix_wrong_open_unary_chain)
    assert len(repairs) == 1
    assert repairs[0][0] == 11
    assert repairs[0][1] == gold_sequences[1][:11] + gold_sequences[1][13:]

    # the TRIPLE_UNARY_START tree has two locations where this is relevant
    # at index 1, the pattern goes (S (VP ...))
    # so choosing S instead of VP means you can skip the VP and only miss that one bracket
    # at index 5, the pattern goes (S (PRN (S (VP ...))) (...))
    # note that this is capturing a unary transition into a larger constituent
    # skipping the PRN is satisfactory
    repairs = get_repairs(gold_sequences[2], wrong_transition, fix_wrong_open_unary_chain)
    assert len(repairs) == 2
    assert repairs[0][0] == 1
    assert repairs[0][1] == gold_sequences[2][:1] + gold_sequences[2][3:]
    assert repairs[1][0] == 5
    assert repairs[1][1] == gold_sequences[2][:5] + gold_sequences[2][7:]

    # The TRIPLE_UNARY_END tree has 2 sections of tree for a total of 3 locations
    # where the repair might happen
    # Surprisingly the unary transition at the very start can only be
    # repaired by skipping it and using the outer S transition instead
    # The second repair overall (first repair in the second location)
    # should have a double skip to reach the S node
    repairs = get_repairs(gold_sequences[3], wrong_transition, fix_wrong_open_unary_chain)
    assert len(repairs) == 3
    assert repairs[0][0] == 1
    assert repairs[0][1] == gold_sequences[3][:1] + gold_sequences[3][3:]
    assert repairs[1][0] == 21
    assert repairs[1][1] == gold_sequences[3][:21] + gold_sequences[3][25:]
    assert repairs[2][0] == 23
    assert repairs[2][1] == gold_sequences[3][:23] + gold_sequences[3][25:]


def test_open_with_stuff(unary_trees, gold_sequences):
    wrong_transition = OpenConstituent("S")
    expected_trees = [
        "(ROOT (S (DT A) (NN record) (NN date) (VP (VBZ has) (RB n't) (VP (VBN been) (VP (VBN set)))) (. .)))",
        "(ROOT (S (NP (RB Not) (PDT all) (DT those)) (SBAR (WHNP (WP who)) (S (VP (VBD wrote)))) (VP (VBP oppose) (NP (DT the) (NNS changes))) (. .)))",
        None,
        "(ROOT (S (NP (NNS optimists)) (VP (VBP expect) (S (NNP Hong) (NNP Kong) (VP (TO to) (VP (VB hum) (ADVP (RB along)) (SBAR (RB as) (S (VP (ADVP (IN before)))))))))))"
    ]

    for tree, gold_sequence, expected in zip(unary_trees, gold_sequences, expected_trees):
        repairs = get_repairs(gold_sequence, wrong_transition, fix_wrong_open_stuff_unary)
        if expected is None:
            assert len(repairs) == 0
        else:
            assert len(repairs) == 1
            result = reconstruct_tree(tree, repairs[0][1])
            assert str(result) == expected

def test_general_open(gold_sequences):
    wrong_transition = OpenConstituent("SBARQ")

    for sequence in gold_sequences:
        repairs = get_repairs(sequence, wrong_transition, fix_wrong_open_general)
        assert len(repairs) == sum(isinstance(x, OpenConstituent) for x in sequence) - 1
        for repair in repairs:
            assert len(repair[1]) == len(sequence)
            assert repair[1][repair[0]] == wrong_transition
            assert repair[1][:repair[0]] == sequence[:repair[0]]
            assert repair[1][repair[0]+1:] == sequence[repair[0]+1:]

def test_missed_unary(unary_trees, gold_sequences):
    shift_transition = Shift()
    close_transition = CloseConstituent()

    expected_close_results = [
        [(12, 2)],
        [(11, 4), (13, 2)],
        # (NP NN thumb) and (NP NN ballooning) are both candidates for this repair
        [(18, 2), (24, 2)],
        [(21, 6), (23, 4), (25, 2)],
    ]

    expected_shift_results = [
        (),
        (),
        (),
        # (ADVP-CLR (RB along)) is followed by a shift
        [(16, 2)],
    ]

    for tree, sequence, expected_close, expected_shift in zip(unary_trees, gold_sequences, expected_close_results, expected_shift_results):
        repairs = get_repairs(sequence, close_transition, fix_missed_unary)
        assert len(repairs) == len(expected_close)
        for repair, (expected_idx, expected_len) in zip(repairs, expected_close):
            assert repair[0] == expected_idx
            assert repair[1] == sequence[:expected_idx] + sequence[expected_idx+expected_len:]

        repairs = get_repairs(sequence, shift_transition, fix_missed_unary)
        assert len(repairs) == len(expected_shift)
        for repair, (expected_idx, expected_len) in zip(repairs, expected_shift):
            assert repair[0] == expected_idx
            assert repair[1] == sequence[:expected_idx] + sequence[expected_idx+expected_len:]

def test_open_shift(unary_trees, gold_sequences):
    shift_transition = Shift()

    expected_repairs = [
        [(7,  "(ROOT (S (NP (DT A) (NN record) (NN date)) (VBZ has) (RB n't) (VP (VBN been) (VP (VBN set))) (. .)))"),
         (10, "(ROOT (S (NP (DT A) (NN record) (NN date)) (VP (VBZ has) (RB n't) (VBN been) (VP (VBN set))) (. .)))")],
        [(7,  "(ROOT (S (NP (NP (RB Not) (PDT all) (DT those)) (WP who) (S (VP (VBD wrote)))) (VP (VBP oppose) (NP (DT the) (NNS changes))) (. .)))"),
         (9,  "(ROOT (S (NP (NP (RB Not) (PDT all) (DT those)) (WHNP (WP who)) (S (VP (VBD wrote)))) (VP (VBP oppose) (NP (DT the) (NNS changes))) (. .)))"),
         (19, "(ROOT (S (NP (NP (RB Not) (PDT all) (DT those)) (SBAR (WHNP (WP who)) (S (VP (VBD wrote))))) (VBP oppose) (NP (DT the) (NNS changes)) (. .)))"),
         (21, "(ROOT (S (NP (NP (RB Not) (PDT all) (DT those)) (SBAR (WHNP (WP who)) (S (VP (VBD wrote))))) (VP (VBP oppose) (DT the) (NNS changes)) (. .)))")],
        [(14, "(ROOT (S (PRN (S (VP (VB See)))) (, ,) (NP (DT the) (JJ other) (NN rule)) (PP (IN of) (NP (NN thumb))) (PP (IN about) (NP (NN ballooning)))))"),
         (16, "(ROOT (S (PRN (S (VP (VB See)))) (, ,) (NP (NP (DT the) (JJ other) (NN rule)) (IN of) (NP (NN thumb)) (PP (IN about) (NP (NN ballooning))))))"),
         (22, "(ROOT (S (PRN (S (VP (VB See)))) (, ,) (NP (NP (DT the) (JJ other) (NN rule)) (PP (IN of) (NP (NN thumb))) (IN about) (NP (NN ballooning)))))")],
        [(5,  "(ROOT (S (NP (NNS optimists)) (VBP expect) (S (NP (NNP Hong) (NNP Kong)) (VP (TO to) (VP (VB hum) (ADVP (RB along)) (SBAR (RB as) (S (VP (ADVP (IN before))))))))))"),
         (10, "(ROOT (S (NP (NNS optimists)) (VP (VBP expect) (NP (NNP Hong) (NNP Kong)) (VP (TO to) (VP (VB hum) (ADVP (RB along)) (SBAR (RB as) (S (VP (ADVP (IN before))))))))))"),
         (12, "(ROOT (S (NP (NNS optimists)) (VP (VBP expect) (S (NP (NNP Hong) (NNP Kong)) (TO to) (VP (VB hum) (ADVP (RB along)) (SBAR (RB as) (S (VP (ADVP (IN before))))))))))"),
         (14, "(ROOT (S (NP (NNS optimists)) (VP (VBP expect) (S (NP (NNP Hong) (NNP Kong)) (VP (TO to) (VB hum) (ADVP (RB along)) (SBAR (RB as) (S (VP (ADVP (IN before))))))))))"),
         (19, "(ROOT (S (NP (NNS optimists)) (VP (VBP expect) (S (NP (NNP Hong) (NNP Kong)) (VP (TO to) (VP (VB hum) (ADVP (RB along)) (RB as) (S (VP (ADVP (IN before))))))))))")]
    ]

    for tree, sequence, expected in zip(unary_trees, gold_sequences, expected_repairs):
        repairs = get_repairs(sequence, shift_transition, fix_open_shift)
        assert len(repairs) == len(expected)
        for repair, (idx, expected_tree) in zip(repairs, expected):
            assert repair[0] == idx
            result_tree = reconstruct_tree(tree, repair[1])
            assert str(result_tree) == expected_tree


def test_open_close(unary_trees, gold_sequences):
    close_transition = CloseConstituent()

    expected_repairs = [
        [(7,  "(ROOT (S (S (NP (DT A) (NN record) (NN date)) (VBZ has)) (RB n't) (VP (VBN been) (VP (VBN set))) (. .)))"),
         (10, "(ROOT (S (NP (DT A) (NN record) (NN date)) (VP (VP (VBZ has) (RB n't) (VBN been)) (VP (VBN set))) (. .)))")],
        # missed the WHNP.  The surrounding SBAR cannot be created, either
        [(7, "(ROOT (S (NP (NP (NP (RB Not) (PDT all) (DT those)) (WP who)) (S (VP (VBD wrote)))) (VP (VBP oppose) (NP (DT the) (NNS changes))) (. .)))"),
         # missed the SBAR
         (9, "(ROOT (S (NP (NP (NP (RB Not) (PDT all) (DT those)) (WHNP (WP who))) (S (VP (VBD wrote)))) (VP (VBP oppose) (NP (DT the) (NNS changes))) (. .)))"),
         # missed the VP around "oppose the changes"
         (19, "(ROOT (S (S (NP (NP (RB Not) (PDT all) (DT those)) (SBAR (WHNP (WP who)) (S (VP (VBD wrote))))) (VBP oppose)) (NP (DT the) (NNS changes)) (. .)))"),
         # missed the NP in "the changes", looks pretty bad tbh
         (21, "(ROOT (S (NP (NP (RB Not) (PDT all) (DT those)) (SBAR (WHNP (WP who)) (S (VP (VBD wrote))))) (VP (VP (VBP oppose) (DT the)) (NNS changes)) (. .)))")],
        [(14, "(ROOT (S (S (PRN (S (VP (VB See)))) (, ,) (NP (DT the) (JJ other) (NN rule))) (PP (IN of) (NP (NN thumb))) (PP (IN about) (NP (NN ballooning)))))"),
         (16, "(ROOT (S (PRN (S (VP (VB See)))) (, ,) (NP (NP (NP (DT the) (JJ other) (NN rule)) (IN of)) (NP (NN thumb)) (PP (IN about) (NP (NN ballooning))))))"),
         (22, "(ROOT (S (PRN (S (VP (VB See)))) (, ,) (NP (NP (NP (DT the) (JJ other) (NN rule)) (PP (IN of) (NP (NN thumb))) (IN about)) (NP (NN ballooning)))))")],
        [(5, "(ROOT (S (S (NP (NNS optimists)) (VBP expect)) (S (NP (NNP Hong) (NNP Kong)) (VP (TO to) (VP (VB hum) (ADVP (RB along)) (SBAR (RB as) (S (VP (ADVP (IN before))))))))))"),
         (10, "(ROOT (S (NP (NNS optimists)) (VP (VP (VBP expect) (NP (NNP Hong) (NNP Kong))) (VP (TO to) (VP (VB hum) (ADVP (RB along)) (SBAR (RB as) (S (VP (ADVP (IN before))))))))))"),
         (12, "(ROOT (S (NP (NNS optimists)) (VP (VBP expect) (S (S (NP (NNP Hong) (NNP Kong)) (TO to)) (VP (VB hum) (ADVP (RB along)) (SBAR (RB as) (S (VP (ADVP (IN before))))))))))"),
         (14, "(ROOT (S (NP (NNS optimists)) (VP (VBP expect) (S (NP (NNP Hong) (NNP Kong)) (VP (VP (TO to) (VB hum)) (ADVP (RB along)) (SBAR (RB as) (S (VP (ADVP (IN before))))))))))"),
         (19, "(ROOT (S (NP (NNS optimists)) (VP (VBP expect) (S (NP (NNP Hong) (NNP Kong)) (VP (TO to) (VP (VP (VB hum) (ADVP (RB along)) (RB as)) (S (VP (ADVP (IN before))))))))))")]
    ]

    for tree, sequence, expected in zip(unary_trees, gold_sequences, expected_repairs):
        repairs = get_repairs(sequence, close_transition, fix_open_close)

        assert len(repairs) == len(expected)
        for repair, (idx, expected_tree) in zip(repairs, expected):
            assert repair[0] == idx
            result_tree = reconstruct_tree(tree, repair[1])
            assert str(result_tree) == expected_tree

def test_shift_close(unary_trees, gold_sequences):
    """
    Test the fix for a shift -> close

    These errors can occur pretty much everywhere, and the fix is quite simple,
    so we only test a few cases.
    """

    close_transition = CloseConstituent()

    expected_tree = "(ROOT (S (NP (NP (DT A)) (NN record) (NN date)) (VP (VBZ has) (RB n't) (VP (VBN been) (VP (VBN set)))) (. .)))"

    repairs = get_repairs(gold_sequences[0], close_transition, fix_shift_close)
    assert len(repairs) == 7
    result_tree = reconstruct_tree(unary_trees[0], repairs[0][1])
    assert str(result_tree) == expected_tree

    repairs = get_repairs(gold_sequences[1], close_transition, fix_shift_close)
    assert len(repairs) == 8

    repairs = get_repairs(gold_sequences[2], close_transition, fix_shift_close)
    assert len(repairs) == 8

    repairs = get_repairs(gold_sequences[3], close_transition, fix_shift_close)
    assert len(repairs) == 9
    for rep in repairs:
        if rep[0] == 16:
            # This one is special because it occurs as part of a unary
            # in other words, it should go unary, shift
            # and instead we are making it close where the unary should be
            # ... the unary would create "(ADVP (RB along))"
            expected_tree = "(ROOT (S (NP (NNS optimists)) (VP (VBP expect) (S (NP (NNP Hong) (NNP Kong)) (VP (TO to) (VP (VP (VB hum) (RB along)) (SBAR (RB as) (S (VP (ADVP (IN before)))))))))))"
            result_tree = reconstruct_tree(unary_trees[3], rep[1])
            assert str(result_tree) == expected_tree
            break
    else:
        raise AssertionError("Did not find an expected repair location")

def test_close_shift_nested(unary_trees, gold_sequences):
    shift_transition = Shift()

    expected_trees = [{},
                      {4: "(ROOT (S (NP (RB Not) (PDT all) (DT those) (SBAR (WHNP (WP who)) (S (VP (VBD wrote))))) (VP (VBP oppose) (NP (DT the) (NNS changes))) (. .)))"},
                      {13: "(ROOT (S (PRN (S (VP (VB See)))) (, ,) (NP (DT the) (JJ other) (NN rule) (PP (IN of) (NP (NN thumb))) (PP (IN about) (NP (NN ballooning))))))"},
                      {}]

    for tree, gold_sequence, expected in zip(unary_trees, gold_sequences, expected_trees):
        repairs = get_repairs(gold_sequence, shift_transition, fix_close_shift_nested)
        assert len(repairs) == len(expected)
        if len(expected) == 1:
            assert repairs[0][0] in expected.keys()
            result_tree = reconstruct_tree(tree, repairs[0][1])
            assert str(result_tree) == expected[repairs[0][0]]

def test_close_shift_shift(unary_trees):
    """
    Test that close -> shift works when there is a single block shifted after

    Includes a test specifically that there is no oracle action when there are two blocks after the missed close
    """
    shift_transition = Shift()

    expected_trees = [[(15, "(ROOT (S (NP (DT A) (NN record) (NN date)) (VP (VBZ has) (RB n't) (VP (VBN been) (VP (VBN set))) (. .))))")],
                      [(24, "(ROOT (S (NP (NP (RB Not) (PDT all) (DT those)) (SBAR (WHNP (WP who)) (S (VP (VBD wrote))))) (VP (VBP oppose) (NP (DT the) (NNS changes)) (. .))))")],
                      [(20, "(ROOT (S (PRN (S (VP (VB See)))) (, ,) (NP (NP (DT the) (JJ other) (NN rule)) (PP (IN of) (NP (NN thumb)) (PP (IN about) (NP (NN ballooning)))))))")],
                      [(17, "(ROOT (S (NP (NNS optimists)) (VP (VBP expect) (S (NP (NNP Hong) (NNP Kong)) (VP (TO to) (VP (VB hum) (ADVP (RB along) (SBAR (RB as) (S (VP (ADVP (IN before))))))))))))")],
                      []]

    np_trees = tree_reader.read_trees(NOUN_PHRASE_TREE)
    np_trees = [t.prune_none().simplify_labels() for t in np_trees]
    assert len(np_trees) == 1

    test_trees = unary_trees + np_trees
    gold_sequences = build_treebank(test_trees, TransitionScheme.IN_ORDER)

    for tree, gold_sequence, expected_repairs in zip(test_trees, gold_sequences, expected_trees):
        repairs = get_repairs(gold_sequence, shift_transition, fix_close_shift_shift)
        assert len(repairs) == len(expected_repairs)
        for repair, expected in zip(repairs, expected_repairs):
            assert repair[0] == expected[0]
            result_tree = reconstruct_tree(tree, repair[1])
            assert str(result_tree) == expected[1]

