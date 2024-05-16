import pytest

from stanza.models.constituency import in_order_compound_oracle
from stanza.models.constituency import tree_reader
from stanza.models.constituency.parse_transitions import CloseConstituent, OpenConstituent, Shift, TransitionScheme
from stanza.models.constituency.transition_sequence import build_treebank

from stanza.tests.constituency.test_transition_sequence import reconstruct_tree

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

# A sample tree from PTB with a triple unary transition (at a location other than root)
# Here we test the incorrect closing of various brackets
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

TREES = [TRIPLE_UNARY_START_TREE]
TREEBANK = "\n".join(TREES)

ROOT_LABELS = ["ROOT"]

@pytest.fixture(scope="module")
def trees():
    trees = tree_reader.read_trees(TREEBANK)
    trees = [t.prune_none().simplify_labels() for t in trees]
    assert len(trees) == len(TREES)

    return trees

@pytest.fixture(scope="module")
def gold_sequences(trees):
    gold_sequences = build_treebank(trees, TransitionScheme.IN_ORDER_COMPOUND)
    return gold_sequences

def get_repairs(gold_sequence, wrong_transition, repair_fn):
    """
    Use the repair function and the wrong transition to iterate over the gold sequence

    Returns a list of possible repairs, one for each position in the sequence
    Repairs are tuples, (idx, seq)
    """
    repairs = [(idx, repair_fn(gold_transition, wrong_transition, gold_sequence, idx, ROOT_LABELS, None, None))
               for idx, gold_transition in enumerate(gold_sequence)]
    repairs = [x for x in repairs if x[1] is not None]
    return repairs

def test_fix_shift_close():
    trees = tree_reader.read_trees(TRIPLE_UNARY_START_TREE)
    trees = [t.prune_none().simplify_labels() for t in trees]
    assert len(trees) == 1
    tree = trees[0]

    gold_sequences = build_treebank(trees, TransitionScheme.IN_ORDER_COMPOUND)

    # there are three places in this tree where a long bracket (more than 2 subtrees)
    # could theoretically be closed and then reopened
    repairs = get_repairs(gold_sequences[0], CloseConstituent(), in_order_compound_oracle.fix_shift_close_error)
    assert len(repairs) == 3

    expected_trees = ["(ROOT (S (S (PRN (S (VP (VB See)))) (, ,)) (NP (NP (DT the) (JJ other) (NN rule)) (PP (IN of) (NP (NN thumb))) (PP (IN about) (NP (NN ballooning))))))",
                      "(ROOT (S (PRN (S (VP (VB See)))) (, ,) (NP (NP (NP (DT the) (JJ other)) (NN rule)) (PP (IN of) (NP (NN thumb))) (PP (IN about) (NP (NN ballooning))))))",
                      "(ROOT (S (PRN (S (VP (VB See)))) (, ,) (NP (NP (NP (DT the) (JJ other) (NN rule)) (PP (IN of) (NP (NN thumb)))) (PP (IN about) (NP (NN ballooning))))))"]

    for repair, expected in zip(repairs, expected_trees):
        repaired_tree = reconstruct_tree(tree, repair[1], transition_scheme=TransitionScheme.IN_ORDER_COMPOUND)
        assert str(repaired_tree) == expected

def test_fix_open_close():
    trees = tree_reader.read_trees(TRIPLE_UNARY_START_TREE)
    trees = [t.prune_none().simplify_labels() for t in trees]
    assert len(trees) == 1
    tree = trees[0]

    gold_sequences = build_treebank(trees, TransitionScheme.IN_ORDER_COMPOUND)

    repairs = get_repairs(gold_sequences[0], CloseConstituent(), in_order_compound_oracle.fix_open_close_error)
    print("------------------")
    for repair in repairs:
        print(repair)
        repaired_tree = reconstruct_tree(tree, repair[1], transition_scheme=TransitionScheme.IN_ORDER_COMPOUND)
        print("{:P}".format(repaired_tree))
