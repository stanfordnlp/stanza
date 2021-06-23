import pytest
from stanza.models.constituency import parse_transitions
from stanza.models.constituency import transition_sequence
from stanza.models.constituency import tree_reader
from stanza.models.constituency.base_model import SimpleModel
from stanza.models.constituency.parse_transitions import *

from stanza.tests import *

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def test_top_down():
    text="((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"
    tree = tree_reader.read_trees(text)[0]

    model = SimpleModel()
    transitions = transition_sequence.build_top_down_sequence(tree)
    state = parse_transitions.initial_state_from_gold_tree(tree, model)

    for t in transitions:
        assert t.is_legal(state, model)
        state = t.apply(state, model)

    # one item for the final tree
    # one item for the sentinel at the end
    assert len(state.constituents) == 2
    # the transition sequence should put all of the words
    # from the buffer onto the tree
    # one spot left for the sentinel value
    assert len(state.word_queue) == 1
    assert len(state.transitions) == len(transitions) + 1

    result_tree = state.constituents.value
    assert result_tree == tree

def test_all_transitions():
    text="((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"
    trees = tree_reader.read_trees(text)
    model = SimpleModel()
    transitions = transition_sequence.build_top_down_treebank(trees)

    expected = [Shift(), CloseConstituent(), CompoundUnary("ROOT"), CompoundUnary("SQ"), CompoundUnary("WHNP"), OpenConstituent("NP"), OpenConstituent("PP"), OpenConstituent("SBARQ"), OpenConstituent("VP")]
    assert transition_sequence.all_transitions(transitions) == expected
