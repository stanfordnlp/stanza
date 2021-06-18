import pytest
from stanza.models.constituency import parse_transitions
from stanza.models.constituency import transition_sequence
from stanza.models.constituency import tree_reader
from stanza.models.constituency.base_model import SimpleModel
from stanza.models.constituency.parse_transitions import *

from stanza.tests import *

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def check_reproduce_tree(transition_scheme):
    text="((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"
    trees = tree_reader.read_trees(text)

    model = SimpleModel(transition_scheme)
    transitions = transition_sequence.build_sequence(trees[0], transition_scheme)
    states = parse_transitions.initial_state_from_gold_trees(trees, model)
    assert(len(states)) == 1
    state = states[0]
    assert state.num_transitions() == 0

    for t in transitions:
        assert t.is_legal(state, model)
        state = t.apply(state, model)

    # one item for the final tree
    # one item for the sentinel at the end
    assert len(state.constituents) == 2
    # the transition sequence should put all of the words
    # from the buffer onto the tree
    # one spot left for the sentinel value
    assert len(state.word_queue) == 7
    assert state.sentence_length == 6
    assert state.word_position == state.sentence_length
    assert len(state.transitions) == len(transitions) + 1

    result_tree = state.constituents.value
    assert result_tree == trees[0]

def test_top_down_unary():
    check_reproduce_tree(transition_scheme=TransitionScheme.TOP_DOWN_UNARY)

def test_top_down_no_unary():
    check_reproduce_tree(transition_scheme=TransitionScheme.TOP_DOWN)

def test_in_order():
    check_reproduce_tree(transition_scheme=TransitionScheme.IN_ORDER)

def test_all_transitions():
    text="((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"
    trees = tree_reader.read_trees(text)
    model = SimpleModel()
    transitions = transition_sequence.build_treebank(trees)

    expected = [Shift(), CloseConstituent(), CompoundUnary("ROOT"), CompoundUnary("SQ"), CompoundUnary("WHNP"), OpenConstituent("NP"), OpenConstituent("PP"), OpenConstituent("SBARQ"), OpenConstituent("VP")]
    assert transition_sequence.all_transitions(transitions) == expected


def test_all_transitions_no_unary():
    text="((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"
    trees = tree_reader.read_trees(text)
    model = SimpleModel()
    transitions = transition_sequence.build_treebank(trees, transition_scheme=TransitionScheme.TOP_DOWN)

    expected = [Shift(), CloseConstituent(), OpenConstituent("NP"), OpenConstituent("PP"), OpenConstituent("ROOT"), OpenConstituent("SBARQ"), OpenConstituent("SQ"), OpenConstituent("VP"), OpenConstituent("WHNP")]
    assert transition_sequence.all_transitions(transitions) == expected

def test_top_down_compound_unary():
    text = "(ROOT (S (NP (DT The) (NNP Arizona) (NNPS Corporations) (NNP Commission)) (VP (VBD authorized) (NP (NP (DT an) (ADJP (CD 11.5)) (NN %) (NN rate) (NN increase)) (PP (IN at) (NP (NNP Tucson) (NNP Electric) (NNP Power) (NNP Co.))) (, ,) (UCP (ADJP (ADJP (RB substantially) (JJR lower)) (SBAR (IN than) (S (VP (VBN recommended) (NP (JJ last) (NN month)) (PP (IN by) (NP (DT a) (NN commission) (NN hearing) (NN officer))))))) (CC and) (NP (NP (QP (RB barely) (PDT half)) (DT the) (NN rise)) (VP (VBN sought) (PP (IN by) (NP (DT the) (NN utility)))))))) (. .)))"

    trees = tree_reader.read_trees(text)
    assert len(trees) == 1

    model = SimpleModel()
    transitions = transition_sequence.build_sequence(trees[0], transition_scheme=TransitionScheme.TOP_DOWN_COMPOUND)

    states = parse_transitions.initial_state_from_gold_trees(trees, model)
    assert len(states) == 1
    state = states[0]

    for t in transitions:
        assert t.is_legal(state, model)
        state = t.apply(state, model)

    result = model.get_top_constituent(state.constituents)
    assert trees[0] == result
