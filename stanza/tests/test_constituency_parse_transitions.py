import pytest

from stanza.models.constituency import parse_transitions
from stanza.tests import *

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]


def build_initial_state():
    words = ["Unban", "Mox", "Opal"]
    tags = ["VB", "NNP", "NNP"]

    state = parse_transitions.initial_state_from_tagged_words(words, tags)
    return state

def test_initial_state():
    state = build_initial_state()

    assert state.sentence_length == 3
    assert state.num_opens == 0
    # each stack has a sentinel value at the end
    assert len(state.word_queue) == 4
    assert len(state.constituents) == 1
    assert len(state.transitions) == 1

def test_shift():
    state = build_initial_state()

    shift = parse_transitions.Shift()
    assert shift.is_legal(state)

    state = shift.apply(state)
    assert len(state.word_queue) == 3
    assert len(state.constituents) == 2
    assert len(state.transitions) == 2
    assert shift.is_legal(state)

    state = shift.apply(state)
    assert len(state.word_queue) == 2
    assert len(state.constituents) == 3
    assert len(state.transitions) == 3
    assert shift.is_legal(state)

    state = shift.apply(state)
    assert len(state.word_queue) == 1
    assert len(state.constituents) == 4
    assert len(state.transitions) == 4
    assert not shift.is_legal(state)

    constituents = state.constituents
    assert constituents.value.children[0].label == 'Opal'
    constituents = constituents.pop()
    assert constituents.value.children[0].label == 'Mox'
    constituents = constituents.pop()
    assert constituents.value.children[0].label == 'Unban'

def test_unary():
    state = build_initial_state()

    shift = parse_transitions.Shift()
    state = shift.apply(state)

    # this is technically the wrong parse but we're being lazy
    unary = parse_transitions.CompoundUnary(['S', 'VP'])
    assert unary.is_legal(state)
    state = unary.apply(state)
    assert not unary.is_legal(state)

    tree = state.constituents.value
    assert tree.label == 'S'
    assert len(tree.children) == 1
    tree = tree.children[0]
    assert tree.label == 'VP'
    assert len(tree.children) == 1
    tree = tree.children[0]
    assert tree.label == 'VB'
    assert tree.is_preterminal()

def test_open():
    state = build_initial_state()

    shift = parse_transitions.Shift()
    state = shift.apply(state)
    state = shift.apply(state)
    assert state.num_opens == 0

    open_transition = parse_transitions.OpenConstituent("VP")
    assert open_transition.is_legal(state)
    state = open_transition.apply(state)
    assert open_transition.is_legal(state)
    assert state.num_opens == 1

    # check that it is illegal if there are too many opens already
    for i in range(20):
        state = open_transition.apply(state)
    assert not open_transition.is_legal(state)
    assert state.num_opens == 21

    # check that it is illegal if the state is out of words
    state = build_initial_state()
    state = shift.apply(state)
    state = shift.apply(state)
    state = shift.apply(state)
    assert not open_transition.is_legal(state)

def test_close():
    # this one actually tests an entire subtree building
    state = build_initial_state()

    shift = parse_transitions.Shift()
    state = shift.apply(state)

    open_transition = parse_transitions.OpenConstituent("NP")
    assert open_transition.is_legal(state)
    state = open_transition.apply(state)
    assert state.num_opens == 1

    state = shift.apply(state)
    state = shift.apply(state)
    assert state.num_opens == 1
    # now should have "mox", "opal" on the constituents

    close_transition = parse_transitions.CloseConstituent()
    assert close_transition.is_legal(state)
    state = close_transition.apply(state)
    assert state.num_opens == 0
    assert not close_transition.is_legal(state)
    
    tree = state.constituents.value
    assert tree.label == 'NP'
    assert len(tree.children) == 2
    assert tree.children[0].is_preterminal()
    assert tree.children[1].is_preterminal()
    assert tree.children[0].children[0].label == 'Mox'
    assert tree.children[1].children[0].label == 'Opal'

    assert len(state.constituents) == 3
