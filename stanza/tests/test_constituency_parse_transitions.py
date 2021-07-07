import pytest

from stanza.models.constituency import parse_transitions
from stanza.models.constituency.base_model import SimpleModel
from stanza.tests import *

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]


def build_initial_state(model):
    words = ["Unban", "Mox", "Opal"]
    tags = ["VB", "NNP", "NNP"]

    state = parse_transitions.initial_state_from_tagged_words(words, tags, model)
    return state

def test_initial_state(model=None):
    if model is None:
        model = SimpleModel()
    state = build_initial_state(model)

    assert state.sentence_length == 3
    assert state.num_opens == 0
    # each stack has a sentinel value at the end
    assert len(state.word_queue) == 4
    assert len(state.constituents) == 1
    assert len(state.transitions) == 1

def test_shift(model=None):
    if model is None:
        model = SimpleModel()
    state = build_initial_state(model)

    open_transition = parse_transitions.OpenConstituent("S")
    state = open_transition.apply(state, model)
    shift = parse_transitions.Shift()
    assert shift.is_legal(state, model)

    state = shift.apply(state, model)
    assert len(state.word_queue) == 3
    # 3 because of the dummy created by the open
    assert len(state.constituents) == 3
    assert len(state.transitions) == 3
    assert shift.is_legal(state, model)

    state = shift.apply(state, model)
    assert len(state.word_queue) == 2
    assert len(state.constituents) == 4
    assert len(state.transitions) == 4
    assert shift.is_legal(state, model)

    state = shift.apply(state, model)
    assert len(state.word_queue) == 1
    assert len(state.constituents) == 5
    assert len(state.transitions) == 5
    assert not shift.is_legal(state, model)

    constituents = state.constituents
    assert model.get_top_constituent(constituents).children[0].label == 'Opal'
    constituents = constituents.pop()
    assert model.get_top_constituent(constituents).children[0].label == 'Mox'
    constituents = constituents.pop()
    assert model.get_top_constituent(constituents).children[0].label == 'Unban'

def test_unary(model=None):
    if model is None:
        model = SimpleModel()
    state = build_initial_state(model)

    shift = parse_transitions.Shift()
    state = shift.apply(state, model)

    # this is technically the wrong parse but we're being lazy
    unary = parse_transitions.CompoundUnary(['S', 'VP'])
    assert unary.is_legal(state, model)
    state = unary.apply(state, model)
    assert not unary.is_legal(state, model)

    tree = model.get_top_constituent(state.constituents)
    assert tree.label == 'S'
    assert len(tree.children) == 1
    tree = tree.children[0]
    assert tree.label == 'VP'
    assert len(tree.children) == 1
    tree = tree.children[0]
    assert tree.label == 'VB'
    assert tree.is_preterminal()

def test_unary_requires_root(model=None):
    if model is None:
        model = SimpleModel()
    state = build_initial_state(model)

    open_transition = parse_transitions.OpenConstituent("S")
    assert open_transition.is_legal(state, model)
    state = open_transition.apply(state, model)

    shift = parse_transitions.Shift()
    assert shift.is_legal(state, model)
    state = shift.apply(state, model)
    assert shift.is_legal(state, model)
    state = shift.apply(state, model)
    assert shift.is_legal(state, model)
    state = shift.apply(state, model)
    assert not shift.is_legal(state, model)

    close_transition = parse_transitions.CloseConstituent()
    assert close_transition.is_legal(state, model)
    state = close_transition.apply(state, model)
    assert not open_transition.is_legal(state, model)
    assert not close_transition.is_legal(state, model)

    np_unary = parse_transitions.CompoundUnary("NP")
    assert not np_unary.is_legal(state, model)
    root_unary = parse_transitions.CompoundUnary("ROOT")
    assert root_unary.is_legal(state, model)
    assert not state.finished(model)
    state = root_unary.apply(state, model)
    assert not root_unary.is_legal(state, model)

    assert state.finished(model)

def test_open(model=None):
    if model is None:
        model = SimpleModel()
    state = build_initial_state(model)

    shift = parse_transitions.Shift()
    state = shift.apply(state, model)
    state = shift.apply(state, model)
    assert state.num_opens == 0

    open_transition = parse_transitions.OpenConstituent("VP")
    assert open_transition.is_legal(state, model)
    state = open_transition.apply(state, model)
    assert open_transition.is_legal(state, model)
    assert state.num_opens == 1

    # check that it is illegal if there are too many opens already
    for i in range(20):
        state = open_transition.apply(state, model)
    assert not open_transition.is_legal(state, model)
    assert state.num_opens == 21

    # check that it is illegal if the state is out of words
    state = build_initial_state(model)
    state = shift.apply(state, model)
    state = shift.apply(state, model)
    state = shift.apply(state, model)
    assert not open_transition.is_legal(state, model)

def test_close(model=None):
    if model is None:
        model = SimpleModel()
    # this one actually tests an entire subtree building
    state = build_initial_state(model)

    shift = parse_transitions.Shift()
    state = shift.apply(state, model)

    open_transition = parse_transitions.OpenConstituent("NP")
    assert open_transition.is_legal(state, model)
    state = open_transition.apply(state, model)
    assert state.num_opens == 1

    state = shift.apply(state, model)
    state = shift.apply(state, model)
    assert state.num_opens == 1
    # now should have "mox", "opal" on the constituents

    close_transition = parse_transitions.CloseConstituent()
    assert close_transition.is_legal(state, model)
    state = close_transition.apply(state, model)
    assert state.num_opens == 0
    assert not close_transition.is_legal(state, model)
    
    tree = model.get_top_constituent(state.constituents)
    assert tree.label == 'NP'
    assert len(tree.children) == 2
    assert tree.children[0].is_preterminal()
    assert tree.children[1].is_preterminal()
    assert tree.children[0].children[0].label == 'Mox'
    assert tree.children[1].children[0].label == 'Opal'

    assert len(state.constituents) == 3

    assert state.all_transitions(model) == [shift, open_transition, shift, shift, close_transition]

def test_hashes():
    transitions = set()

    shift = parse_transitions.Shift()
    assert shift not in transitions
    transitions.add(shift)
    assert shift in transitions
    shift = parse_transitions.Shift()
    assert shift in transitions

    for i in range(5):
        transitions.add(shift)
    assert len(transitions) == 1

    unary = parse_transitions.CompoundUnary("asdf")
    assert unary not in transitions
    transitions.add(unary)
    assert unary in transitions

    unary = parse_transitions.CompoundUnary(["asdf", "zzzz"])
    assert unary not in transitions
    transitions.add(unary)
    transitions.add(unary)
    transitions.add(unary)
    unary = parse_transitions.CompoundUnary(["asdf", "zzzz"])
    assert unary in transitions

    # check that the str and the list constructors result in the same item
    assert len(transitions) == 3
    unary = parse_transitions.CompoundUnary(["asdf"])
    assert unary in transitions

    oc = parse_transitions.OpenConstituent("asdf")
    assert oc not in transitions
    transitions.add(oc)
    assert oc in transitions
    transitions.add(oc)
    transitions.add(oc)
    assert len(transitions) == 4
    assert parse_transitions.OpenConstituent("asdf") in transitions

    cc = parse_transitions.CloseConstituent()
    assert cc not in transitions
    transitions.add(cc)
    transitions.add(cc)
    transitions.add(cc)
    assert cc in transitions
    cc = parse_transitions.CloseConstituent()
    assert cc in transitions
    assert len(transitions) == 5


def test_sort():
    expected = []

    expected.append(parse_transitions.Shift())
    expected.append(parse_transitions.CloseConstituent())
    expected.append(parse_transitions.CompoundUnary(["NP"]))
    expected.append(parse_transitions.CompoundUnary(["NP", "VP"]))
    expected.append(parse_transitions.OpenConstituent("mox"))
    expected.append(parse_transitions.OpenConstituent("opal"))
    expected.append(parse_transitions.OpenConstituent("unban"))

    transitions = set(expected)
    transitions = sorted(transitions)
    assert transitions == expected
