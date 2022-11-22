import pytest

from stanza.models.constituency import parse_transitions
from stanza.models.constituency.base_model import SimpleModel, UNARY_LIMIT
from stanza.models.constituency.parse_transitions import TransitionScheme, Shift, CloseConstituent, OpenConstituent
from stanza.tests import *

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]


def build_initial_state(model, num_states=1):
    words = ["Unban", "Mox", "Opal"]
    tags = ["VB", "NNP", "NNP"]
    sentences = [list(zip(words, tags)) for _ in range(num_states)]

    states = model.initial_state_from_words(sentences)
    assert len(states) == num_states
    assert all(state.num_transitions() == 0 for state in states)
    return states

def test_initial_state(model=None):
    if model is None:
        model = SimpleModel()
    states = build_initial_state(model)
    assert len(states) == 1
    state = states[0]

    assert state.sentence_length == 3
    assert state.num_opens == 0
    # each stack has a sentinel value at the end
    assert len(state.word_queue) == 5
    assert len(state.constituents) == 1
    assert len(state.transitions) == 1
    assert state.word_position == 0

def test_shift(model=None):
    if model is None:
        model = SimpleModel()
    state = build_initial_state(model)[0]

    open_transition = parse_transitions.OpenConstituent("ROOT")
    state = open_transition.apply(state, model)
    open_transition = parse_transitions.OpenConstituent("S")
    state = open_transition.apply(state, model)
    shift = parse_transitions.Shift()
    assert shift.is_legal(state, model)
    assert len(state.word_queue) == 5
    assert state.word_position == 0

    state = shift.apply(state, model)
    assert len(state.word_queue) == 5
    # 4 because of the dummy created by the opens
    assert len(state.constituents) == 4
    assert len(state.transitions) == 4
    assert shift.is_legal(state, model)
    assert state.word_position == 1
    assert not state.empty_word_queue()

    state = shift.apply(state, model)
    assert len(state.word_queue) == 5
    assert len(state.constituents) == 5
    assert len(state.transitions) == 5
    assert shift.is_legal(state, model)
    assert state.word_position == 2
    assert not state.empty_word_queue()

    state = shift.apply(state, model)
    assert len(state.word_queue) == 5
    assert len(state.constituents) == 6
    assert len(state.transitions) == 6
    assert not shift.is_legal(state, model)
    assert state.word_position == 3
    assert state.empty_word_queue()

    constituents = state.constituents
    assert model.get_top_constituent(constituents).children[0].label == 'Opal'
    constituents = constituents.pop()
    assert model.get_top_constituent(constituents).children[0].label == 'Mox'
    constituents = constituents.pop()
    assert model.get_top_constituent(constituents).children[0].label == 'Unban'

def test_initial_unary(model=None):
    # it doesn't make sense to start with a CompoundUnary
    if model is None:
        model = SimpleModel()

    state = build_initial_state(model)[0]
    unary = parse_transitions.CompoundUnary('ROOT', 'VP')
    assert unary.label == ('ROOT', 'VP',)
    assert not unary.is_legal(state, model)
    unary = parse_transitions.CompoundUnary('VP')
    assert unary.label == ('VP',)
    assert not unary.is_legal(state, model)


def test_unary(model=None):
    if model is None:
        model = SimpleModel()
    state = build_initial_state(model)[0]

    shift = parse_transitions.Shift()
    state = shift.apply(state, model)

    # this is technically the wrong parse but we're being lazy
    unary = parse_transitions.CompoundUnary('S', 'VP')
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
        model = SimpleModel(transition_scheme=TransitionScheme.TOP_DOWN_UNARY)
    state = build_initial_state(model)[0]

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
    state = build_initial_state(model)[0]

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
    state = build_initial_state(model)[0]
    state = shift.apply(state, model)
    state = shift.apply(state, model)
    state = shift.apply(state, model)
    assert not open_transition.is_legal(state, model)

def test_compound_open(model=None):
    if model is None:
        model = SimpleModel()
    state = build_initial_state(model)[0]

    open_transition = parse_transitions.OpenConstituent("ROOT", "S")
    assert open_transition.is_legal(state, model)
    shift = parse_transitions.Shift()
    close_transition = parse_transitions.CloseConstituent()

    state = open_transition.apply(state, model)
    state = shift.apply(state, model)
    state = shift.apply(state, model)
    state = shift.apply(state, model)
    state = close_transition.apply(state, model)

    tree = model.get_top_constituent(state.constituents)
    assert tree.label == 'ROOT'
    assert len(tree.children) == 1
    tree = tree.children[0]
    assert tree.label == 'S'
    assert len(tree.children) == 3
    assert tree.children[0].children[0].label == 'Unban'
    assert tree.children[1].children[0].label == 'Mox'
    assert tree.children[2].children[0].label == 'Opal'

def test_in_order_open(model=None):
    if model is None:
        model = SimpleModel(TransitionScheme.IN_ORDER)
    state = build_initial_state(model)[0]

    shift = parse_transitions.Shift()
    assert shift.is_legal(state, model)
    state = shift.apply(state, model)
    assert not shift.is_legal(state, model)

    open_vp = parse_transitions.OpenConstituent("VP")
    assert open_vp.is_legal(state, model)
    state = open_vp.apply(state, model)
    assert not open_vp.is_legal(state, model)

    close_trans = parse_transitions.CloseConstituent()
    assert close_trans.is_legal(state, model)
    state = close_trans.apply(state, model)

    open_s = parse_transitions.OpenConstituent("S")
    assert open_s.is_legal(state, model)
    state = open_s.apply(state, model)
    assert not open_vp.is_legal(state, model)

    # check that root transitions won't happen in the middle of a parse
    open_root = parse_transitions.OpenConstituent("ROOT")
    assert not open_root.is_legal(state, model)

    # build (NP (NNP Mox) (NNP Opal))
    open_np = parse_transitions.OpenConstituent("NP")
    assert shift.is_legal(state, model)
    state = shift.apply(state, model)
    assert open_np.is_legal(state, model)
    # make sure root can't happen in places where an arbitrary open is legal
    assert not open_root.is_legal(state, model)
    state = open_np.apply(state, model)
    assert shift.is_legal(state, model)
    state = shift.apply(state, model)
    assert close_trans.is_legal(state, model)
    state = close_trans.apply(state, model)

    assert close_trans.is_legal(state, model)
    state = close_trans.apply(state, model)

    assert open_root.is_legal(state, model)
    state = open_root.apply(state, model)

def test_too_many_unaries_close():
    """
    This tests rejecting Close at the start of a sequence after too many unary transitions

    The model should reject doing multiple "unaries" - eg, Open then Close - in an IN_ORDER sequence
    """
    model = SimpleModel(TransitionScheme.IN_ORDER)
    state = build_initial_state(model)[0]

    shift = parse_transitions.Shift()
    assert shift.is_legal(state, model)
    state = shift.apply(state, model)

    open_np = parse_transitions.OpenConstituent("NP")
    close_trans = parse_transitions.CloseConstituent()
    for _ in range(UNARY_LIMIT):
        assert open_np.is_legal(state, model)
        state = open_np.apply(state, model)

        assert close_trans.is_legal(state, model)
        state = close_trans.apply(state, model)

    assert open_np.is_legal(state, model)
    state = open_np.apply(state, model)
    assert not close_trans.is_legal(state, model)

def test_too_many_unaries_open():
    """
    This tests rejecting Open in the middle of a sequence after too many unary transitions

    The model should reject doing multiple "unaries" - eg, Open then Close - in an IN_ORDER sequence
    """
    model = SimpleModel(TransitionScheme.IN_ORDER)
    state = build_initial_state(model)[0]

    shift = parse_transitions.Shift()
    assert shift.is_legal(state, model)
    state = shift.apply(state, model)

    open_np = parse_transitions.OpenConstituent("NP")
    close_trans = parse_transitions.CloseConstituent()

    assert open_np.is_legal(state, model)
    state = open_np.apply(state, model)
    assert not open_np.is_legal(state, model)
    assert shift.is_legal(state, model)
    state = shift.apply(state, model)

    for _ in range(UNARY_LIMIT):
        assert open_np.is_legal(state, model)
        state = open_np.apply(state, model)

        assert close_trans.is_legal(state, model)
        state = close_trans.apply(state, model)

    assert not open_np.is_legal(state, model)

def test_close(model=None):
    if model is None:
        model = SimpleModel()

    # this one actually tests an entire subtree building
    state = build_initial_state(model)[0]

    open_transition_vp = parse_transitions.OpenConstituent("VP")
    assert open_transition_vp.is_legal(state, model)
    state = open_transition_vp.apply(state, model)
    assert state.num_opens == 1

    shift = parse_transitions.Shift()
    assert shift.is_legal(state, model)
    state = shift.apply(state, model)

    open_transition_np = parse_transitions.OpenConstituent("NP")
    assert open_transition_np.is_legal(state, model)
    state = open_transition_np.apply(state, model)
    assert state.num_opens == 2

    assert shift.is_legal(state, model)
    state = shift.apply(state, model)
    assert shift.is_legal(state, model)
    state = shift.apply(state, model)
    assert not shift.is_legal(state, model)
    assert state.num_opens == 2
    # now should have "mox", "opal" on the constituents

    close_transition = parse_transitions.CloseConstituent()
    assert close_transition.is_legal(state, model)
    state = close_transition.apply(state, model)
    assert state.num_opens == 1
    assert close_transition.is_legal(state, model)
    state = close_transition.apply(state, model)
    assert state.num_opens == 0
    assert not close_transition.is_legal(state, model)

    tree = model.get_top_constituent(state.constituents)
    assert tree.label == 'VP'
    assert len(tree.children) == 2
    tree = tree.children[1]
    assert tree.label == 'NP'
    assert len(tree.children) == 2
    assert tree.children[0].is_preterminal()
    assert tree.children[1].is_preterminal()
    assert tree.children[0].children[0].label == 'Mox'
    assert tree.children[1].children[0].label == 'Opal'

    # extra one for None at the start of the TreeStack
    assert len(state.constituents) == 2

    assert state.all_transitions(model) == [open_transition_vp, shift, open_transition_np, shift, shift, close_transition, close_transition]

def test_in_order_compound_finalize(model=None):
    """
    Test the Finalize transition is only legal at the end of a sequence
    """
    if model is None:
        model = SimpleModel(transition_scheme=TransitionScheme.IN_ORDER_COMPOUND)

    state = build_initial_state(model)[0]

    finalize = parse_transitions.Finalize("ROOT")

    shift = parse_transitions.Shift()
    assert shift.is_legal(state, model)
    assert not finalize.is_legal(state, model)
    state = shift.apply(state, model)

    open_transition = parse_transitions.OpenConstituent("NP")
    assert open_transition.is_legal(state, model)
    assert not finalize.is_legal(state, model)
    state = open_transition.apply(state, model)
    assert state.num_opens == 1

    assert shift.is_legal(state, model)
    assert not finalize.is_legal(state, model)
    state = shift.apply(state, model)
    assert shift.is_legal(state, model)
    assert not finalize.is_legal(state, model)
    state = shift.apply(state, model)

    close_transition = parse_transitions.CloseConstituent()
    assert close_transition.is_legal(state, model)
    state = close_transition.apply(state, model)
    assert state.num_opens == 0
    assert not close_transition.is_legal(state, model)
    assert finalize.is_legal(state, model)

    state = finalize.apply(state, model)
    assert not finalize.is_legal(state, model)
    tree = model.get_top_constituent(state.constituents)
    assert tree.label == 'ROOT'

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

    unary = parse_transitions.CompoundUnary("asdf", "zzzz")
    assert unary not in transitions
    transitions.add(unary)
    transitions.add(unary)
    transitions.add(unary)
    unary = parse_transitions.CompoundUnary("asdf", "zzzz")
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
    expected.append(parse_transitions.CompoundUnary("NP"))
    expected.append(parse_transitions.CompoundUnary("NP", "VP"))
    expected.append(parse_transitions.OpenConstituent("mox"))
    expected.append(parse_transitions.OpenConstituent("opal"))
    expected.append(parse_transitions.OpenConstituent("unban"))

    transitions = set(expected)
    transitions = sorted(transitions)
    assert transitions == expected

def test_check_transitions():
    """
    Test that check_transitions passes or fails a couple simple, small test cases
    """
    transitions = {Shift(), CloseConstituent(), OpenConstituent("NP"), OpenConstituent("VP")}

    other = {Shift(), CloseConstituent(), OpenConstituent("NP"), OpenConstituent("VP")}
    parse_transitions.check_transitions(transitions, other, "test")

    # This will get a pass because it is a unary made out of existing unaries
    other = {Shift(), CloseConstituent(), OpenConstituent("NP", "VP")}
    parse_transitions.check_transitions(transitions, other, "test")

    # This should fail
    with pytest.raises(RuntimeError):
        other = {Shift(), CloseConstituent(), OpenConstituent("NP", "ZP")}
        parse_transitions.check_transitions(transitions, other, "test")
