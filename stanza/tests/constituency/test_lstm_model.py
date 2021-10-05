import os

import pytest

from stanza.models.constituency import parse_transitions
from stanza.tests import *
from stanza.tests.constituency import test_parse_transitions
from stanza.tests.constituency.test_trainer import build_trainer, pt

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def build_model(pt, *args):
    trainer = build_trainer(pt, *args)
    return trainer.model

@pytest.fixture(scope="module")
def unary_model(pt):
    return build_model(pt, "--transition_scheme", "TOP_DOWN_UNARY")

def test_initial_state(unary_model):
    test_parse_transitions.test_initial_state(unary_model)

def test_shift(pt):
    # TODO: might be good to include some tests specifically for shift
    # in the context of a model with unaries
    model = build_model(pt)
    test_parse_transitions.test_shift(model)

def test_unary(unary_model):
    test_parse_transitions.test_unary(unary_model)

def test_unary_requires_root(unary_model):
    test_parse_transitions.test_unary_requires_root(unary_model)

def test_open(unary_model):
    test_parse_transitions.test_open(unary_model)

def test_compound_open(pt):
    model = build_model(pt, '--transition_scheme', "TOP_DOWN_COMPOUND")
    test_parse_transitions.test_compound_open(model)

def test_in_order_open(pt):
    model = build_model(pt, '--transition_scheme', "IN_ORDER")
    test_parse_transitions.test_in_order_open(model)

def test_close(unary_model):
    test_parse_transitions.test_close(unary_model)

def run_forward_checks(model):
    state = test_parse_transitions.build_initial_state(model)[0]
    model((state,))

    shift = parse_transitions.Shift()
    state = shift.apply(state, model)
    model((state,))

    open_transition = parse_transitions.OpenConstituent("NP")
    assert open_transition.is_legal(state, model)
    state = open_transition.apply(state, model)
    assert state.num_opens == 1
    model((state,))

    state = shift.apply(state, model)
    model((state,))
    state = shift.apply(state, model)
    model((state,))
    assert state.num_opens == 1
    # now should have "mox", "opal" on the constituents

    close_transition = parse_transitions.CloseConstituent()
    assert close_transition.is_legal(state, model)
    state = close_transition.apply(state, model)
    assert state.num_opens == 0

    model((state,))

def test_unary_forward(pt, unary_model):
    """
    Checks that the forward pass doesn't crash when run after various operations

    Doesn't check the forward pass for making reasonable answers
    """
    run_forward_checks(unary_model)

def test_lstm_forward(pt):
    model = build_model(pt, '--num_lstm_layers', '1')
    run_forward_checks(model)
    model = build_model(pt, '--num_lstm_layers', '2')
    run_forward_checks(model)
    model = build_model(pt, '--num_lstm_layers', '3')
    run_forward_checks(model)

def test_multiple_output_forward(pt):
    """
    Test a couple different sizes of output layers
    """
    model = build_model(pt, '--num_output_layers', '1', '--num_lstm_layers', '2')
    run_forward_checks(model)

    model = build_model(pt, '--num_output_layers', '2', '--num_lstm_layers', '2')
    run_forward_checks(model)

def test_no_tag_embedding_forward(pt):
    """
    Test that the model continues to work if the tag embedding is turned on or off
    """
    model = build_model(pt, '--tag_embedding_dim', '20')
    run_forward_checks(model)

    model = build_model(pt, '--tag_embedding_dim', '0')
    run_forward_checks(model)

def test_forward_con_lstm(pt):
    """
    Tests an older version of the model
    """
    model = build_model(pt, '--num_lstm_layers', '2', '--constituency_lstm')
    run_forward_checks(model)

def test_forward_combined_dummy(pt):
    """
    Tests combined dummy and open node embeddings
    """
    model = build_model(pt, '--combined_dummy_embedding')
    run_forward_checks(model)

    model = build_model(pt, '--no_combined_dummy_embedding')
    run_forward_checks(model)

def test_forward_charlm(pt):
    """
    Tests loading and running a charlm

    Note that this doesn't test the results of the charlm itself,
    just that the model is shaped correctly
    """
    forward_charlm_path = os.path.join(TEST_MODELS_DIR, "en", "forward_charlm", "1billion.pt")
    backward_charlm_path = os.path.join(TEST_MODELS_DIR, "en", "backward_charlm", "1billion.pt")
    assert os.path.exists(forward_charlm_path), "Need to download en test models (or update path to the forward charlm)"
    assert os.path.exists(backward_charlm_path), "Need to download en test models (or update path to the backward charlm)"

    model = build_model(pt, '--charlm_forward_file', forward_charlm_path, '--charlm_backward_file', backward_charlm_path)
    run_forward_checks(model)
