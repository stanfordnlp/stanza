import os
import tempfile

import pytest

from stanza.models import constituency_parser
from stanza.models.common import pretrain
from stanza.models.constituency import lstm_model
from stanza.models.constituency import parse_transitions
from stanza.models.constituency import parse_tree
from stanza.models.constituency import transition_sequence
from stanza.models.constituency import tree_reader
from stanza.tests import *
from stanza.tests.constituency import test_parse_transitions

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

TREEBANK = """
( (S
    (VP (VBG Enjoying)
      (NP (PRP$  my) (JJ favorite) (NN Friday) (NN tradition)))
    (. .)))

( (NP
    (VP (VBG Sitting)
      (PP (IN in)
        (NP (DT a) (RB stifling) (JJ hot) (NNP South) (NNP Station)))
      (VP (VBG waiting)
        (PP (IN for)
          (NP (PRP$  my) (JJ delayed) (NNP @MBTA) (NN train)))))
    (. .)))

( (S
    (NP (PRP I))
    (VP 
      (ADVP (RB really))
      (VBP hate)
      (NP (DT the) (NNP @MBTA)))))

( (S
    (S (VP (VB Seek)))
    (CC and)
    (S (NP (PRP ye))
      (VP (MD shall)
        (VP (VB find))))
    (. .)))
"""

@pytest.fixture(scope="module")
def pt():
    return pretrain.Pretrain(vec_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.xz', save_to_file=False)

def build_model(pt, *args):
    # TODO: build a fake embedding some other way?
    trees = tree_reader.read_trees(TREEBANK)

    args = constituency_parser.parse_args(args)

    transitions = constituency_parser.build_treebank(trees, args)
    transitions = transition_sequence.all_transitions(transitions)
    constituents = parse_tree.Tree.get_unique_constituent_labels(trees)
    tags = parse_tree.Tree.get_unique_tags(trees)
    words = parse_tree.Tree.get_unique_words(trees)
    rare_words = parse_tree.Tree.get_rare_words(trees)
    root_labels = parse_tree.Tree.get_root_labels(trees)
    open_nodes = constituency_parser.get_open_nodes(trees, args)

    model = lstm_model.LSTMModel(pt, transitions, constituents, tags, words, rare_words, root_labels, open_nodes, args)
    return model

@pytest.fixture(scope="module")
def unary_model(pt):
    return build_model(pt, "--use_compound_unary")

def test_initial_model(unary_model):
    # does nothing, just tests that the construction went okay
    pass
    
def test_initial_state(unary_model):
    test_parse_transitions.test_initial_state(unary_model)

def test_shift(unary_model):
    test_parse_transitions.test_shift(unary_model)

def test_unary(unary_model):
    test_parse_transitions.test_unary(unary_model)

def test_unary_requires_root(unary_model):
    test_parse_transitions.test_unary_requires_root(unary_model)

def test_open(unary_model):
    test_parse_transitions.test_open(unary_model)

def test_compound_open(pt):
    model = build_model(pt, '--use_compound_open')
    test_parse_transitions.test_compound_open(model)

def test_close(unary_model):
    test_parse_transitions.test_close(unary_model)

def run_forward_checks(model):
    state = test_parse_transitions.build_initial_state(model)
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

def test_forward(pt, unary_model):
    """
    Checks that the forward pass doesn't crash when run after various operations

    Doesn't check the forward pass for making reasonable answers
    """
    run_forward_checks(unary_model)
    model = build_model(pt, '--num_lstm_layers', '1')
    run_forward_checks(model)
    model = build_model(pt, '--num_lstm_layers', '2')
    run_forward_checks(model)
    model = build_model(pt, '--num_lstm_layers', '3')
    run_forward_checks(model)

def test_save_load_model(pt, unary_model):
    """
    Just tests that saving and loading works without crashs.

    Currently no test of the values themselves
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, "parser.pt")
        lstm_model.save(filename, unary_model)
        assert os.path.exists(filename)
        foo = lstm_model.load(filename, pt)
