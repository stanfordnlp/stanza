import pytest

from stanza.models import constituency_parser
from stanza.models.common import pretrain
from stanza.models.constituency import lstm_model
from stanza.models.constituency import parse_transitions
from stanza.models.constituency import parse_tree
from stanza.models.constituency import transition_sequence
from stanza.models.constituency import tree_reader
from stanza.tests import *
from stanza.tests import test_constituency_parse_transitions

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
def model():
    # TODO: build a fake embedding some other way?
    pt = pretrain.Pretrain(vec_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.xz', save_to_file=False)

    trees = tree_reader.read_trees(TREEBANK)

    transitions = transition_sequence.build_top_down_treebank(trees)
    transitions = transition_sequence.all_transitions(transitions)
    constituents = parse_tree.Tree.get_unique_constituent_labels(trees)
    tags = parse_tree.Tree.get_unique_tags(trees)
    root_labels = parse_tree.Tree.get_root_labels(trees)

    args = constituency_parser.parse_args(args=[])

    model = lstm_model.LSTMModel(pt, transitions, constituents, tags, root_labels, args)
    return model

def test_initial_model(model):
    # does nothing, just tests that the construction went okay
    pass
    
def test_initial_state(model):
    test_constituency_parse_transitions.test_initial_state(model)

def test_shift(model):
    test_constituency_parse_transitions.test_shift(model)

def test_unary(model):
    test_constituency_parse_transitions.test_unary(model)

def test_unary_requires_root(model):
    test_constituency_parse_transitions.test_unary_requires_root(model)

def test_open(model):
    test_constituency_parse_transitions.test_open(model)

def test_close(model):
    test_constituency_parse_transitions.test_close(model)

def test_forward(model):
    """
    Checks that the forward pass doesn't crash when run after various operations
    Doesn't check the forward pass for making reasonable answers
    """
    state = test_constituency_parse_transitions.build_initial_state(model)
    model(state)

    shift = parse_transitions.Shift()
    state = shift.apply(state, model)
    model(state)

    open_transition = parse_transitions.OpenConstituent("NP")
    assert open_transition.is_legal(state, model)
    state = open_transition.apply(state, model)
    assert state.num_opens == 1
    model(state)

    state = shift.apply(state, model)
    model(state)
    state = shift.apply(state, model)
    model(state)
    assert state.num_opens == 1
    # now should have "mox", "opal" on the constituents

    close_transition = parse_transitions.CloseConstituent()
    assert close_transition.is_legal(state, model)
    state = close_transition.apply(state, model)
    assert state.num_opens == 0
    model(state)
