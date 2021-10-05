import logging
import tempfile

import pytest

from stanza.models import constituency_parser
from stanza.models.common import pretrain
from stanza.models.constituency import lstm_model
from stanza.models.constituency import trainer
from stanza.models.constituency import tree_reader
from stanza.tests import *

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

logger = logging.getLogger('stanza.constituency.trainer')
logger.setLevel(logging.WARNING)

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

def build_trainer(pt, *args):
    # TODO: build a fake embedding some other way?
    train_trees = tree_reader.read_trees(TREEBANK)
    dev_trees = train_trees[-1:]

    args = constituency_parser.parse_args(args)
    forward_charlm = trainer.load_charlm(args['charlm_forward_file'])
    backward_charlm = trainer.load_charlm(args['charlm_backward_file'])

    model, _, _ = trainer.build_trainer(args, train_trees, dev_trees, pt, forward_charlm, backward_charlm)
    assert isinstance(model.model, lstm_model.LSTMModel)
    return model

def test_initial_model(pt):
    """
    does nothing, just tests that the construction went okay
    """
    build_trainer(pt)


def test_save_load_model(pt):
    """
    Just tests that saving and loading works without crashs.

    Currently no test of the values themselves
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        tr = build_trainer(pt)

        # attempt saving
        filename = os.path.join(tmpdirname, "parser.pt")
        tr.save(filename)

        assert os.path.exists(filename)

        # load it back in
        tr.load(filename, pt, None, None, False)
