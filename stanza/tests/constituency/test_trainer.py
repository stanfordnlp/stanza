import logging
import tempfile

import pytest
import torch

from stanza.models import constituency_parser
from stanza.models.common import pretrain
from stanza.models.common.utils import set_random_seed
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

def build_trainer(pt, *args, treebank=TREEBANK):
    # TODO: build a fake embedding some other way?
    train_trees = tree_reader.read_trees(treebank)
    dev_trees = train_trees[-1:]

    args = constituency_parser.parse_args(args)
    forward_charlm = trainer.load_charlm(args['charlm_forward_file'])
    backward_charlm = trainer.load_charlm(args['charlm_backward_file'])
    model_load_name = args['load_name']

    model, _, _ = trainer.build_trainer(args, train_trees, dev_trees, pt, forward_charlm, backward_charlm, None, None, model_load_name)
    assert isinstance(model.model, lstm_model.LSTMModel)
    return model

class TestTrainer:
    @pytest.fixture(scope="class")
    def pt(self):
        return pretrain.Pretrain(vec_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.xz', save_to_file=False)

    def test_initial_model(self, pt):
        """
        does nothing, just tests that the construction went okay
        """
        build_trainer(pt)


    def test_save_load_model(self, pt):
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
            tr.load(filename, pt)

    def test_relearn_structure(self, pt):
        """
        Test that starting a trainer with --relearn_structure copies the old model
        """

        with tempfile.TemporaryDirectory() as tmpdirname:
            set_random_seed(1000, False)
            args = ['--pattn_num_layers', '0', '--lattn_d_proj', '0', '--hidden_size', '20', '--delta_embedding_dim', '10']
            tr = build_trainer(pt, *args)

            # attempt saving
            filename = os.path.join(tmpdirname, "parser.pt")
            tr.save(filename)

            set_random_seed(1001, False)
            args = ['--pattn_num_layers', '1', '--lattn_d_proj', '0', '--hidden_size', '20', '--delta_embedding_dim', '10', '--relearn_structure', '--load_name', filename]
            tr2 = build_trainer(pt, *args)

            assert torch.allclose(tr.model.delta_embedding.weight, tr2.model.delta_embedding.weight)
            assert torch.allclose(tr.model.output_layers[0].weight, tr2.model.output_layers[0].weight)
            # the norms will be the same, as the non-zero values are all the same
            assert torch.allclose(torch.linalg.norm(tr.model.word_lstm.weight_ih_l0), torch.linalg.norm(tr2.model.word_lstm.weight_ih_l0))
            
