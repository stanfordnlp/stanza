import logging
import tempfile

import pytest
import torch

from stanza import Pipeline

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

def build_trainer(wordvec_pretrain_file, *args, treebank=TREEBANK):
    # TODO: build a fake embedding some other way?
    train_trees = tree_reader.read_trees(treebank)
    dev_trees = train_trees[-1:]

    args = ['--wordvec_pretrain_file', wordvec_pretrain_file] + list(args)
    args = constituency_parser.parse_args(args)

    pt = trainer.load_pretrain(args['wordvec_pretrain_file'])
    forward_charlm = trainer.load_charlm(args['charlm_forward_file'])
    backward_charlm = trainer.load_charlm(args['charlm_backward_file'])
    # might be None, unless we're testing loading an existing model
    model_load_name = args['load_name']

    model, _, _ = trainer.build_trainer(args, train_trees, dev_trees, pt, forward_charlm, backward_charlm, None, None, model_load_name)
    assert isinstance(model.model, lstm_model.LSTMModel)
    return model

class TestTrainer:
    @pytest.fixture(scope="class")
    def wordvec_pretrain_file(self):
        return f'{TEST_WORKING_DIR}/in/tiny_emb.pt'

    def test_initial_model(self, wordvec_pretrain_file):
        """
        does nothing, just tests that the construction went okay
        """
        args = ['wordvec_pretrain_file', wordvec_pretrain_file]
        build_trainer(wordvec_pretrain_file)


    def test_save_load_model(self, wordvec_pretrain_file):
        """
        Just tests that saving and loading works without crashs.

        Currently no test of the values themselves
        """
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            tr = build_trainer(wordvec_pretrain_file)

            # attempt saving
            filename = os.path.join(tmpdirname, "parser.pt")
            tr.save(filename)

            assert os.path.exists(filename)

            # load it back in
            tr.load(filename)

    def test_relearn_structure(self, wordvec_pretrain_file):
        """
        Test that starting a trainer with --relearn_structure copies the old model
        """

        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            set_random_seed(1000, False)
            args = ['--pattn_num_layers', '0', '--lattn_d_proj', '0', '--hidden_size', '20', '--delta_embedding_dim', '10']
            tr = build_trainer(wordvec_pretrain_file, *args)

            # attempt saving
            filename = os.path.join(tmpdirname, "parser.pt")
            tr.save(filename)

            set_random_seed(1001, False)
            args = ['--pattn_num_layers', '1', '--lattn_d_proj', '0', '--hidden_size', '20', '--delta_embedding_dim', '10', '--relearn_structure', '--load_name', filename]
            tr2 = build_trainer(wordvec_pretrain_file, *args)

            assert torch.allclose(tr.model.delta_embedding.weight, tr2.model.delta_embedding.weight)
            assert torch.allclose(tr.model.output_layers[0].weight, tr2.model.output_layers[0].weight)
            # the norms will be the same, as the non-zero values are all the same
            assert torch.allclose(torch.linalg.norm(tr.model.word_lstm.weight_ih_l0), torch.linalg.norm(tr2.model.word_lstm.weight_ih_l0))

    def test_train(self, wordvec_pretrain_file):
        """
        Test the whole thing for a few iterations on the fake data
        """
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            train_treebank_file = os.path.join(tmpdirname, "train.mrg")
            with open(train_treebank_file, 'w', encoding='utf-8') as fout:
                fout.write(TREEBANK)
                fout.write(TREEBANK)

            eval_treebank_file = os.path.join(tmpdirname, "eval.mrg")
            with open(eval_treebank_file, 'w', encoding='utf-8') as fout:
                fout.write(TREEBANK)

            # let's not make the model huge...
            args = ['--pattn_num_layers', '0', '--lattn_d_proj', '0', '--hidden_size', '20', '--delta_embedding_dim', '10',
                    '--wordvec_pretrain_file', wordvec_pretrain_file, '--data_dir', tmpdirname, '--save_dir', tmpdirname, '--save_name', 'test.pt',
                    '--train_file', train_treebank_file, '--eval_file', eval_treebank_file,
                    '--epochs', '5', '--epoch_size', '6', '--train_batch_size', '3',
                    '--shorthand', 'en_test']
            args = constituency_parser.parse_args(args)
            # just in case we change the defaults in the future
            args['wandb'] = None

            save_name = os.path.join(args['save_dir'], args['save_name'])
            latest_name = os.path.join(args['save_dir'], 'latest.pt')
            assert not os.path.exists(save_name)
            retag_pipeline = Pipeline(lang="en", processors="tokenize, pos", tokenize_pretokenized=True)
            trainer.train(args, save_name, None, latest_name, retag_pipeline)

            # check that the model can be loaded back
            assert os.path.exists(save_name)
            tr = trainer.Trainer.load(save_name, load_optimizer=True)
            assert tr.optimizer is not None
            assert tr.scheduler is not None
            assert tr.epochs_trained >= 1

            tr = trainer.Trainer.load(latest_name, load_optimizer=True)
            assert tr.optimizer is not None
            assert tr.scheduler is not None
            assert tr.epochs_trained == 5
