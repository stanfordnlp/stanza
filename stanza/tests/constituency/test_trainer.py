from collections import defaultdict
import logging
import pathlib
import tempfile

import pytest
import torch
from torch import optim

from stanza import Pipeline

from stanza.models import constituency_parser
from stanza.models.common import pretrain
from stanza.models.common.foundation_cache import FoundationCache
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
    silver_trees = []

    args = ['--wordvec_pretrain_file', wordvec_pretrain_file] + list(args)
    args = constituency_parser.parse_args(args)

    foundation_cache = FoundationCache()
    # might be None, unless we're testing loading an existing model
    model_load_name = args['load_name']

    model, _, _, _ = trainer.build_trainer(args, train_trees, dev_trees, silver_trees, foundation_cache, model_load_name)
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
            set_random_seed(1000)
            args = ['--pattn_num_layers', '0', '--lattn_d_proj', '0', '--hidden_size', '20', '--delta_embedding_dim', '10']
            tr = build_trainer(wordvec_pretrain_file, *args)

            # attempt saving
            filename = os.path.join(tmpdirname, "parser.pt")
            tr.save(filename)

            set_random_seed(1001)
            args = ['--pattn_num_layers', '1', '--lattn_d_proj', '0', '--hidden_size', '20', '--delta_embedding_dim', '10', '--relearn_structure', '--load_name', filename]
            tr2 = build_trainer(wordvec_pretrain_file, *args)

            assert torch.allclose(tr.model.delta_embedding.weight, tr2.model.delta_embedding.weight)
            assert torch.allclose(tr.model.output_layers[0].weight, tr2.model.output_layers[0].weight)
            # the norms will be the same, as the non-zero values are all the same
            assert torch.allclose(torch.linalg.norm(tr.model.word_lstm.weight_ih_l0), torch.linalg.norm(tr2.model.word_lstm.weight_ih_l0))

    def write_treebanks(self, tmpdirname):
        train_treebank_file = os.path.join(tmpdirname, "train.mrg")
        with open(train_treebank_file, 'w', encoding='utf-8') as fout:
            fout.write(TREEBANK)
            fout.write(TREEBANK)

        eval_treebank_file = os.path.join(tmpdirname, "eval.mrg")
        with open(eval_treebank_file, 'w', encoding='utf-8') as fout:
            fout.write(TREEBANK)

        return train_treebank_file, eval_treebank_file

    def training_args(self, wordvec_pretrain_file, tmpdirname, train_treebank_file, eval_treebank_file, *additional_args):
        # let's not make the model huge...
        args = ['--pattn_num_layers', '0', '--pattn_d_model', '128', '--lattn_d_proj', '0', '--use_lattn', '--hidden_size', '20', '--delta_embedding_dim', '10',
                '--wordvec_pretrain_file', wordvec_pretrain_file, '--data_dir', tmpdirname,
                '--save_dir', tmpdirname, '--save_name', 'test.pt', '--save_each_name', os.path.join(tmpdirname, 'each_%02d.pt'),
                '--train_file', train_treebank_file, '--eval_file', eval_treebank_file,
                '--epoch_size', '6', '--train_batch_size', '3',
                '--shorthand', 'en_test']
        args = args + list(additional_args)
        args = constituency_parser.parse_args(args)
        # just in case we change the defaults in the future
        args['wandb'] = None
        return args

    def run_train_test(self, wordvec_pretrain_file, tmpdirname, num_epochs=5, extra_args=None, use_silver=False, exists_ok=False):
        """
        Runs a test of the trainer for a few iterations.

        Checks some basic properties of the saved model, but doesn't
        check for the accuracy of the results
        """
        if extra_args is None:
            extra_args = []
        extra_args += ['--epochs', '%d' % num_epochs]

        train_treebank_file, eval_treebank_file = self.write_treebanks(tmpdirname)
        if use_silver:
            extra_args += ['--silver_file', str(eval_treebank_file)]
        args = self.training_args(wordvec_pretrain_file, tmpdirname, train_treebank_file, eval_treebank_file, *extra_args)

        each_name = args['save_each_name']
        if not exists_ok:
            assert not os.path.exists(args['save_name'])
        retag_pipeline = Pipeline(lang="en", processors="tokenize, pos", tokenize_pretokenized=True, dir=TEST_MODELS_DIR)
        tr = trainer.train(args, None, each_name, [retag_pipeline])
        # check that hooks are in the model if expected
        for p in tr.model.parameters():
            if p.requires_grad:
                if args['grad_clipping'] is not None:
                    assert len(p._backward_hooks) == 1
                else:
                    assert p._backward_hooks is None

        # check that the model can be loaded back
        assert os.path.exists(args['save_name'])
        tr = trainer.Trainer.load(args['save_name'], load_optimizer=True)
        assert tr.optimizer is not None
        assert tr.scheduler is not None
        assert tr.epochs_trained >= 1
        for p in tr.model.parameters():
            if p.requires_grad:
                assert p._backward_hooks is None

        tr = trainer.Trainer.load(args['checkpoint_save_name'], load_optimizer=True)
        assert tr.optimizer is not None
        assert tr.scheduler is not None
        assert tr.epochs_trained == num_epochs

        for i in range(1, num_epochs+1):
            model_name = each_name % i
            assert os.path.exists(model_name)
            tr = trainer.Trainer.load(model_name, load_optimizer=True)
            assert tr.epochs_trained == i
            assert tr.batches_trained == (4 * i if use_silver else 2 * i)

        return args

    def test_train(self, wordvec_pretrain_file):
        """
        Test the whole thing for a few iterations on the fake data
        """
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            self.run_train_test(wordvec_pretrain_file, tmpdirname)

    def test_train_silver(self, wordvec_pretrain_file):
        """
        Test the whole thing for a few iterations on the fake data

        This tests that it works if you give it a silver file
        The check for the use of the silver data is that the
        number of batches trained should go up
        """
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            self.run_train_test(wordvec_pretrain_file, tmpdirname, use_silver=True)

    def test_train_checkpoint(self, wordvec_pretrain_file):
        """
        Test the whole thing for a few iterations, then restart

        This tests that the 5th iteration save file is not rewritten
        and that the iterations continue to 10

        TODO: could make it more robust by verifying that only 5 more
        epochs are trained.  Perhaps a "most recent epochs" could be
        saved in the trainer
        """
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            args = self.run_train_test(wordvec_pretrain_file, tmpdirname, use_silver=False)
            save_5 = args['save_each_name'] % 5
            save_10 = args['save_each_name'] % 10
            assert os.path.exists(save_5)
            assert not os.path.exists(save_10)

            save_5_stat = pathlib.Path(save_5).stat()

            self.run_train_test(wordvec_pretrain_file, tmpdirname, num_epochs=10, use_silver=False, exists_ok=True)
            assert os.path.exists(save_5)
            assert os.path.exists(save_10)

            assert pathlib.Path(save_5).stat().st_mtime == save_5_stat.st_mtime

    def run_multistage_tests(self, wordvec_pretrain_file, tmpdirname, use_lattn, extra_args=None):
            train_treebank_file, eval_treebank_file = self.write_treebanks(tmpdirname)
            args = ['--multistage', '--pattn_num_layers', '1']
            if use_lattn:
                args += ['--lattn_d_proj', '16']
            if extra_args:
                args += extra_args
            args = self.run_train_test(wordvec_pretrain_file, tmpdirname, num_epochs=8, extra_args=args)
            each_name = os.path.join(args['save_dir'], 'each_%02d.pt')

            word_input_sizes = defaultdict(list)
            for i in range(1, 9):
                model_name = each_name % i
                assert os.path.exists(model_name)
                tr = trainer.Trainer.load(model_name, load_optimizer=True)
                assert tr.epochs_trained == i
                word_input_sizes[tr.model.word_input_size].append(i)
            if use_lattn:
                # there should be three stages: no attn, pattn, pattn+lattn
                assert len(word_input_sizes) == 3
                word_input_keys = sorted(word_input_sizes.keys())
                assert word_input_sizes[word_input_keys[0]] == [1, 2, 3, 4]
                assert word_input_sizes[word_input_keys[1]] == [5, 6]
                assert word_input_sizes[word_input_keys[2]] == [7, 8]
            else:
                # with no lattn, there are two stages: no attn, pattn
                assert len(word_input_sizes) == 2
                word_input_keys = sorted(word_input_sizes.keys())
                assert word_input_sizes[word_input_keys[0]] == [1, 2, 3, 4]
                assert word_input_sizes[word_input_keys[1]] == [5, 6, 7, 8]

    def test_multistage_lattn(self, wordvec_pretrain_file):
        """
        Test a multistage training for a few iterations on the fake data

        This should start with no pattn or lattn, have pattn in the middle, then lattn at the end
        """
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            self.run_multistage_tests(wordvec_pretrain_file, tmpdirname, use_lattn=True)

    def test_multistage_no_lattn(self, wordvec_pretrain_file):
        """
        Test a multistage training for a few iterations on the fake data

        This should start with no pattn or lattn, have pattn in the middle, then lattn at the end
        """
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            self.run_multistage_tests(wordvec_pretrain_file, tmpdirname, use_lattn=False)

    def test_multistage_optimizer(self, wordvec_pretrain_file):
        """
        Test that the correct optimizers are built for a multistage training process
        """
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            extra_args = ['--optim', 'adamw']
            self.run_multistage_tests(wordvec_pretrain_file, tmpdirname, use_lattn=False, extra_args=extra_args)

            # check that the optimizers which get rebuilt when loading
            # the models are adadelta for the first half of the
            # multistage, then adamw
            each_name = os.path.join(tmpdirname, 'each_%02d.pt')
            for i in range(1, 3):
                model_name = each_name % i
                tr = trainer.Trainer.load(model_name, load_optimizer=True)
                assert tr.epochs_trained == i
                assert isinstance(tr.optimizer, optim.Adadelta)
                # double check that this is actually a valid test
                assert not isinstance(tr.optimizer, optim.AdamW)

            for i in range(4, 8):
                model_name = each_name % i
                tr = trainer.Trainer.load(model_name, load_optimizer=True)
                assert tr.epochs_trained == i
                assert isinstance(tr.optimizer, optim.AdamW)


    def test_grad_clip_hooks(self, wordvec_pretrain_file):
        """
        Verify that grad clipping is not saved with the model, but is attached at training time
        """
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            args = ['--grad_clipping', '25']
            self.run_train_test(wordvec_pretrain_file, tmpdirname, extra_args=args)

    def test_analyze_trees(self, wordvec_pretrain_file):
        test_str = "(ROOT (S (NP (PRP I)) (VP (VBP wan) (S (VP (TO na) (VP (VB lick) (NP (NP (NNP Sh'reyan) (POS 's)) (NNS antennae))))))))  (ROOT (S (NP (DT This) (NN interface)) (VP (VBZ sucks))))"

        test_tree = tree_reader.read_trees(test_str)
        assert len(test_tree) == 2

        args = ['--pattn_num_layers', '0', '--lattn_d_proj', '0', '--hidden_size', '20', '--delta_embedding_dim', '10']
        tr = build_trainer(wordvec_pretrain_file, *args)

        results = tr.model.analyze_trees(test_tree)
        assert len(results) == 2
        assert len(results[0].predictions) == 1
        assert results[0].predictions[0].tree == test_tree[0]
        assert results[0].state is not None
        assert isinstance(results[0].state.score, torch.Tensor)
        assert results[0].state.score.shape == torch.Size([])
        assert len(results[0].constituents) == 9
        assert results[0].constituents[-1].value == test_tree[0]
        # the way the results are built, the next-to-last entry
        # should be the thing just below the root
        assert results[0].constituents[-2].value == test_tree[0].children[0]

        assert len(results[1].predictions) == 1
        assert results[1].predictions[0].tree == test_tree[1]
        assert results[1].state is not None
        assert isinstance(results[1].state.score, torch.Tensor)
        assert results[1].state.score.shape == torch.Size([])
        assert len(results[1].constituents) == 4
        assert results[1].constituents[-1].value == test_tree[1]
        assert results[1].constituents[-2].value == test_tree[1].children[0]
