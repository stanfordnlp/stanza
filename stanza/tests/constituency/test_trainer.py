from collections import defaultdict
import logging
import pathlib
import tempfile

import pytest
import torch
from torch import nn
from torch import optim

from stanza import Pipeline

from stanza.models import constituency_parser
from stanza.models.common import pretrain
from stanza.models.common.bert_embedding import load_bert, load_tokenizer
from stanza.models.common.foundation_cache import FoundationCache
from stanza.models.common.utils import set_random_seed
from stanza.models.constituency import lstm_model
from stanza.models.constituency.parse_transitions import Transition
from stanza.models.constituency import parser_training
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

    model, _, _, _ = parser_training.build_trainer(args, train_trees, dev_trees, silver_trees, foundation_cache, model_load_name)
    assert isinstance(model.model, lstm_model.LSTMModel)
    return model

class TestTrainer:
    @pytest.fixture(scope="class")
    def wordvec_pretrain_file(self):
        return f'{TEST_WORKING_DIR}/in/tiny_emb.pt'

    @pytest.fixture(scope="class")
    def tiny_random_xlnet(self, tmp_path_factory):
        """
        Download the tiny-random-xlnet model and make a concrete copy of it

        The issue here is that the "random" nature of the original
        makes it difficult or impossible to test that the values in
        the transformer don't change during certain operations.
        Saving a concrete instantiation of those random numbers makes
        it so we can test there is no difference when training only a
        subset of the layers, for example
        """
        xlnet_name = 'hf-internal-testing/tiny-random-xlnet'
        xlnet_model, xlnet_tokenizer = load_bert(xlnet_name)
        path = str(tmp_path_factory.mktemp('tiny-random-xlnet'))
        xlnet_model.save_pretrained(path)
        xlnet_tokenizer.save_pretrained(path)
        return path

    @pytest.fixture(scope="class")
    def tiny_random_bart(self, tmp_path_factory):
        """
        Download the tiny-random-bart model and make a concrete copy of it

        Issue is the same as with tiny_random_xlnet
        """
        bart_name = 'hf-internal-testing/tiny-random-bart'
        bart_model, bart_tokenizer = load_bert(bart_name)
        path = str(tmp_path_factory.mktemp('tiny-random-bart'))
        bart_model.save_pretrained(path)
        bart_tokenizer.save_pretrained(path)
        return path

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
        (checks some fields to make sure they are regenerated correctly)
        """
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            tr = build_trainer(wordvec_pretrain_file)
            transitions = tr.model.transitions

            # attempt saving
            filename = os.path.join(tmpdirname, "parser.pt")
            tr.save(filename)

            assert os.path.exists(filename)

            # load it back in
            tr2 = tr.load(filename)
            trans2 = tr2.model.transitions
            assert(transitions == trans2)
            assert all(isinstance(x, Transition) for x in trans2)

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
                '--save_dir', tmpdirname, '--save_name', 'test.pt', '--save_each_start', '0', '--save_each_name', os.path.join(tmpdirname, 'each_%02d.pt'),
                '--train_file', train_treebank_file, '--eval_file', eval_treebank_file,
                '--epoch_size', '6', '--train_batch_size', '3',
                '--shorthand', 'en_test']
        args = args + list(additional_args)
        args = constituency_parser.parse_args(args)
        # just in case we change the defaults in the future
        args['wandb'] = None
        return args

    def run_train_test(self, wordvec_pretrain_file, tmpdirname, num_epochs=5, extra_args=None, use_silver=False, exists_ok=False, foundation_cache=None):
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
        retag_pipeline = Pipeline(lang="en", processors="tokenize, pos", tokenize_pretokenized=True, dir=TEST_MODELS_DIR, foundation_cache=foundation_cache)
        trained_model = parser_training.train(args, None, [retag_pipeline])
        # check that hooks are in the model if expected
        for p in trained_model.model.parameters():
            if p.requires_grad:
                if args['grad_clipping'] is not None:
                    assert len(p._backward_hooks) == 1
                else:
                    assert p._backward_hooks is None

        # check that the model can be loaded back
        assert os.path.exists(args['save_name'])
        peft_name = trained_model.model.peft_name
        tr = trainer.Trainer.load(args['save_name'], load_optimizer=True, foundation_cache=retag_pipeline.foundation_cache, peft_name=trained_model.model.peft_name)
        assert tr.optimizer is not None
        assert tr.scheduler is not None
        assert tr.epochs_trained >= 1
        for p in tr.model.parameters():
            if p.requires_grad:
                assert p._backward_hooks is None

        tr = trainer.Trainer.load(args['checkpoint_save_name'], load_optimizer=True, foundation_cache=retag_pipeline.foundation_cache, peft_name=trained_model.model.peft_name)
        assert tr.optimizer is not None
        assert tr.scheduler is not None
        assert tr.epochs_trained == num_epochs

        for i in range(1, num_epochs+1):
            model_name = each_name % i
            assert os.path.exists(model_name)
            tr = trainer.Trainer.load(model_name, load_optimizer=True, foundation_cache=retag_pipeline.foundation_cache, peft_name=trained_model.model.peft_name)
            assert tr.epochs_trained == i
            assert tr.batches_trained == (4 * i if use_silver else 2 * i)

        return args, trained_model

    def test_train(self, wordvec_pretrain_file):
        """
        Test the whole thing for a few iterations on the fake data
        """
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            self.run_train_test(wordvec_pretrain_file, tmpdirname)

    def test_early_dropout(self, wordvec_pretrain_file):
        """
        Test the whole thing for a few iterations on the fake data
        """
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            args = ['--early_dropout', '3']
            _, model = self.run_train_test(wordvec_pretrain_file, tmpdirname, num_epochs=6, extra_args=args)
            model = model.model
            dropouts = [(name, module) for name, module in model.named_children() if isinstance(module, nn.Dropout)]
            assert len(dropouts) > 0, "Didn't find any dropouts in the model!"
            for name, module in dropouts:
                assert module.p == 0.0, "Dropout module %s was not set to 0 with early_dropout"

        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            # test that when turned off, early_dropout doesn't happen
            args = ['--early_dropout', '-1']
            _, model = self.run_train_test(wordvec_pretrain_file, tmpdirname, num_epochs=6, extra_args=args)
            model = model.model
            dropouts = [(name, module) for name, module in model.named_children() if isinstance(module, nn.Dropout)]
            assert len(dropouts) > 0, "Didn't find any dropouts in the model!"
            if all(module.p == 0.0 for _, module in dropouts):
                raise AssertionError("All dropouts were 0 after training even though early_dropout was set to -1")

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
            args, _ = self.run_train_test(wordvec_pretrain_file, tmpdirname, use_silver=False)
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
            args, _ = self.run_train_test(wordvec_pretrain_file, tmpdirname, num_epochs=8, extra_args=args)
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
                assert word_input_sizes[word_input_keys[0]] == [1, 2, 3]
                assert word_input_sizes[word_input_keys[1]] == [4, 5]
                assert word_input_sizes[word_input_keys[2]] == [6, 7, 8]
            else:
                # with no lattn, there are two stages: no attn, pattn
                assert len(word_input_sizes) == 2
                word_input_keys = sorted(word_input_sizes.keys())
                assert word_input_sizes[word_input_keys[0]] == [1, 2, 3]
                assert word_input_sizes[word_input_keys[1]] == [4, 5, 6, 7, 8]

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

    def bert_weights_allclose(self, bert_model, parser_model):
        """
        Return True if all bert weights are close, False otherwise
        """
        for name, parameter in bert_model.named_parameters():
            other_name = "bert_model." + name
            other_parameter = parser_model.model.get_parameter(other_name)
            if not torch.allclose(parameter.cpu(), other_parameter.cpu()):
                return False
        return True

    def frozen_transformer_test(self, wordvec_pretrain_file, transformer_name):
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            foundation_cache = FoundationCache()
            args = ['--bert_model', transformer_name]
            args, trained_model = self.run_train_test(wordvec_pretrain_file, tmpdirname, extra_args=args, foundation_cache=foundation_cache)
            bert_model, bert_tokenizer = foundation_cache.load_bert(transformer_name)
            assert self.bert_weights_allclose(bert_model, trained_model)

            checkpoint = torch.load(args['save_name'], lambda storage, loc: storage, weights_only=True)
            params = checkpoint['params']
            # check that the bert model wasn't saved in the model
            assert all(not x.startswith("bert_model.") for x in params['model'].keys())
            # make sure we're looking at the right thing
            assert any(x.startswith("output_layers.") for x in params['model'].keys())

            # check that the cached model is used as expected when loading a bert model
            trained_model = trainer.Trainer.load(args['save_name'], foundation_cache=foundation_cache)
            assert trained_model.model.bert_model is bert_model

    def test_bert_frozen(self, wordvec_pretrain_file):
        """
        Check that the parameters of the bert model don't change when training a basic model
        """
        self.frozen_transformer_test(wordvec_pretrain_file, 'hf-internal-testing/tiny-bert')

    def test_xlnet_frozen(self, wordvec_pretrain_file, tiny_random_xlnet):
        """
        Check that the parameters of an xlnet model don't change when training a basic model
        """
        self.frozen_transformer_test(wordvec_pretrain_file, tiny_random_xlnet)

    def test_bart_frozen(self, wordvec_pretrain_file, tiny_random_bart):
        """
        Check that the parameters of an xlnet model don't change when training a basic model
        """
        self.frozen_transformer_test(wordvec_pretrain_file, tiny_random_bart)

    def test_bert_finetune_one_epoch(self, wordvec_pretrain_file):
        """
        Check that the parameters the bert model DO change over a single training step
        """
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            transformer_name = 'hf-internal-testing/tiny-bert'
            args = ['--bert_model', transformer_name, '--bert_finetune', '--optim', 'adadelta']
            args, trained_model = self.run_train_test(wordvec_pretrain_file, tmpdirname, num_epochs=1, extra_args=args)

            # check that the weights are different
            foundation_cache = FoundationCache()
            bert_model, bert_tokenizer = foundation_cache.load_bert(transformer_name)
            assert not self.bert_weights_allclose(bert_model, trained_model)

            # double check that a new bert is created instead of using the FoundationCache when the bert has been trained
            model_name = args['save_name']
            assert os.path.exists(model_name)
            no_finetune_args = self.training_args(wordvec_pretrain_file, tmpdirname, None, None, "--no_bert_finetune", "--no_stage1_bert_finetune", '--bert_model', transformer_name)
            tr = trainer.Trainer.load(model_name, args=no_finetune_args, foundation_cache=foundation_cache)
            assert tr.model.bert_model is not bert_model
            assert not self.bert_weights_allclose(bert_model, tr)
            assert self.bert_weights_allclose(trained_model.model.bert_model, tr)

            new_save_name = os.path.join(tmpdirname, "test_resave_bert.pt")
            assert not os.path.exists(new_save_name)
            tr.save(new_save_name, save_optimizer=False)
            tr2 = trainer.Trainer.load(new_save_name, args=no_finetune_args, foundation_cache=foundation_cache)
            # check that the resaved model included its finetuned bert weights
            assert tr2.model.bert_model is not bert_model
            # the finetuned bert weights should also be scheduled for saving the next time as well
            assert not tr2.model.is_unsaved_module("bert_model")

    def finetune_transformer_test(self, wordvec_pretrain_file, transformer_name):
        """
        Check that the parameters of the transformer DO change when using bert_finetune
        """
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            args = ['--bert_model', transformer_name, '--bert_finetune', '--optim', 'adamw']
            args, trained_model = self.run_train_test(wordvec_pretrain_file, tmpdirname, extra_args=args)

            # check that the weights are different
            foundation_cache = FoundationCache()
            bert_model, bert_tokenizer = foundation_cache.load_bert(transformer_name)
            assert not self.bert_weights_allclose(bert_model, trained_model)

            # double check that a new bert is created instead of using the FoundationCache when the bert has been trained
            no_finetune_args = self.training_args(wordvec_pretrain_file, tmpdirname, None, None, "--no_bert_finetune", "--no_stage1_bert_finetune", '--bert_model', transformer_name)
            trained_model = trainer.Trainer.load(args['save_name'], args=no_finetune_args, foundation_cache=foundation_cache)
            assert not trained_model.model.args['bert_finetune']
            assert not trained_model.model.args['stage1_bert_finetune']
            assert trained_model.model.bert_model is not bert_model

    def test_bert_finetune(self, wordvec_pretrain_file):
        """
        Check that the parameters of a bert model DO change when using bert_finetune
        """
        self.finetune_transformer_test(wordvec_pretrain_file, 'hf-internal-testing/tiny-bert')

    def test_xlnet_finetune(self, wordvec_pretrain_file, tiny_random_xlnet):
        """
        Check that the parameters of an xlnet model DO change when using bert_finetune
        """
        self.finetune_transformer_test(wordvec_pretrain_file, tiny_random_xlnet)

    def test_stage1_bert_finetune(self, wordvec_pretrain_file):
        """
        Check that the parameters the bert model DO change when using stage1_bert_finetune, but only for the first couple steps
        """
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            bert_model_name = 'hf-internal-testing/tiny-bert'
            args = ['--bert_model', bert_model_name, '--stage1_bert_finetune', '--optim', 'adamw']
            # need to use num_epochs==6 so that epochs 1 and 2 are saved to be different
            # a test of 5 or less means that sometimes it will reload the params
            # at step 2 to get ready for the following iterations with adamw
            args, trained_model = self.run_train_test(wordvec_pretrain_file, tmpdirname, num_epochs=6, extra_args=args)

            # check that the weights are different
            foundation_cache = FoundationCache()
            bert_model, bert_tokenizer = foundation_cache.load_bert(bert_model_name)
            assert not self.bert_weights_allclose(bert_model, trained_model)

            # double check that a new bert is created instead of using the FoundationCache when the bert has been trained
            no_finetune_args = self.training_args(wordvec_pretrain_file, tmpdirname, None, None, "--no_bert_finetune", "--no_stage1_bert_finetune", '--bert_model', bert_model_name, '--optim', 'adamw')
            num_epochs = trained_model.model.args['epochs']
            each_name = os.path.join(tmpdirname, 'each_%02d.pt')
            for i in range(1, num_epochs+1):
                model_name = each_name % i
                assert os.path.exists(model_name)
                tr = trainer.Trainer.load(model_name, args=no_finetune_args, foundation_cache=foundation_cache)
                assert tr.model.bert_model is not bert_model
                assert not self.bert_weights_allclose(bert_model, tr)
                if i >= num_epochs // 2:
                    assert self.bert_weights_allclose(trained_model.model.bert_model, tr)

            # verify that models 1 and 2 are saved to be different
            model_name_1 = each_name % 1
            model_name_2 = each_name % 2
            tr_1 = trainer.Trainer.load(model_name_1, args=no_finetune_args, foundation_cache=foundation_cache)
            tr_2 = trainer.Trainer.load(model_name_2, args=no_finetune_args, foundation_cache=foundation_cache)
            assert not self.bert_weights_allclose(tr_1.model.bert_model, tr_2)


    def one_layer_finetune_transformer_test(self, wordvec_pretrain_file, transformer_name):
        """
        Check that the parameters the bert model DO change when using bert_finetune
        """
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            args = ['--bert_model', transformer_name, '--bert_finetune', '--bert_finetune_layers', '1', '--optim', 'adamw', '--bert_finetune_layers', '1']
            args, trained_model = self.run_train_test(wordvec_pretrain_file, tmpdirname, extra_args=args)

            # check that the weights of the last layer are different,
            # but the weights of the earlier layers and
            # non-transformer-layers are the same
            foundation_cache = FoundationCache()
            bert_model, bert_tokenizer = foundation_cache.load_bert(transformer_name)
            assert bert_model.config.num_hidden_layers > 1
            layer_name = "layer.%d." % (bert_model.config.num_hidden_layers - 1)
            for name, parameter in bert_model.named_parameters():
                other_name = "bert_model." + name
                other_parameter = trained_model.model.get_parameter(other_name)
                if layer_name in name:
                    if 'rel_attn.seg_embed' in name or 'rel_attn.r_s_bias' in name:
                        # not sure why this happens for xlnet, just roll with it
                        continue
                    assert not torch.allclose(parameter.cpu(), other_parameter.cpu())
                else:
                    assert torch.allclose(parameter.cpu(), other_parameter.cpu())

    def test_bert_finetune_one_layer(self, wordvec_pretrain_file):
        self.one_layer_finetune_transformer_test(wordvec_pretrain_file, 'hf-internal-testing/tiny-bert')

    def test_xlnet_finetune_one_layer(self, wordvec_pretrain_file, tiny_random_xlnet):
        self.one_layer_finetune_transformer_test(wordvec_pretrain_file, tiny_random_xlnet)

    def test_peft_finetune(self, tmp_path, wordvec_pretrain_file):
        transformer_name = 'hf-internal-testing/tiny-bert'
        args = ['--bert_model', transformer_name, '--bert_finetune', '--optim', 'adamw', '--use_peft']
        args, trained_model = self.run_train_test(wordvec_pretrain_file, str(tmp_path), extra_args=args)

    def test_peft_twostage_finetune(self, wordvec_pretrain_file):
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            num_epochs = 6
            transformer_name = 'hf-internal-testing/tiny-bert'
            args = ['--bert_model', transformer_name, '--stage1_bert_finetune', '--optim', 'adamw', '--use_peft']
            args, trained_model = self.run_train_test(wordvec_pretrain_file, tmpdirname, num_epochs=num_epochs, extra_args=args)
            for epoch in range(num_epochs):
                filename_prev = args['save_each_name'] % epoch
                filename_next = args['save_each_name'] % (epoch+1)
                trainer_prev = trainer.Trainer.load(filename_prev, args=args, load_optimizer=False)
                trainer_next = trainer.Trainer.load(filename_next, args=args, load_optimizer=False)

                lora_names = [name for name, _ in trainer_prev.model.bert_model.named_parameters() if name.find("lora") >= 0]
                if epoch < 2:
                    assert not any(torch.allclose(trainer_prev.model.bert_model.get_parameter(name).cpu(),
                                                  trainer_next.model.bert_model.get_parameter(name).cpu())
                                   for name in lora_names)
                elif epoch > 2:
                    assert all(torch.allclose(trainer_prev.model.bert_model.get_parameter(name).cpu(),
                                              trainer_next.model.bert_model.get_parameter(name).cpu())
                               for name in lora_names)
