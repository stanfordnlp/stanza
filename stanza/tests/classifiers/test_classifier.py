import glob
import os

import pytest

import numpy as np
import torch

import stanza
import stanza.models.classifier as classifier
import stanza.models.classifiers.data as data
from stanza.models.classifiers.trainer import Trainer
from stanza.models.common import pretrain
from stanza.models.common import utils

from stanza.tests import TEST_MODELS_DIR
from stanza.tests.classifiers.test_data import train_file, dev_file, test_file, DATASET, SENTENCES

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

EMB_DIM = 5

@pytest.fixture(scope="module")
def fake_embeddings(tmp_path_factory):
    """
    will return a path to a fake embeddings file with the words in SENTENCES
    """
    # could set np random seed here
    words = sorted(set([x.lower() for y in SENTENCES for x in y]))
    words = words[:-1]
    embedding_dir = tmp_path_factory.mktemp("data")
    embedding_txt = embedding_dir / "embedding.txt"
    embedding_pt  = embedding_dir / "embedding.pt"
    embedding = np.random.random((len(words), EMB_DIM))

    with open(embedding_txt, "w", encoding="utf-8") as fout:
        for word, emb in zip(words, embedding):
            fout.write(word)
            fout.write("\t")
            fout.write("\t".join(str(x) for x in emb))
            fout.write("\n")

    pt = pretrain.Pretrain(str(embedding_pt), str(embedding_txt))
    pt.load()
    assert os.path.exists(embedding_pt)
    return embedding_pt

class TestClassifier:
    def build_model(self, tmp_path, fake_embeddings, train_file, dev_file, extra_args=None, checkpoint_file=None):
        """
        Build a model to be used by one of the later tests
        """
        save_dir = str(tmp_path / "classifier")
        save_name = "model.pt"
        args = ["--save_dir", save_dir,
                "--save_name", save_name,
                "--wordvec_pretrain_file", str(fake_embeddings),
                "--filter_channels", "20",
                "--fc_shapes", "20,10",
                "--train_file", str(train_file),
                "--dev_file", str(dev_file),
                "--max_epochs", "2",
                "--batch_size", "60"]
        if extra_args is not None:
            args = args + extra_args
        args = classifier.parse_args(args)
        train_set = data.read_dataset(args.train_file, args.wordvec_type, args.min_train_len)
        if checkpoint_file:
            trainer = Trainer.load(checkpoint_file, args, load_optimizer=True)
        else:
            trainer = Trainer.build_new_model(args, train_set)
        return trainer, train_set, args

    def run_training(self, tmp_path, fake_embeddings, train_file, dev_file, extra_args=None, checkpoint_file=None):
        """
        Iterate a couple times over a model
        """
        trainer, train_set, args = self.build_model(tmp_path, fake_embeddings, train_file, dev_file, extra_args, checkpoint_file)
        dev_set = data.read_dataset(args.dev_file, args.wordvec_type, args.min_train_len)
        labels = data.dataset_labels(train_set)

        save_filename = os.path.join(args.save_dir, args.save_name)
        if checkpoint_file is None:
            checkpoint_file = utils.checkpoint_name(args.save_dir, save_filename, args.checkpoint_save_name)
        classifier.train_model(trainer, save_filename, checkpoint_file, args, train_set, dev_set, labels)
        return trainer, save_filename, checkpoint_file

    def test_build_model(self, tmp_path, fake_embeddings, train_file, dev_file):
        """
        Test that building a basic model works
        """
        self.build_model(tmp_path, fake_embeddings, train_file, dev_file, extra_args=["--bilstm_hidden_dim", "20"])

    def test_save_load(self, tmp_path, fake_embeddings, train_file, dev_file):
        """
        Test that a basic model can save & load
        """
        trainer, _, args = self.build_model(tmp_path, fake_embeddings, train_file, dev_file, extra_args=["--bilstm_hidden_dim", "20"])

        save_filename = os.path.join(args.save_dir, args.save_name)
        trainer.save(save_filename)

        args.load_name = args.save_name
        trainer = Trainer.load(args.load_name, args)
        args.load_name = save_filename
        trainer = Trainer.load(args.load_name, args)

    def test_train_basic(self, tmp_path, fake_embeddings, train_file, dev_file):
        self.run_training(tmp_path, fake_embeddings, train_file, dev_file, extra_args=["--bilstm_hidden_dim", "20"])

    def test_train_bilstm(self, tmp_path, fake_embeddings, train_file, dev_file):
        """
        Test w/ and w/o bilstm variations of the classifier
        """
        args = ["--bilstm", "--bilstm_hidden_dim", "20"]
        self.run_training(tmp_path, fake_embeddings, train_file, dev_file, args)

        args = ["--no_bilstm"]
        self.run_training(tmp_path, fake_embeddings, train_file, dev_file, args)

    def test_train_maxpool_width(self, tmp_path, fake_embeddings, train_file, dev_file):
        """
        Test various maxpool widths

        Also sets --filter_channels to a multiple of 2 but not of 3 for
        the test to make sure the math is done correctly on a non-divisible width
        """
        args = ["--maxpool_width", "1", "--filter_channels", "20", "--bilstm_hidden_dim", "20"]
        self.run_training(tmp_path, fake_embeddings, train_file, dev_file, args)

        args = ["--maxpool_width", "2", "--filter_channels", "20", "--bilstm_hidden_dim", "20"]
        self.run_training(tmp_path, fake_embeddings, train_file, dev_file, args)

        args = ["--maxpool_width", "3", "--filter_channels", "20", "--bilstm_hidden_dim", "20"]
        self.run_training(tmp_path, fake_embeddings, train_file, dev_file, args)

    def test_train_conv_2d(self, tmp_path, fake_embeddings, train_file, dev_file):
        args = ["--filter_sizes", "(3,4,5)", "--filter_channels", "20", "--bilstm_hidden_dim", "20"]
        self.run_training(tmp_path, fake_embeddings, train_file, dev_file, args)

        args = ["--filter_sizes", "((3,2),)", "--filter_channels", "20", "--bilstm_hidden_dim", "20"]
        self.run_training(tmp_path, fake_embeddings, train_file, dev_file, args)

        args = ["--filter_sizes", "((3,2),3)", "--filter_channels", "20", "--bilstm_hidden_dim", "20"]
        self.run_training(tmp_path, fake_embeddings, train_file, dev_file, args)

    def test_train_filter_channels(self, tmp_path, fake_embeddings, train_file, dev_file):
        args = ["--filter_sizes", "((3,2),3)", "--filter_channels", "20", "--no_bilstm"]
        trainer, _, _ = self.run_training(tmp_path, fake_embeddings, train_file, dev_file, args)
        assert trainer.model.fc_input_size == 40

        args = ["--filter_sizes", "((3,2),3)", "--filter_channels", "15,20", "--no_bilstm"]
        trainer, _, _ = self.run_training(tmp_path, fake_embeddings, train_file, dev_file, args)
        # 50 = 2x15 for the 2d conv (over 5 dim embeddings) + 20
        assert trainer.model.fc_input_size == 50

    def test_train_bert(self, tmp_path, fake_embeddings, train_file, dev_file):
        """
        Test on a tiny Bert WITHOUT finetuning, which hopefully does not take up too much disk space or memory
        """
        bert_model = "hf-internal-testing/tiny-bert"

        trainer, save_filename, _ = self.run_training(tmp_path, fake_embeddings, train_file, dev_file, extra_args=["--bilstm_hidden_dim", "20", "--bert_model", bert_model])
        assert os.path.exists(save_filename)
        saved_model = torch.load(save_filename, lambda storage, loc: storage, weights_only=True)
        # check that the bert model wasn't saved as part of the classifier
        assert not saved_model['params']['config']['force_bert_saved']
        assert not any(x.startswith("bert_model") for x in saved_model['params']['model'].keys())

    def test_finetune_bert(self, tmp_path, fake_embeddings, train_file, dev_file):
        """
        Test on a tiny Bert WITH finetuning, which hopefully does not take up too much disk space or memory
        """
        bert_model = "hf-internal-testing/tiny-bert"

        trainer, save_filename, _ = self.run_training(tmp_path, fake_embeddings, train_file, dev_file, extra_args=["--bilstm_hidden_dim", "20", "--bert_model", bert_model, "--bert_finetune"])
        assert os.path.exists(save_filename)
        saved_model = torch.load(save_filename, lambda storage, loc: storage, weights_only=True)
        # after finetuning the bert model, make sure that the save file DOES contain parts of the transformer
        assert saved_model['params']['config']['force_bert_saved']
        assert any(x.startswith("bert_model") for x in saved_model['params']['model'].keys())

    def test_finetune_bert_layers(self, tmp_path, fake_embeddings, train_file, dev_file):
        """Test on a tiny Bert WITH finetuning, which hopefully does not take up too much disk space or memory, using 2 layers

        As an added bonus (or eager test), load the finished model and continue
        training from there.  Then check that the initial model and
        the middle model are different, then that the middle model and
        final model are different

        """
        bert_model = "hf-internal-testing/tiny-bert"

        trainer, save_filename, checkpoint_file = self.run_training(tmp_path, fake_embeddings, train_file, dev_file, extra_args=["--bilstm_hidden_dim", "20", "--bert_model", bert_model, "--bert_finetune", "--bert_hidden_layers", "2", "--save_intermediate_models"])
        assert os.path.exists(save_filename)

        save_path = os.path.split(save_filename)[0]

        initial_model = glob.glob(os.path.join(save_path, "*E0000*"))
        assert len(initial_model) == 1
        initial_model = initial_model[0]
        initial_model = torch.load(initial_model, lambda storage, loc: storage, weights_only=True)

        second_model_file = glob.glob(os.path.join(save_path, "*E0002*"))
        assert len(second_model_file) == 1
        second_model_file = second_model_file[0]
        second_model = torch.load(second_model_file, lambda storage, loc: storage, weights_only=True)

        for layer_idx in range(2):
            bert_names = [x for x in second_model['params']['model'].keys() if x.startswith("bert_model") and "layer.%d." % layer_idx in x]
            assert len(bert_names) > 0
            assert all(x in initial_model['params']['model'] and x in second_model['params']['model'] for x in bert_names)
            assert not all(torch.allclose(initial_model['params']['model'].get(x), second_model['params']['model'].get(x)) for x in bert_names)

        # put some random marker in the file to look for later,
        # check the continued training didn't clobber the expected file
        assert "asdf" not in second_model
        second_model["asdf"] = 1234
        torch.save(second_model, second_model_file)

        trainer, save_filename, checkpoint_file = self.run_training(tmp_path, fake_embeddings, train_file, dev_file, extra_args=["--bilstm_hidden_dim", "20", "--bert_model", bert_model, "--bert_finetune", "--bert_hidden_layers", "2", "--save_intermediate_models", "--max_epochs", "5"], checkpoint_file=checkpoint_file)

        second_model_file_redo = glob.glob(os.path.join(save_path, "*E0002*"))
        assert len(second_model_file_redo) == 1
        assert second_model_file == second_model_file_redo[0]
        second_model = torch.load(second_model_file, lambda storage, loc: storage, weights_only=True)
        assert "asdf" in second_model

        fifth_model_file = glob.glob(os.path.join(save_path, "*E0005*"))
        assert len(fifth_model_file) == 1

        final_model = torch.load(fifth_model_file[0], lambda storage, loc: storage, weights_only=True)
        for layer_idx in range(2):
            bert_names = [x for x in final_model['params']['model'].keys() if x.startswith("bert_model") and "layer.%d." % layer_idx in x]
            assert len(bert_names) > 0
            assert all(x in final_model['params']['model'] and x in second_model['params']['model'] for x in bert_names)
            assert not all(torch.allclose(final_model['params']['model'].get(x), second_model['params']['model'].get(x)) for x in bert_names)

    def test_finetune_peft(self, tmp_path, fake_embeddings, train_file, dev_file):
        """
        Test on a tiny Bert with PEFT finetuning
        """
        bert_model = "hf-internal-testing/tiny-bert"

        trainer, save_filename, _ = self.run_training(tmp_path, fake_embeddings, train_file, dev_file, extra_args=["--bilstm_hidden_dim", "20", "--bert_model", bert_model, "--bert_finetune", "--use_peft", "--lora_modules_to_save", "pooler"])
        assert os.path.exists(save_filename)
        saved_model = torch.load(save_filename, lambda storage, loc: storage, weights_only=True)
        # after finetuning the bert model, make sure that the save file DOES contain parts of the transformer, but only in peft form
        assert saved_model['params']['config']['bert_model'] == bert_model
        assert saved_model['params']['config']['force_bert_saved']
        assert saved_model['params']['config']['use_peft']

        assert not saved_model['params']['config']['has_charlm_forward']
        assert not saved_model['params']['config']['has_charlm_backward']

        assert len(saved_model['params']['bert_lora']) > 0
        assert any(x.find(".pooler.") >= 0 for x in saved_model['params']['bert_lora'])
        assert any(x.find(".encoder.") >= 0 for x in saved_model['params']['bert_lora'])
        assert not any(x.startswith("bert_model") for x in saved_model['params']['model'].keys())

        # The Pipeline should load and run a PEFT trained model,
        # although obviously we don't expect the results to do
        # anything correct
        pipeline = stanza.Pipeline("en", download_method=None, model_dir=TEST_MODELS_DIR, processors="tokenize,sentiment", sentiment_model_path=save_filename, sentiment_pretrain_path=str(fake_embeddings))
        doc = pipeline("This is a test")

    def test_finetune_peft_restart(self, tmp_path, fake_embeddings, train_file, dev_file):
        """
        Test that if we restart training on a peft model, the peft weights change
        """
        bert_model = "hf-internal-testing/tiny-bert"

        trainer, save_file, checkpoint_file = self.run_training(tmp_path, fake_embeddings, train_file, dev_file, extra_args=["--bilstm_hidden_dim", "20", "--bert_model", bert_model, "--bert_finetune", "--use_peft", "--lora_modules_to_save", "pooler", "--save_intermediate_models"])

        assert os.path.exists(save_file)
        saved_model = torch.load(save_file, lambda storage, loc: storage, weights_only=True)
        assert any(x.find(".encoder.") >= 0 for x in saved_model['params']['bert_lora'])


        trainer, save_file, checkpoint_file = self.run_training(tmp_path, fake_embeddings, train_file, dev_file, extra_args=["--bilstm_hidden_dim", "20", "--bert_model", bert_model, "--bert_finetune", "--use_peft", "--lora_modules_to_save", "pooler", "--save_intermediate_models", "--max_epochs", "5"], checkpoint_file=checkpoint_file)

        save_path = os.path.split(save_file)[0]

        initial_model_file = glob.glob(os.path.join(save_path, "*E0000*"))
        assert len(initial_model_file) == 1
        initial_model_file = initial_model_file[0]
        initial_model = torch.load(initial_model_file, lambda storage, loc: storage, weights_only=True)

        second_model_file = glob.glob(os.path.join(save_path, "*E0002*"))
        assert len(second_model_file) == 1
        second_model_file = second_model_file[0]
        second_model = torch.load(second_model_file, lambda storage, loc: storage, weights_only=True)

        final_model_file = glob.glob(os.path.join(save_path, "*E0005*"))
        assert len(final_model_file) == 1
        final_model_file = final_model_file[0]
        final_model = torch.load(final_model_file, lambda storage, loc: storage, weights_only=True)

        # params in initial_model & second_model start with "base_model.model."
        # whereas params in final_model start directly with "encoder" or "pooler"
        initial_lora = initial_model['params']['bert_lora']
        second_lora = second_model['params']['bert_lora']
        final_lora = final_model['params']['bert_lora']
        for side in ("_A.", "_B."):
            for layer in (".0.", ".1."):
                initial_params = sorted([x for x in initial_lora if x.find(".encoder.") > 0 and x.find(side) > 0 and x.find(layer) > 0])
                second_params = sorted([x for x in second_lora if x.find(".encoder.") > 0 and x.find(side) > 0 and x.find(layer) > 0])
                final_params = sorted([x for x in final_lora if x.startswith("encoder.") > 0 and x.find(side) > 0 and x.find(layer) > 0])
                assert len(initial_params) > 0
                assert len(initial_params) == len(second_params)
                assert len(initial_params) == len(final_params)
                for x, y in zip(second_params, final_params):
                    assert x.endswith(y)
                    if side != "_A.":  # the A tensors don't move very much, if at all
                        assert not torch.allclose(initial_lora.get(x), second_lora.get(x))
                        assert not torch.allclose(second_lora.get(x), final_lora.get(y))

