import os

import pytest

import stanza
import stanza.models.classifier as classifier
import stanza.models.classifiers.data as data
from stanza.models.classifiers.trainer import Trainer
from stanza.tests import TEST_MODELS_DIR
from stanza.tests.classifiers.test_classifier import fake_embeddings
from stanza.tests.classifiers.test_data import train_file_with_trees, dev_file_with_trees
from stanza.models.common import utils
from stanza.tests.constituency.test_trainer import build_trainer, TREEBANK

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

class TestConstituencyClassifier:
    @pytest.fixture(scope="class")
    def constituency_model(self, fake_embeddings, tmp_path_factory):
        args = ['--pattn_num_layers', '0', '--lattn_d_proj', '0', '--hidden_size', '20', '--delta_embedding_dim', '10']
        trainer = build_trainer(str(fake_embeddings), *args, treebank=TREEBANK)

        trainer_pt = str(tmp_path_factory.mktemp("constituency") / "constituency.pt")
        trainer.save(trainer_pt, save_optimizer=False)
        return trainer_pt

    def build_model(self, tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees, extra_args=None):
        """
        Build a Constituency Classifier model to be used by one of the later tests
        """
        save_dir = str(tmp_path / "classifier")
        save_name = "model.pt"
        args = ["--save_dir", save_dir,
                "--save_name", save_name,
                "--model_type", "constituency",
                "--constituency_model", constituency_model,
                "--wordvec_pretrain_file", str(fake_embeddings),
                "--fc_shapes", "20,10",
                "--train_file", str(train_file_with_trees),
                "--dev_file", str(dev_file_with_trees),
                "--max_epochs", "2",
                "--batch_size", "60"]
        if extra_args is not None:
            args = args + extra_args
        args = classifier.parse_args(args)
        train_set = data.read_dataset(args.train_file, args.wordvec_type, args.min_train_len)
        trainer = Trainer.build_new_model(args, train_set)
        return trainer, train_set, args

    def run_training(self, tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees, extra_args=None):
        """
        Iterate a couple times over a model
        """
        trainer, train_set, args = self.build_model(tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees, extra_args)
        dev_set = data.read_dataset(args.dev_file, args.wordvec_type, args.min_train_len)
        labels = data.dataset_labels(train_set)

        save_filename = os.path.join(args.save_dir, args.save_name)
        checkpoint_file = utils.checkpoint_name(args.save_dir, save_filename, args.checkpoint_save_name)
        classifier.train_model(trainer, save_filename, checkpoint_file, args, train_set, dev_set, labels)
        return trainer, train_set, args

    def test_build_model(self, tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees):
        """
        Test that building a basic constituency-based model works
        """
        self.build_model(tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees)

    def test_save_load(self, tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees):
        """
        Test that a constituency model can save & load
        """
        trainer, _, args = self.build_model(tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees)

        save_filename = os.path.join(args.save_dir, args.save_name)
        trainer.save(save_filename)

        args.load_name = args.save_name
        trainer = Trainer.load(args.load_name, args)

    def test_train_basic(self, tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees):
        self.run_training(tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees)

    def test_train_pipeline(self, tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees):
        """
        Test that writing out a temp model, then loading it in the pipeline is a thing that works
        """
        trainer, _, args = self.run_training(tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees)
        save_filename = os.path.join(args.save_dir, args.save_name)
        assert os.path.exists(save_filename)
        assert os.path.exists(args.constituency_model)

        pipeline_args = {"lang": "en",
                         "download_method": None,
                         "model_dir": TEST_MODELS_DIR,
                         "processors": "tokenize,pos,constituency,sentiment",
                         "tokenize_pretokenized": True,
                         "constituency_model_path": args.constituency_model,
                         "constituency_pretrain_path": args.wordvec_pretrain_file,
                         "constituency_backward_charlm_path": None,
                         "constituency_forward_charlm_path": None,
                         "sentiment_model_path": save_filename,
                         "sentiment_pretrain_path": args.wordvec_pretrain_file,
                         "sentiment_backward_charlm_path": None,
                         "sentiment_forward_charlm_path": None}
        pipeline = stanza.Pipeline(**pipeline_args)
        doc = pipeline("This is a test")
        # since the model is random, we have no expectations for what the result actually is
        assert doc.sentences[0].sentiment is not None


    def test_train_all_words(self, tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees):
        self.run_training(tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees, ['--constituency_all_words'])

        self.run_training(tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees, ['--no_constituency_all_words'])

    def test_train_top_layer(self, tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees):
        self.run_training(tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees, ['--constituency_top_layer'])

        self.run_training(tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees, ['--no_constituency_top_layer'])

    def test_train_attn(self, tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees):
        self.run_training(tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees, ['--constituency_node_attn', '--no_constituency_all_words'])

        self.run_training(tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees, ['--constituency_node_attn', '--constituency_all_words'])

        self.run_training(tmp_path, constituency_model, fake_embeddings, train_file_with_trees, dev_file_with_trees, ['--no_constituency_node_attn'])

