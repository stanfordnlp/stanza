import json
import os

import pytest

import numpy as np

import stanza.models.classifier as classifier
import stanza.models.classifiers.cnn_classifier as cnn_classifier
import stanza.models.classifiers.data as data
from stanza.models.classifiers.trainer import Trainer
from stanza.models.classifiers.classifier_args import WVType
from stanza.models.common import pretrain
from stanza.models.common import utils
from stanza.models.common.vocab import PAD, UNK

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

SENTENCES = [
    ["I", "hate", "the", "Opal", "banning"],
    ["Tell", "my", "wife", "hello"],
    ["I", "like", "Sh'reyan", "'s", "antennae"],
]

DATASET = [
    {"sentiment": "0", "text": SENTENCES[0]},
    {"sentiment": "1", "text": SENTENCES[1]},
    {"sentiment": "2", "text": SENTENCES[2]},
]

EMB_DIM = 5

class TestClassifier:
    @pytest.fixture(scope="class")
    def train_file(self, tmp_path_factory):
        train_set = DATASET * 20
        train_filename = tmp_path_factory.mktemp("data") / "train.json"
        with open(train_filename, "w", encoding="utf-8") as fout:
            json.dump(train_set, fout, ensure_ascii=False)
        return train_filename

    @pytest.fixture(scope="class")
    def dev_file(self, tmp_path_factory):
        dev_set = DATASET * 2
        dev_filename = tmp_path_factory.mktemp("data") / "dev.json"
        with open(dev_filename, "w", encoding="utf-8") as fout:
            json.dump(dev_set, fout, ensure_ascii=False)
        return dev_filename

    @pytest.fixture(scope="class")
    def test_file(self, tmp_path_factory):
        test_set = DATASET
        test_filename = tmp_path_factory.mktemp("data") / "test.json"
        with open(test_filename, "w", encoding="utf-8") as fout:
            json.dump(test_set, fout, ensure_ascii=False)
        return test_filename

    @pytest.fixture(scope="class")
    def fake_embeddings(self, tmp_path_factory):
        # could set np random seed here
        words = sorted(set([x.lower() for y in SENTENCES for x in y]))
        words = words[:-1]
        embedding_txt = tmp_path_factory.mktemp("data") / "embedding.txt"
        embedding_pt  = tmp_path_factory.mktemp("data") / "embedding.pt"
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

    def test_read_data(self, train_file):
        """
        Test reading of the json format
        """
        train_set = data.read_dataset(str(train_file), WVType.OTHER, 1)
        assert len(train_set) == 60

    def test_dataset_vocab(self, train_file):
        """
        Converting a dataset to vocab should have a specific set of words along with PAD and UNK
        """
        train_set = data.read_dataset(str(train_file), WVType.OTHER, 1)
        vocab = data.dataset_vocab(train_set)
        expected = set([PAD, UNK] + [x.lower() for y in SENTENCES for x in y])
        assert set(vocab) == expected

    def test_dataset_labels(self, train_file):
        """
        Test the extraction of labels from a dataset
        """
        train_set = data.read_dataset(str(train_file), WVType.OTHER, 1)
        labels = data.dataset_labels(train_set)
        assert labels == ["0", "1", "2"]

    def build_model(self, tmp_path, fake_embeddings, train_file, dev_file, extra_args=None):
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
        trainer = Trainer.build_new_model(args, train_set)
        return trainer, train_set, args

    def run_training(self, tmp_path, fake_embeddings, train_file, dev_file, extra_args=None):
        """
        Iterate a couple times over a model
        """
        trainer, train_set, args = self.build_model(tmp_path, fake_embeddings, train_file, dev_file, extra_args)
        dev_set = data.read_dataset(args.dev_file, args.wordvec_type, args.min_train_len)
        labels = data.dataset_labels(train_set)

        save_filename = os.path.join(args.save_dir, args.save_name)
        checkpoint_file = utils.checkpoint_name(args.save_dir, save_filename, args.checkpoint_save_name)
        classifier.train_model(trainer, save_filename, checkpoint_file, args, train_set, dev_set, labels)
        return trainer

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
        trainer = self.run_training(tmp_path, fake_embeddings, train_file, dev_file, args)
        assert trainer.model.fc_input_size == 40

        args = ["--filter_sizes", "((3,2),3)", "--filter_channels", "15,20", "--no_bilstm"]
        trainer = self.run_training(tmp_path, fake_embeddings, train_file, dev_file, args)
        # 50 = 2x15 for the 2d conv (over 5 dim embeddings) + 20
        assert trainer.model.fc_input_size == 50
