import pytest

import stanza.models.classifiers.data as data
from stanza.models.classifiers.utils import WVType
from stanza.models.common.vocab import PAD, UNK
from stanza.tests.classifiers.test_classifier import train_file, dev_file, test_file, SENTENCES

class TestClassifierData:
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

