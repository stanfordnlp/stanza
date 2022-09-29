import pytest

import stanza.models.classifiers.data as data
from stanza.models.classifiers.utils import WVType
from stanza.models.common.vocab import PAD, UNK
from stanza.tests.classifiers.test_classifier import train_file, dev_file, test_file, DATASET, SENTENCES

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

    def test_sort_by_length(self, train_file):
        """
        There are two unique lengths in the toy dataset
        """
        train_set = data.read_dataset(str(train_file), WVType.OTHER, 1)
        sorted_dataset = data.sort_dataset_by_len(train_set)
        assert list(sorted_dataset.keys()) == [4, 5]
        assert len(sorted_dataset[4]) == len(train_set) // 3
        assert len(sorted_dataset[5]) == 2 * len(train_set) // 3

    def test_check_labels(self, train_file):
        """
        Check that an exception is thrown for an unknown label
        """
        train_set = data.read_dataset(str(train_file), WVType.OTHER, 1)
        labels = sorted(set([x["sentiment"] for x in DATASET]))
        assert len(labels) > 1
        data.check_labels(labels, train_set)
        with pytest.raises(RuntimeError):
            data.check_labels(labels[:1], train_set)

