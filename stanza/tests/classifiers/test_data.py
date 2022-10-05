import json
import pytest

import stanza.models.classifiers.data as data
from stanza.models.classifiers.utils import WVType
from stanza.models.common.vocab import PAD, UNK
from stanza.models.constituency.parse_tree import Tree

SENTENCES = [
    ["I", "hate", "the", "Opal", "banning"],
    ["Tell", "my", "wife", "hello"], # obviously this is the neutral result
    ["I", "like", "Sh'reyan", "'s", "antennae"],
]

DATASET = [
    {"sentiment": "0", "text": SENTENCES[0]},
    {"sentiment": "1", "text": SENTENCES[1]},
    {"sentiment": "2", "text": SENTENCES[2]},
]

TREES = [
    "(ROOT (S (NP (PRP I)) (VP (VBP hate) (NP (DT the) (NN Opal) (NN banning)))))",
    "(ROOT (S (VP (VB Tell) (NP (PRP$ my) (NN wife)) (NP (UH hello)))))",
    "(ROOT (S (NP (PRP I)) (VP (VBP like) (NP (NP (NNP Sh'reyan) (POS 's)) (NNS antennae)))))",
]

DATASET_WITH_TREES = [
    {"sentiment": "0", "text": SENTENCES[0], "constituency": TREES[0]},
    {"sentiment": "1", "text": SENTENCES[1], "constituency": TREES[1]},
    {"sentiment": "2", "text": SENTENCES[2], "constituency": TREES[2]},
]

@pytest.fixture(scope="module")
def train_file(tmp_path_factory):
    train_set = DATASET * 20
    train_filename = tmp_path_factory.mktemp("data") / "train.json"
    with open(train_filename, "w", encoding="utf-8") as fout:
        json.dump(train_set, fout, ensure_ascii=False)
    return train_filename

@pytest.fixture(scope="module")
def dev_file(tmp_path_factory):
    dev_set = DATASET * 2
    dev_filename = tmp_path_factory.mktemp("data") / "dev.json"
    with open(dev_filename, "w", encoding="utf-8") as fout:
        json.dump(dev_set, fout, ensure_ascii=False)
    return dev_filename

@pytest.fixture(scope="module")
def test_file(tmp_path_factory):
    test_set = DATASET
    test_filename = tmp_path_factory.mktemp("data") / "test.json"
    with open(test_filename, "w", encoding="utf-8") as fout:
        json.dump(test_set, fout, ensure_ascii=False)
    return test_filename

@pytest.fixture(scope="module")
def train_file_with_trees(tmp_path_factory):
    train_set = DATASET_WITH_TREES * 20
    train_filename = tmp_path_factory.mktemp("data") / "train_trees.json"
    with open(train_filename, "w", encoding="utf-8") as fout:
        json.dump(train_set, fout, ensure_ascii=False)
    return train_filename

@pytest.fixture(scope="module")
def dev_file_with_trees(tmp_path_factory):
    dev_set = DATASET_WITH_TREES * 2
    dev_filename = tmp_path_factory.mktemp("data") / "dev_trees.json"
    with open(dev_filename, "w", encoding="utf-8") as fout:
        json.dump(dev_set, fout, ensure_ascii=False)
    return dev_filename

class TestClassifierData:
    def test_read_data(self, train_file):
        """
        Test reading of the json format
        """
        train_set = data.read_dataset(str(train_file), WVType.OTHER, 1)
        assert len(train_set) == 60

    def test_read_data_with_trees(self, train_file, train_file_with_trees):
        """
        Test reading of the json format
        """
        train_trees_set = data.read_dataset(str(train_file_with_trees), WVType.OTHER, 1)
        assert len(train_trees_set) == 60
        for idx, x in enumerate(train_trees_set):
            assert isinstance(x.constituency, Tree)
            assert str(x.constituency) == TREES[idx % len(TREES)]

        train_set = data.read_dataset(str(train_file), WVType.OTHER, 1)

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

