"""
A few tests of the utils module for the sentiment datasets
"""

import os
import pytest

import stanza

import stanza.models.classifiers.data as data
from stanza.models.classifiers.utils import WVType
from stanza.utils.datasets.sentiment import process_utils
from stanza.utils.datasets.sentiment.process_utils import SentimentDatum

from stanza.tests import TEST_MODELS_DIR
from stanza.tests.classifiers.test_classifier import train_file, dev_file, test_file


def test_write_list(tmp_path, train_file):
    """
    Test that writing a single list of items to an output file works

    TODO: when read_dataset reads SentimentDatum, no need to create these objects
    """
    train_set = data.read_dataset(train_file, WVType.OTHER, 1)

    dataset = [SentimentDatum(*x) for x in train_set]
    dataset_file = tmp_path / "foo.json"
    process_utils.write_list(dataset_file, dataset)

    train_copy = data.read_dataset(dataset_file, WVType.OTHER, 1)
    assert train_copy == train_set

def test_write_dataset(tmp_path, train_file, dev_file, test_file):
    """
    Test that writing all three parts of a dataset works

    TODO: when read_dataset reads SentimentDatum, no need to create these objects
    """
    dataset = [[SentimentDatum(*x) for x in data.read_dataset(filename, WVType.OTHER, 1)] for filename in (train_file, dev_file, test_file)]
    process_utils.write_dataset(dataset, tmp_path, "en_test")

    expected_files = ['en_test.train.json', 'en_test.dev.json', 'en_test.test.json']
    dataset_files = os.listdir(tmp_path)
    assert sorted(dataset_files) == sorted(expected_files)

    for filename, expected in zip(expected_files, dataset):
        written = data.read_dataset(tmp_path / filename, WVType.OTHER, 1)
        assert written == expected

def test_read_snippets(tmp_path):
    """
    Test the basic operation of the read_snippets function
    """
    filename = tmp_path / "foo.csv"
    with open(filename, "w", encoding="utf-8") as fout:
        fout.write("FOO\tThis is a test\thappy\n")
        fout.write("FOO\tThis is a second sentence\tsad\n")

    nlp = stanza.Pipeline("en", dir=TEST_MODELS_DIR, processors="tokenize", download_method=None)

    mapping = {"happy": 0, "sad": 1}

    snippets = process_utils.read_snippets(filename, 2, 1, "en", mapping, nlp=nlp)
    assert len(snippets) == 2
    assert snippets == [SentimentDatum(sentiment=0, text=['This', 'is', 'a', 'test']),
                        SentimentDatum(sentiment=1, text=['This', 'is', 'a', 'second', 'sentence'])]

