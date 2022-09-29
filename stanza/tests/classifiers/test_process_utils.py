"""
A few tests of the utils module for the sentiment datasets
"""

import os
import pytest

import stanza

from stanza.models.classifiers import data
from stanza.models.classifiers.data import SentimentDatum
from stanza.models.classifiers.utils import WVType
from stanza.utils.datasets.sentiment import process_utils

from stanza.tests import TEST_MODELS_DIR
from stanza.tests.classifiers.test_data import train_file, dev_file, test_file


def test_write_list(tmp_path, train_file):
    """
    Test that writing a single list of items to an output file works
    """
    train_set = data.read_dataset(train_file, WVType.OTHER, 1)

    dataset_file = tmp_path / "foo.json"
    process_utils.write_list(dataset_file, train_set)

    train_copy = data.read_dataset(dataset_file, WVType.OTHER, 1)
    assert train_copy == train_set

def test_write_dataset(tmp_path, train_file, dev_file, test_file):
    """
    Test that writing all three parts of a dataset works
    """
    dataset = [data.read_dataset(filename, WVType.OTHER, 1) for filename in (train_file, dev_file, test_file)]
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

def test_read_snippets_two_columns(tmp_path):
    """
    Test what happens when multiple columns are combined for the sentiment value
    """
    filename = tmp_path / "foo.csv"
    with open(filename, "w", encoding="utf-8") as fout:
        fout.write("FOO\tThis is a test\thappy\tfoo\n")
        fout.write("FOO\tThis is a second sentence\tsad\tbar\n")
        fout.write("FOO\tThis is a third sentence\tsad\tfoo\n")

    nlp = stanza.Pipeline("en", dir=TEST_MODELS_DIR, processors="tokenize", download_method=None)

    mapping = {("happy", "foo"): 0, ("sad", "bar"): 1, ("sad", "foo"): 2}

    snippets = process_utils.read_snippets(filename, (2,3), 1, "en", mapping, nlp=nlp)
    assert len(snippets) == 3
    assert snippets == [SentimentDatum(sentiment=0, text=['This', 'is', 'a', 'test']),
                        SentimentDatum(sentiment=1, text=['This', 'is', 'a', 'second', 'sentence']),
                        SentimentDatum(sentiment=2, text=['This', 'is', 'a', 'third', 'sentence'])]

