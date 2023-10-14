"""
A few tests of specific operations from the Dataset
"""

import os
import pytest

from stanza.models.common.doc import *
from stanza.models import tagger
from stanza.models.pos.data import Dataset
from stanza.utils.conll import CoNLL

from stanza.tests.pos.test_tagger import TRAIN_DATA, TRAIN_DATA_NO_XPOS, TRAIN_DATA_NO_UPOS, TRAIN_DATA_NO_FEATS

def test_basic_reading():
    """
    Test that a dataset with no xpos is detected by the Dataset
    """
    # empty args for building the data object
    args = tagger.parse_args(args=[])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA)

    data = Dataset(train_doc, args, None)
    assert data.has_upos
    assert data.has_xpos
    assert data.has_feats

def test_no_xpos():
    """
    Test that a dataset with no xpos is detected by the Dataset
    """
    # empty args for building the data object
    args = tagger.parse_args(args=[])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA_NO_XPOS)

    data = Dataset(train_doc, args, None)
    assert data.has_upos
    assert not data.has_xpos
    assert data.has_feats

def test_no_upos():
    """
    Test that a dataset with no upos is detected by the Dataset
    """
    # empty args for building the data object
    args = tagger.parse_args(args=[])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA_NO_UPOS)

    data = Dataset(train_doc, args, None)
    assert not data.has_upos
    assert data.has_xpos
    assert data.has_feats

def test_no_feats():
    """
    Test that a dataset with no feats is detected by the Dataset
    """
    # empty args for building the data object
    args = tagger.parse_args(args=[])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA_NO_FEATS)

    data = Dataset(train_doc, args, None)
    assert data.has_upos
    assert data.has_xpos
    assert not data.has_feats

def test_no_augment():
    """
    Test that with no punct removing augmentation, the doc always has punct at the end
    """
    args = tagger.parse_args(args=["--shorthand", "en_test", "--augment_nopunct", "0.0"])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA)
    data = Dataset(train_doc, args, None)
    data = data.to_loader(batch_size=2)

    for i in range(50):
        for batch in data:
            for text in batch.text:
                assert text[-1] in (".", "!")

def test_augment():
    """
    Test that with 100% punct removing augmentation, the doc never has punct at the end
    """
    args = tagger.parse_args(args=["--shorthand", "en_test", "--augment_nopunct", "1.0"])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA)
    data = Dataset(train_doc, args, None)
    data = data.to_loader(batch_size=2)

    for i in range(50):
        for batch in data:
            for text in batch.text:
                assert text[-1] not in (".", "!")

def test_sometimes_augment():
    """
    Test 50% punct removing augmentation

    With this frequency, we should get a reasonable number of docs
    with a punct at the end and a reasonable without.
    """
    args = tagger.parse_args(args=["--shorthand", "en_test", "--augment_nopunct", "0.5"])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA)
    data = Dataset(train_doc, args, None)
    data = data.to_loader(batch_size=2)

    count_with = 0
    count_without = 0
    for i in range(50):
        for batch in data:
            for text in batch.text:
                if text[-1] in (".", "!"):
                    count_with += 1
                else:
                    count_without += 1

    # this should never happen
    # literally less than 1 in 10^20th odds
    assert count_with > 5
    assert count_without > 5


