"""
A few tests of specific operations from the Dataset
"""

import os
import pytest

import torch

from stanza.models.common.doc import *
from stanza.models import tagger
from stanza.models.pos.data import Dataset, merge_datasets
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

def test_merge_some_xpos():
    # batch size 2 for 3 total elements so that it sometimes randomly
    # puts the sentence with no xpos in a block by itself and
    # sometimes randomly puts it in a block with 1 other element
    args = tagger.parse_args(args=['--batch_size', '2'])

    train_docs = [CoNLL.conll2doc(input_str=TRAIN_DATA),
                  CoNLL.conll2doc(input_str=TRAIN_DATA_NO_XPOS)]

    # TODO: maybe refactor the reading code in the main body of the tagger for easier testing
    vocab = Dataset.init_vocab(train_docs, args)
    train_data = [Dataset(i, args, None, vocab=vocab, evaluation=False) for i in train_docs]
    lens = list(len(x) for x in train_data)
    assert lens == [2, 1]
    merged = merge_datasets(train_data)
    assert len(merged) == 3
    train_batches = merged.to_loader(batch_size=args["batch_size"], shuffle=True)
    it_first = 0
    for _ in range(200):
        for batch_idx, batch in enumerate(iter(train_batches)):
            if batch.text[-1][0] == 'It':
                if batch_idx == 0:
                    it_first += 1
                # TODO: I would expect this to always be False, unless
                # I have misinterpreted the process for the masking,
                # but there are times it comes back True instead.  I
                # think that is an effect of the has_xpos not being
                # sorted with everything else by length
                print(torch.any(batch.xpos[-1]))

    # check that the sentence w/o xpos is sometimes but not always in the first batch
    assert it_first > 5
    assert it_first < 195

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


