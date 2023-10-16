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
