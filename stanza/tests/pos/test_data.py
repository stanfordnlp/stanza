"""
A few tests of specific operations from the DataLoader
"""

import os
import pytest

from stanza.models.common.doc import *
from stanza.models import tagger
from stanza.models.pos.data import DataLoader
from stanza.utils.conll import CoNLL

from stanza.tests.pos.test_tagger import TRAIN_DATA_NO_XPOS, TRAIN_DATA_NO_UPOS, TRAIN_DATA_NO_FEATS

def test_no_xpos():
    """
    Test that a dataset with no xpos is detected by the DataLoader
    """
    # empty args for building the data object
    args = tagger.parse_args(args=[])

    train_data, _ = CoNLL.conll2dict(input_str=TRAIN_DATA_NO_XPOS)
    train_doc = Document(train_data)

    data = DataLoader(train_doc, args['batch_size'], args, None)
    assert data.has_upos
    assert not data.has_xpos
    assert data.has_feats

def test_no_upos():
    """
    Test that a dataset with no upos is detected by the DataLoader
    """
    # empty args for building the data object
    args = tagger.parse_args(args=[])

    train_data, _ = CoNLL.conll2dict(input_str=TRAIN_DATA_NO_UPOS)
    train_doc = Document(train_data)

    data = DataLoader(train_doc, args['batch_size'], args, None)
    assert not data.has_upos
    assert data.has_xpos
    assert data.has_feats

def test_no_feats():
    """
    Test that a dataset with no feats is detected by the DataLoader
    """
    # empty args for building the data object
    args = tagger.parse_args(args=[])

    train_data, _ = CoNLL.conll2dict(input_str=TRAIN_DATA_NO_FEATS)
    train_doc = Document(train_data)

    data = DataLoader(train_doc, args['batch_size'], args, None)
    assert data.has_upos
    assert data.has_xpos
    assert not data.has_feats
