import os

import pytest
import numpy as np
import torch

from stanza.models.common import pretrain
from stanza.models.common.vocab import UNK_ID
from stanza.tests import *

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def check_vocab(vocab):
    # 4 base vectors, plus the 3 vectors actually present in the file
    assert len(vocab) == 7
    assert 'unban' in vocab
    assert 'mox' in vocab
    assert 'opal' in vocab

def check_embedding(emb, unk=False):
    expected = np.array([[ 0.,  0.,  0.,  0.,],
                         [ 0.,  0.,  0.,  0.,],
                         [ 0.,  0.,  0.,  0.,],
                         [ 0.,  0.,  0.,  0.,],
                         [ 1.,  2.,  3.,  4.,],
                         [ 5.,  6.,  7.,  8.,],
                         [ 9., 10., 11., 12.,]])
    if unk:
        expected[UNK_ID] = -1
    np.testing.assert_allclose(emb, expected)

def check_pretrain(pt):
    check_vocab(pt.vocab)
    check_embedding(pt.emb)

def build_pretrain(filename=None):
    vocab = pretrain.PretrainedWordVocab(["unban", "mox", "opal"])
    emb = torch.tensor([[0, 0, 0, 0]] * 4 + [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.float32)
    return pretrain.Pretrain(filename, vocab=vocab, emb=emb)

def test_load_pretrain():
    pt = pretrain.Pretrain(f'{TEST_WORKING_DIR}/in/tiny_emb.pt')
    check_pretrain(pt)

def test_save_and_load(tmp_path):
    """
    A Pretrain built from a vocab and emb can be saved and loaded back
    """
    filename = str(tmp_path / "subdir" / "tiny.pt")
    pt = build_pretrain()
    check_pretrain(pt)
    pt.save(filename)

    pt2 = pretrain.Pretrain(filename)
    check_pretrain(pt2)

    data = torch.load(filename, weights_only=True)
    check_embedding(data['emb'])

def test_missing_file(tmp_path):
    """
    A missing pretrain file is an error which says how to make one
    """
    pt = pretrain.Pretrain(str(tmp_path / "missing.pt"))
    with pytest.raises(FileNotFoundError, match="convert_pretrain"):
        pt.vocab

def test_not_a_pretrain(tmp_path):
    filename = str(tmp_path / "not_a_pretrain.pt")
    torch.save({'foo': 1}, filename)
    with pytest.raises(RuntimeError, match="not a stanza pretrain"):
        pretrain.Pretrain(filename).emb

def test_constructor_errors():
    vocab = pretrain.PretrainedWordVocab(["unban", "mox", "opal"])
    emb = torch.zeros((7, 4))
    with pytest.raises(ValueError):
        pretrain.Pretrain()
    with pytest.raises(ValueError):
        pretrain.Pretrain(vocab=vocab)
    with pytest.raises(ValueError):
        pretrain.Pretrain(emb=emb)
    with pytest.raises(ValueError):
        pretrain.Pretrain(vocab=vocab, emb=torch.zeros((6, 4)))

def test_whitespace_lookup():
    """
    Words stored with nbsp can be looked up with an ascii space
    """
    vocab = pretrain.PretrainedWordVocab(["unban\xa0mox", "opal", "foo"])
    pt = pretrain.Pretrain(vocab=vocab, emb=torch.zeros((7, 4)))
    assert "unban\xa0mox" in pt.vocab
    # this one also works because of the normalize_unit in PretrainedWordVocab
    assert "unban mox" in pt.vocab
