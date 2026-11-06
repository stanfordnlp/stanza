"""
Tests for converting word vector files to Stanza pretrains

Run with:
    pytest stanza/tests/utils/test_convert_pretrain.py
"""

import os

import pytest

from stanza.models.common import pretrain
from stanza.tests import TEST_WORKING_DIR
from stanza.tests.common.test_pretrain import check_embedding, check_pretrain
from stanza.utils.pretrain.convert_pretrain import convert_pretrain, main

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

@pytest.mark.parametrize("suffix", ["txt", "xz", "gz", "zip", "csv"])
def test_convert_tiny_emb(suffix):
    pt = convert_pretrain(f'{TEST_WORKING_DIR}/in/tiny_emb.{suffix}')
    check_pretrain(pt)

def write_vectors(tmp_path, text):
    filename = tmp_path / "tiny.txt"
    filename.write_text(text, encoding="utf-8")
    return str(filename)

SPACE_PRETRAIN = """
3 4
unban mox 1 2 3 4
opal 5 6 7 8
foo 9 10 11 12
""".strip()

def test_whitespace(tmp_path):
    """
    A word with an ascii space in it gets the correct number of dimensions, with the space converted to nbsp
    """
    pt = convert_pretrain(write_vectors(tmp_path, SPACE_PRETRAIN))
    check_embedding(pt.emb)
    assert "unban\xa0mox" in pt.vocab
    assert "unban mox" in pt.vocab

NO_HEADER_PRETRAIN = """
unban 1 2 3 4
mox 5 6 7 8
opal 9 10 11 12
""".strip()

def test_no_header(tmp_path):
    """
    Check converting a vector file with no rows,cols header
    """
    pt = convert_pretrain(write_vectors(tmp_path, NO_HEADER_PRETRAIN))
    check_pretrain(pt)

UNK_PRETRAIN = """
unban 1 2 3 4
mox 5 6 7 8
opal 9 10 11 12
<unk> -1 -1 -1 -1
""".strip()

def test_unk_pretrain(tmp_path):
    """
    Check converting a vector file with <unk> at the end, like GloVe has
    """
    pt = convert_pretrain(write_vectors(tmp_path, UNK_PRETRAIN))
    check_embedding(pt.emb, unk=True)

@pytest.mark.parametrize("suffix", ["txt", "csv"])
def test_max_vocab(suffix):
    """
    max_vocab counts the VOCAB_PREFIX entries, for text and csv alike
    """
    pt = convert_pretrain(f'{TEST_WORKING_DIR}/in/tiny_emb.{suffix}', max_vocab=6)
    assert len(pt.vocab) == 6
    assert 'unban' in pt.vocab
    assert 'mox' in pt.vocab
    assert 'opal' not in pt.vocab
    assert pt.emb.shape == (6, 4)

def test_main(tmp_path):
    """
    The script writes a pretrain, and leaves an existing one alone
    """
    output_pt = str(tmp_path / "tiny.pt")
    main([output_pt, f'{TEST_WORKING_DIR}/in/tiny_emb.xz'])
    check_pretrain(pretrain.Pretrain(output_pt))

    mtime = os.path.getmtime(output_pt)
    main([output_pt, write_vectors(tmp_path, UNK_PRETRAIN)])
    assert os.path.getmtime(output_pt) == mtime
    check_pretrain(pretrain.Pretrain(output_pt))
