import os
import tempfile

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

def test_text_pretrain():
    pt = pretrain.Pretrain(vec_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.txt', save_to_file=False)
    check_pretrain(pt)

def test_xz_pretrain():
    pt = pretrain.Pretrain(vec_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.xz', save_to_file=False)
    check_pretrain(pt)

def test_gz_pretrain():
    pt = pretrain.Pretrain(vec_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.gz', save_to_file=False)
    check_pretrain(pt)

def test_zip_pretrain():
    pt = pretrain.Pretrain(vec_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.zip', save_to_file=False)
    check_pretrain(pt)

def test_csv_pretrain():
    pt = pretrain.Pretrain(csv_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.csv', save_to_file=False)
    check_pretrain(pt)

def test_resave_pretrain():
    """
    Test saving a pretrain and then loading from the existing file
    """
    test_pt_file = tempfile.NamedTemporaryFile(dir=f'{TEST_WORKING_DIR}/out', suffix=".pt", delete=False)
    try:
        test_pt_file.close()
        # note that this tests the ability to save a pretrain and the
        # ability to fall back when the existing pretrain isn't working
        pt = pretrain.Pretrain(filename=test_pt_file.name,
                               vec_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.xz')
        check_pretrain(pt)

        pt2 = pretrain.Pretrain(filename=test_pt_file.name,
                               vec_filename=f'unban_mox_opal')
        check_pretrain(pt2)

        pt3 = torch.load(test_pt_file.name, weights_only=True)
        check_embedding(pt3['emb'])
    finally:
        os.unlink(test_pt_file.name)

SPACE_PRETRAIN="""
3 4
unban mox 1 2 3 4
opal 5 6 7 8
foo 9 10 11 12
""".strip()

def test_whitespace():
    """
    Test reading a pretrain with an ascii space in it

    The vocab word with a space in it should have the correct number
    of dimensions read, with the space converted to nbsp
    """
    test_txt_file = tempfile.NamedTemporaryFile(dir=f'{TEST_WORKING_DIR}/out', suffix=".txt", delete=False)
    try:
        test_txt_file.write(SPACE_PRETRAIN.encode())
        test_txt_file.close()

        pt = pretrain.Pretrain(vec_filename=test_txt_file.name, save_to_file=False)
        check_embedding(pt.emb)
        assert "unban\xa0mox" in pt.vocab
        # this one also works because of the normalize_unit in vocab.py
        assert "unban mox" in pt.vocab
    finally:
        os.unlink(test_txt_file.name)

NO_HEADER_PRETRAIN="""
unban 1 2 3 4
mox 5 6 7 8
opal 9 10 11 12
""".strip()

def test_no_header():
    """
    Check loading a pretrain with no rows,cols header
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdir:
        filename = os.path.join(tmpdir, "tiny.txt")
        with open(filename, "w", encoding="utf-8") as fout:
            fout.write(NO_HEADER_PRETRAIN)
        pt = pretrain.Pretrain(vec_filename=filename, save_to_file=False)
        check_embedding(pt.emb)

UNK_PRETRAIN="""
unban 1 2 3 4
mox 5 6 7 8
opal 9 10 11 12
<unk> -1 -1 -1 -1
""".strip()

def test_unk_pretrain():
    """
    Check loading a pretrain with <unk> at the end, like GloVe does
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdir:
        filename = os.path.join(tmpdir, "tiny.txt")
        with open(filename, "w", encoding="utf-8") as fout:
            fout.write(UNK_PRETRAIN)
        pt = pretrain.Pretrain(vec_filename=filename, save_to_file=False)
        check_embedding(pt.emb, unk=True)

# OTA pretrain uses words already in normalized form, since that is what
# the vector file contains.  the normalization is applied at lookup time
# to map input text in either orthographic convention to the stored form.
OTA_PRETRAIN="""
3 4
pâdişâh 1 2 3 4
gazel 5 6 7 8
kitâb 9 10 11 12
""".strip()

def write_pretrain_txt(tmpdir, content):
    filename = os.path.join(tmpdir, "tiny.txt")
    with open(filename, "w", encoding="utf-8") as fout:
        fout.write(content)
    return filename

def test_normalization_defaults_to_none():
    """
    A pretrain loaded without specifying text_normalization should default to NONE
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdir:
        filename = write_pretrain_txt(tmpdir, NO_HEADER_PRETRAIN)
        pt = pretrain.Pretrain(vec_filename=filename, save_to_file=False)
        assert pt.vocab._text_normalization is pretrain.TextNormalization.NONE

def test_normalization_none_saves_and_loads():
    """
    TextNormalization.NONE round-trips correctly through save/load
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdir:
        txt_file = write_pretrain_txt(tmpdir, NO_HEADER_PRETRAIN)
        pt_file = os.path.join(tmpdir, "tiny.pt")
        pt = pretrain.Pretrain(filename=pt_file, vec_filename=txt_file,
                               text_normalization=pretrain.TextNormalization.NONE)
        pt.load()
        pt2 = pretrain.Pretrain(filename=pt_file)
        assert pt2.vocab._text_normalization is pretrain.TextNormalization.NONE

def test_normalization_ota_saves_and_loads():
    """
    TextNormalization.OTA round-trips correctly through save/load
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdir:
        txt_file = write_pretrain_txt(tmpdir, OTA_PRETRAIN)
        pt_file = os.path.join(tmpdir, "ota.pt")
        pt = pretrain.Pretrain(filename=pt_file, vec_filename=txt_file,
                               text_normalization=pretrain.TextNormalization.OTA)
        pt.load()
        pt2 = pretrain.Pretrain(filename=pt_file)
        assert pt2.vocab._text_normalization is pretrain.TextNormalization.OTA

def test_normalization_ota_lookup():
    """
    OTA normalization converts input text to the stored form at lookup time.
    Words in the alternate orthographic convention (with diacritics, uppercase, etc.)
    should resolve to the same embedding as their normalized equivalents.
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdir:
        txt_file = write_pretrain_txt(tmpdir, OTA_PRETRAIN)
        pt = pretrain.Pretrain(vec_filename=txt_file, save_to_file=False,
                               text_normalization=pretrain.TextNormalization.OTA)

        # words already in normalized form should be found directly
        assert 'pâdişâh' in pt.vocab
        assert 'gazel' in pt.vocab
        assert 'kitâb' in pt.vocab

        # words in the alternate convention should normalize to the stored form
        assert pt.vocab.unit2id('Pādişāh') == pt.vocab.unit2id('pâdişâh')
        assert pt.vocab.unit2id('ġazel') == pt.vocab.unit2id('gazel')
        assert pt.vocab.unit2id('kitāb') == pt.vocab.unit2id('kitâb')

def test_normalization_none_no_ota_conversion():
    """
    A pretrain with TextNormalization.NONE should NOT apply OTA conversion,
    so OTA diacritic forms are treated as distinct (unknown) words.
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdir:
        txt_file = write_pretrain_txt(tmpdir, OTA_PRETRAIN)
        pt = pretrain.Pretrain(vec_filename=txt_file, save_to_file=False,
                               text_normalization=pretrain.TextNormalization.NONE)
        # ġazel should NOT map to gazel under NONE normalization
        assert pt.vocab.unit2id('ġazel') != pt.vocab.unit2id('gazel')

def test_old_model_missing_normalization_flag():
    """
    A .pt file saved without a text_normalization key (pre-1.13 model) should
    load without error and default to TextNormalization.NONE
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdir:
        txt_file = write_pretrain_txt(tmpdir, NO_HEADER_PRETRAIN)
        pt_file = os.path.join(tmpdir, "old_model.pt")

        # build a pt file the old way, without text_normalization in the dict
        pt = pretrain.Pretrain(vec_filename=txt_file, save_to_file=False)
        old_style_data = {'vocab': pt.vocab.state_dict(), 'emb': pt.emb}
        torch.save(old_style_data, pt_file, _use_new_zipfile_serialization=False)

        pt2 = pretrain.Pretrain(filename=pt_file)
        assert pt2.vocab._text_normalization is pretrain.TextNormalization.NONE
        check_embedding(pt2.emb)
