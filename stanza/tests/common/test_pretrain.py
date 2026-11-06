import gzip
import lzma
import os
import struct
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

def write_word2vec_binary(filename, entries):
    """
    Write entries of (word, vector) in the word2vec binary format
    """
    cols = len(entries[0][1])
    with open(filename, "wb") as fout:
        fout.write(("%d %d\n" % (len(entries), cols)).encode())
        for word, vector in entries:
            fout.write(word.encode() + b" ")
            fout.write(np.array(vector, dtype='<f4').tobytes())
            fout.write(b"\n")

W2V_ENTRIES = [("unban", [1, 2, 3, 4]),
               ("mox",   [5, 6, 7, 8]),
               ("opal",  [9, 10, 11, 12])]

def test_word2vec_binary():
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdir:
        filename = os.path.join(tmpdir, "tiny.bin")
        write_word2vec_binary(filename, W2V_ENTRIES)
        assert pretrain.detect_vector_format(filename) == "word2vec_binary"
        pt = pretrain.Pretrain(vec_filename=filename, save_to_file=False)
        check_pretrain(pt)

def test_word2vec_binary_unk():
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdir:
        filename = os.path.join(tmpdir, "tiny.bin")
        write_word2vec_binary(filename, W2V_ENTRIES + [("<unk>", [-1, -1, -1, -1])])
        pt = pretrain.Pretrain(vec_filename=filename, save_to_file=False)
        check_vocab(pt.vocab)
        check_embedding(pt.emb, unk=True)

def test_word2vec_binary_max_vocab():
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdir:
        filename = os.path.join(tmpdir, "tiny.bin.gz")
        write_word2vec_binary(os.path.join(tmpdir, "tiny.bin"), W2V_ENTRIES)
        with open(os.path.join(tmpdir, "tiny.bin"), "rb") as fin, gzip.open(filename, "wb") as fout:
            fout.write(fin.read())
        words, emb, failed = pretrain.Pretrain.read_from_file(filename, max_vocab=2)
        assert words == ["unban", "mox"]
        np.testing.assert_allclose(emb[4:], [[1, 2, 3, 4], [5, 6, 7, 8]])

def test_text_named_bin():
    """
    A text file is read as text no matter what it is called
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdir:
        filename = os.path.join(tmpdir, "tiny.bin")
        with open(filename, "w", encoding="utf-8") as fout:
            fout.write(SPACE_PRETRAIN)
        assert pretrain.detect_vector_format(filename) == "text"
        pt = pretrain.Pretrain(vec_filename=filename, save_to_file=False)
        check_embedding(pt.emb)
        assert "unban\xa0mox" in pt.vocab

def write_fasttext_binary(filename, words, dim=4, minn=2, maxn=3, bucket=11):
    """
    Write a tiny fastText .bin model, following FastText::saveModel

    Row r of the input matrix is [r, 10r, r*r, 1], so a word vector
    is the average of those rows over the word and its n-grams.
    The output matrix is all zeros, since only the input matrix
    matters for word vectors.
    """
    nwords = len(words)
    rows = nwords + bucket
    with open(filename, "wb") as fout:
        fout.write(struct.pack('<ii', pretrain.FASTTEXT_MAGIC, 12))
        # dim, ws, epoch, minCount, neg, wordNgrams, loss=ns, model=skipgram, bucket, minn, maxn, lrUpdateRate, t
        fout.write(struct.pack('<12id', dim, 5, 5, 1, 5, 1, 2, 2, bucket, minn, maxn, 100, 1e-4))
        # size, nwords, nlabels, ntokens, pruneidx_size
        fout.write(struct.pack('<iiiqq', nwords, nwords, 0, 1000, -1))
        for idx, word in enumerate(words):
            fout.write(word.encode() + b'\0')
            # count, then entry type 0 for a word
            fout.write(struct.pack('<qb', 100 - idx, 0))
        fout.write(struct.pack('<?qq', False, rows, dim))
        matrix = np.array([[r, 10 * r, r * r, 1] for r in range(rows)], dtype='<f4')
        fout.write(matrix.tobytes())
        fout.write(struct.pack('<?qq', False, nwords, dim))
        fout.write(np.zeros((nwords, dim), dtype='<f4').tobytes())

# </s> gets no n-grams, and żółw has multibyte characters,
# whose bytes fastText sign extends when hashing
FASTTEXT_WORDS = ["</s>", "unban", "mox", "żółw"]

# values from get_word_vector in the fastText library on the file
# written by write_fasttext_binary(FASTTEXT_WORDS)
FASTTEXT_EXPECTED = np.array([[0.0,     0.0,   0.0,  1.0],
                              [9.25,   92.5,  97.75,  1.0],
                              [7.625,  76.25, 73.375, 1.0],
                              [8.5,    85.0,  86.7,   1.0]])

def test_fasttext_binary():
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdir:
        filename = os.path.join(tmpdir, "tiny.bin")
        write_fasttext_binary(filename, FASTTEXT_WORDS)
        assert pretrain.detect_vector_format(filename) == "fasttext"
        pt = pretrain.Pretrain(vec_filename=filename, save_to_file=False)
        assert len(pt.vocab) == len(FASTTEXT_WORDS) + 4
        for word, expected in zip(FASTTEXT_WORDS, FASTTEXT_EXPECTED):
            np.testing.assert_allclose(pt.emb[pt.vocab.unit2id(word)], expected, rtol=1e-6)
        np.testing.assert_allclose(pt.emb[:4], np.zeros((4, 4)))

def test_fasttext_binary_max_vocab():
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdir:
        filename = os.path.join(tmpdir, "tiny.bin.xz")
        raw_filename = os.path.join(tmpdir, "tiny.bin")
        write_fasttext_binary(raw_filename, FASTTEXT_WORDS)
        with open(raw_filename, "rb") as fin, lzma.open(filename, "wb") as fout:
            fout.write(fin.read())
        words, emb, failed = pretrain.Pretrain.read_from_file(filename, max_vocab=2)
        assert words == FASTTEXT_WORDS[:2]
        np.testing.assert_allclose(emb[4:], FASTTEXT_EXPECTED[:2], rtol=1e-6)

def test_fasttext_quantized():
    """
    Quantized models are rejected rather than misread
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdir:
        filename = os.path.join(tmpdir, "tiny.bin")
        write_fasttext_binary(filename, FASTTEXT_WORDS)
        with open(filename, "rb") as fin:
            data = bytearray(fin.read())
        # the quant flag comes right after the dictionary
        dict_end = 8 + 56 + 28 + sum(len(word.encode()) + 1 + 9 for word in FASTTEXT_WORDS)
        assert data[dict_end] == 0
        data[dict_end] = 1
        with open(filename, "wb") as fout:
            fout.write(data)
        with pytest.raises(ValueError, match="quantized"):
            pretrain.Pretrain.read_from_file(filename)
