"""
Unit tests for the word vector readers in stanza.utils.pretrain.vector_files

The fixtures are written by hand in each format, so the layout each
reader expects is visible here.

Run with:
    pytest stanza/tests/utils/test_vector_files.py
"""

import gzip
import lzma
import struct

import numpy as np
import pytest

from stanza.models.common.vocab import UNK_ID
from stanza.utils.pretrain import vector_files

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

EXPECTED = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

def compress(path, opener, suffix):
    compressed = path.with_name(path.name + suffix)
    with open(path, "rb") as fin, opener.open(compressed, "wb") as fout:
        fout.write(fin.read())
    return str(compressed)

def check_prefix_rows(emb, unk=None):
    expected = np.zeros((4, emb.shape[1]))
    if unk is not None:
        expected[UNK_ID] = unk
    np.testing.assert_allclose(emb[:4], expected)

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

W2V_ENTRIES = [("unban", EXPECTED[0]),
               ("mox",   EXPECTED[1]),
               ("opal",  EXPECTED[2])]

def test_word2vec_binary(tmp_path):
    filename = tmp_path / "tiny.bin"
    write_word2vec_binary(filename, W2V_ENTRIES)
    assert vector_files.detect_vector_format(str(filename)) == "word2vec_binary"
    words, emb, failed = vector_files.read_vector_file(str(filename))
    assert words == ["unban", "mox", "opal"]
    assert failed == 0
    check_prefix_rows(emb)
    np.testing.assert_allclose(emb[4:], EXPECTED)

def test_word2vec_binary_unk(tmp_path):
    filename = tmp_path / "tiny.bin"
    write_word2vec_binary(filename, W2V_ENTRIES + [("<unk>", [-1, -1, -1, -1])])
    words, emb, failed = vector_files.read_vector_file(str(filename))
    assert words == ["unban", "mox", "opal"]
    check_prefix_rows(emb, unk=-1)
    np.testing.assert_allclose(emb[4:], EXPECTED)

def test_word2vec_binary_max_vocab(tmp_path):
    raw_filename = tmp_path / "tiny.bin"
    write_word2vec_binary(raw_filename, W2V_ENTRIES)
    filename = compress(raw_filename, gzip, ".gz")
    words, emb, failed = vector_files.read_vector_file(filename, max_vocab=2)
    assert words == ["unban", "mox"]
    np.testing.assert_allclose(emb[4:], EXPECTED[:2])

SPACE_VECTORS = """
3 4
unban mox 1 2 3 4
opal 5 6 7 8
foo 9 10 11 12
""".strip()

def test_text_named_bin(tmp_path):
    """
    A text file is read as text no matter what it is called
    """
    filename = tmp_path / "tiny.bin"
    filename.write_text(SPACE_VECTORS, encoding="utf-8")
    assert vector_files.detect_vector_format(str(filename)) == "text"
    words, emb, failed = vector_files.read_vector_file(str(filename))
    assert words == ["unban\xa0mox", "opal", "foo"]
    np.testing.assert_allclose(emb[4:], EXPECTED)

def test_text_no_header(tmp_path):
    filename = tmp_path / "tiny.txt"
    filename.write_text("unban 1 2 3 4\nmox 5 6 7 8\nopal 9 10 11 12\n", encoding="utf-8")
    assert vector_files.detect_vector_format(str(filename)) == "text"
    words, emb, failed = vector_files.read_vector_file(str(filename))
    assert words == ["unban", "mox", "opal"]
    np.testing.assert_allclose(emb[4:], EXPECTED)

def test_csv(tmp_path):
    filename = tmp_path / "tiny.csv"
    filename.write_text("4\nunban,1,2,3,4\nmox,5,6,7,8\nopal,9,10,11,12\n", encoding="utf-8")
    words, emb = vector_files.read_csv_vectors(str(filename))
    assert words == ["unban", "mox", "opal"]
    check_prefix_rows(emb)
    np.testing.assert_allclose(emb[4:], EXPECTED)

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
        fout.write(struct.pack('<ii', vector_files.FASTTEXT_MAGIC, 12))
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

def test_fasttext_binary(tmp_path):
    filename = tmp_path / "tiny.bin"
    write_fasttext_binary(filename, FASTTEXT_WORDS)
    assert vector_files.detect_vector_format(str(filename)) == "fasttext"
    words, emb, failed = vector_files.read_vector_file(str(filename))
    assert words == FASTTEXT_WORDS
    assert failed == 0
    check_prefix_rows(emb)
    np.testing.assert_allclose(emb[4:], FASTTEXT_EXPECTED, rtol=1e-6)

def test_fasttext_binary_max_vocab(tmp_path):
    raw_filename = tmp_path / "tiny.bin"
    write_fasttext_binary(raw_filename, FASTTEXT_WORDS)
    filename = compress(raw_filename, lzma, ".xz")
    words, emb, failed = vector_files.read_vector_file(filename, max_vocab=2)
    assert words == FASTTEXT_WORDS[:2]
    np.testing.assert_allclose(emb[4:], FASTTEXT_EXPECTED[:2], rtol=1e-6)

def test_fasttext_subword_rows():
    """
    Row lists checked against get_subwords in the fastText library for the fixture model
    """
    assert vector_files.fasttext_subword_rows("unban".encode(), 2, 3, 11, 4) == [9, 11, 14, 10, 12, 10, 7, 7, 14, 6, 10]
    assert vector_files.fasttext_subword_rows("mox".encode(), 2, 3, 11, 4) == [6, 13, 4, 7, 9, 14, 6]
    assert vector_files.fasttext_subword_rows("żółw".encode(), 2, 3, 11, 4) == [14, 12, 8, 5, 4, 12, 8, 13, 6]
    # no n-grams at all when maxn is 0
    assert vector_files.fasttext_subword_rows("unban".encode(), 0, 0, 11, 4) == []

def test_fasttext_quantized(tmp_path):
    """
    Quantized models are rejected rather than misread
    """
    filename = tmp_path / "tiny.bin"
    write_fasttext_binary(filename, FASTTEXT_WORDS)
    data = bytearray(filename.read_bytes())
    # the quant flag comes right after the dictionary
    dict_end = 8 + 56 + 28 + sum(len(word.encode()) + 1 + 9 for word in FASTTEXT_WORDS)
    assert data[dict_end] == 0
    data[dict_end] = 1
    filename.write_bytes(data)
    with pytest.raises(ValueError, match="quantized"):
        vector_files.read_vector_file(str(filename))
