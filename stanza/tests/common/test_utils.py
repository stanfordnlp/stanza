import lzma
import os
import tempfile

import pytest

import stanza
import stanza.models.common.utils as utils
from stanza.tests import *

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def test_wordvec_not_found():
    """
    get_wordvec_file should fail if neither word2vec nor fasttext exists
    """
    with tempfile.TemporaryDirectory(dir=f'{TEST_WORKING_DIR}/out') as temp_dir:
        with pytest.raises(FileNotFoundError):
            utils.get_wordvec_file(wordvec_dir=temp_dir, shorthand='en_foo')


def test_word2vec_xz():
    """
    Test searching for word2vec and xz files
    """
    with tempfile.TemporaryDirectory(dir=f'{TEST_WORKING_DIR}/out') as temp_dir:
        # make a fake directory for English word vectors
        word2vec_dir = os.path.join(temp_dir, 'word2vec', 'English')
        os.makedirs(word2vec_dir)

        # make a fake English word vector file
        fake_file = os.path.join(word2vec_dir, 'en.vectors.xz')
        fout = open(fake_file, 'w')
        fout.close()

        # get_wordvec_file should now find this fake file
        filename = utils.get_wordvec_file(wordvec_dir=temp_dir, shorthand='en_foo')
        assert filename == fake_file

def test_fasttext_txt():
    """
    Test searching for fasttext and txt files
    """
    with tempfile.TemporaryDirectory(dir=f'{TEST_WORKING_DIR}/out') as temp_dir:
        # make a fake directory for English word vectors
        fasttext_dir = os.path.join(temp_dir, 'fasttext', 'English')
        os.makedirs(fasttext_dir)

        # make a fake English word vector file
        fake_file = os.path.join(fasttext_dir, 'en.vectors.txt')
        fout = open(fake_file, 'w')
        fout.close()

        # get_wordvec_file should now find this fake file
        filename = utils.get_wordvec_file(wordvec_dir=temp_dir, shorthand='en_foo')
        assert filename == fake_file

def test_wordvec_type():
    """
    If we supply our own wordvec type, get_wordvec_file should find that
    """
    with tempfile.TemporaryDirectory(dir=f'{TEST_WORKING_DIR}/out') as temp_dir:
        # make a fake directory for English word vectors
        google_dir = os.path.join(temp_dir, 'google', 'English')
        os.makedirs(google_dir)

        # make a fake English word vector file
        fake_file = os.path.join(google_dir, 'en.vectors.txt')
        fout = open(fake_file, 'w')
        fout.close()

        # get_wordvec_file should now find this fake file
        filename = utils.get_wordvec_file(wordvec_dir=temp_dir, shorthand='en_foo', wordvec_type='google')
        assert filename == fake_file

        # this file won't be found using the normal defaults
        with pytest.raises(FileNotFoundError):
            utils.get_wordvec_file(wordvec_dir=temp_dir, shorthand='en_foo')

def test_sort_with_indices():
    data = [[1, 2, 3], [4, 5], [6]]
    ordered, orig_idx = utils.sort_with_indices(data, key=len)
    assert ordered == ([6], [4, 5], [1, 2, 3])
    assert orig_idx == (2, 1, 0)

    unsorted = utils.unsort(ordered, orig_idx)
    assert data == unsorted

def test_empty_sort_with_indices():
    ordered, orig_idx = utils.sort_with_indices([])
    assert len(ordered) == 0
    assert len(orig_idx) == 0

    unsorted = utils.unsort(ordered, orig_idx)
    assert [] == unsorted


def test_split_into_batches():
    data = []
    for i in range(5):
        data.append(["Unban", "mox", "opal", str(i)])

    data.append(["Do", "n't", "ban", "Urza", "'s", "Saga", "that", "card", "is", "great"])
    data.append(["Ban", "Ragavan"])

    # small batches will put one element in each interval
    batches = utils.split_into_batches(data, 5)
    assert batches == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]

    # this one has a batch interrupted in the middle by a large element
    batches = utils.split_into_batches(data, 8)
    assert batches == [(0, 2), (2, 4), (4, 5), (5, 6), (6, 7)]

    # this one has the large element at the start of its own batch
    batches = utils.split_into_batches(data[1:], 8)
    assert batches == [(0, 2), (2, 4), (4, 5), (5, 6)]

    # overloading the test!  assert that the key & reverse is working
    ordered, orig_idx = utils.sort_with_indices(data, key=len, reverse=True)
    assert [len(x) for x in ordered] == [10, 4, 4, 4, 4, 4, 2]

    # this has the large element at the start
    batches = utils.split_into_batches(ordered, 8)
    assert batches == [(0, 1), (1, 3), (3, 5), (5, 7)]

    # double check that unsort is working as expected
    assert data == utils.unsort(ordered, orig_idx)


def test_find_missing_tags():
    assert utils.find_missing_tags(["O", "PER", "LOC"], ["O", "PER", "LOC"]) == []
    assert utils.find_missing_tags(["O", "PER", "LOC"], ["O", "PER", "LOC", "ORG"]) == ['ORG']
    assert utils.find_missing_tags([["O", "PER"], ["O", "LOC"]], [["O", "PER"], ["LOC", "ORG"]]) == ['ORG']


def test_open_read_text():
    """
    test that we can read either .xz or regular txt
    """
    TEXT = "this is a test"
    with tempfile.TemporaryDirectory() as tempdir:
        # test text file
        filename = os.path.join(tempdir, "foo.txt")
        with open(filename, "w") as fout:
            fout.write(TEXT)
        with utils.open_read_text(filename) as fin:
            in_text = fin.read()
            assert TEXT == in_text

        assert fin.closed

        # the context should close the file when we throw an exception!
        try:
            with utils.open_read_text(filename) as finex:
                assert not finex.closed
                raise ValueError("unban mox opal!")
        except ValueError:
            pass
        assert finex.closed

        # test xz file
        filename = os.path.join(tempdir, "foo.txt.xz")
        with lzma.open(filename, "wt") as fout:
            fout.write(TEXT)
        with utils.open_read_text(filename) as finxz:
            in_text = finxz.read()
            assert TEXT == in_text

        assert finxz.closed

        # the context should close the file when we throw an exception!
        try:
            with utils.open_read_text(filename) as finexxz:
                assert not finexxz.closed
                raise ValueError("unban mox opal!")
        except ValueError:
            pass
        assert finexxz.closed


def test_checkpoint_name():
    """
    Test some expected results for the checkpoint names
    """
    # use os.path.split so that the test is agnostic of file separator on Linux or Windows
    checkpoint = utils.checkpoint_name("saved_models", "kk_oscar_forward_charlm.pt", None)
    assert os.path.split(checkpoint) == ("saved_models", "kk_oscar_forward_charlm_checkpoint.pt")

    checkpoint = utils.checkpoint_name("saved_models", "kk_oscar_forward_charlm", None)
    assert os.path.split(checkpoint) == ("saved_models", "kk_oscar_forward_charlm_checkpoint")

    checkpoint = utils.checkpoint_name("saved_models", "kk_oscar_forward_charlm", "othername.pt")
    assert os.path.split(checkpoint) == ("saved_models", "othername.pt")

