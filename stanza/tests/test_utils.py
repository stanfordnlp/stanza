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
