import tempfile

import pytest

import stanza
import stanza.models.common.utils as utils
from tests import *

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

