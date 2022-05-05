"""
Currently tests a few configurations of files for creating a charlm vocab

Also has a skeleton test of loading & saving a charlm
"""

from collections import Counter
import glob
import lzma
import os
import tempfile

import pytest

from stanza.models.common import char_model
from stanza.tests import TEST_MODELS_DIR

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

fake_text_1 = """
Unban mox opal!
I hate watching Peppa Pig
"""

fake_text_2 = """
This is plastic cheese
"""

def test_single_file_vocab():
    with tempfile.TemporaryDirectory() as tempdir:
        sample_file = os.path.join(tempdir, "text.txt")
        with open(sample_file, "w", encoding="utf-8") as fout:
            fout.write(fake_text_1)
        vocab = char_model.build_charlm_vocab(sample_file)

    for i in fake_text_1:
        assert i in vocab
    assert "Q" not in vocab

def test_single_file_xz_vocab():
    with tempfile.TemporaryDirectory() as tempdir:
        sample_file = os.path.join(tempdir, "text.txt.xz")
        with lzma.open(sample_file, "wt", encoding="utf-8") as fout:
            fout.write(fake_text_1)
        vocab = char_model.build_charlm_vocab(sample_file)

    for i in fake_text_1:
        assert i in vocab
    assert "Q" not in vocab

def test_single_file_dir_vocab():
    with tempfile.TemporaryDirectory() as tempdir:
        sample_file = os.path.join(tempdir, "text.txt")
        with open(sample_file, "w", encoding="utf-8") as fout:
            fout.write(fake_text_1)
        vocab = char_model.build_charlm_vocab(tempdir)

    for i in fake_text_1:
        assert i in vocab
    assert "Q" not in vocab

def test_multiple_files_vocab():
    with tempfile.TemporaryDirectory() as tempdir:
        sample_file = os.path.join(tempdir, "t1.txt")
        with open(sample_file, "w", encoding="utf-8") as fout:
            fout.write(fake_text_1)
        sample_file = os.path.join(tempdir, "t2.txt.xz")
        with lzma.open(sample_file, "wt", encoding="utf-8") as fout:
            fout.write(fake_text_2)
        vocab = char_model.build_charlm_vocab(tempdir)

    for i in fake_text_1:
        assert i in vocab
    for i in fake_text_2:
        assert i in vocab
    assert "Q" not in vocab

def test_cutoff_vocab():
    with tempfile.TemporaryDirectory() as tempdir:
        sample_file = os.path.join(tempdir, "t1.txt")
        with open(sample_file, "w", encoding="utf-8") as fout:
            fout.write(fake_text_1)
        sample_file = os.path.join(tempdir, "t2.txt.xz")
        with lzma.open(sample_file, "wt", encoding="utf-8") as fout:
            fout.write(fake_text_2)

        vocab = char_model.build_charlm_vocab(tempdir, cutoff=2)

    counts = Counter(fake_text_1) + Counter(fake_text_2)
    for letter, count in counts.most_common():
        if count < 2:
            assert letter not in vocab
        else:
            assert letter in vocab


@pytest.fixture
def english_forward():
    # eg, stanza_test/models/en/forward_charlm/1billion.pt
    models_path = os.path.join(TEST_MODELS_DIR, "en", "forward_charlm", "*")
    models = glob.glob(models_path)
    # we expect at least one English model downloaded for the tests
    assert len(models) >= 1
    model_file = models[0]
    return char_model.CharacterLanguageModel.load(model_file)

@pytest.fixture
def english_backward():
    # eg, stanza_test/models/en/forward_charlm/1billion.pt
    models_path = os.path.join(TEST_MODELS_DIR, "en", "backward_charlm", "*")
    models = glob.glob(models_path)
    # we expect at least one English model downloaded for the tests
    assert len(models) >= 1
    model_file = models[0]
    return char_model.CharacterLanguageModel.load(model_file)

def test_load_model(english_forward, english_backward):
    """
    Check that basic loading functions work
    """
    assert english_forward.is_forward_lm
    assert not english_backward.is_forward_lm

def test_save_load_model(english_forward, english_backward):
    """
    Load, save, and load again
    """
    with tempfile.TemporaryDirectory() as tempdir:
        for charlm in (english_forward, english_backward):
            save_file = os.path.join(tempdir, "resaved", "charlm.pt")
            charlm.save(save_file)
            reloaded = char_model.CharacterLanguageModel.load(save_file)
            assert charlm.is_forward_lm == reloaded.is_forward_lm
