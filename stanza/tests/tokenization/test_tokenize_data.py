"""
Very simple test of the mwt counting functionality in tokenization/data.py

TODO: could add a bunch more simple tests, including tests of reading
the data from a temp file, for example
"""

import pytest
import tempfile
import stanza

from stanza import Pipeline
from stanza.tests import *
from stanza.models.tokenization.data import DataLoader

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def write_tokenizer_input(test_dir, raw_text, labels):
    """
    Writes raw_text and labels to randomly named files in test_dir

    Note that the tempfiles are not set to automatically clean up.
    This will not be a problem if you put them in a tempdir.
    """
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=test_dir, delete=False) as fout:
        txt_file = fout.name
        fout.write(raw_text)

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=test_dir, delete=False) as fout:
        label_file = fout.name
        fout.write(labels)

    return txt_file, label_file

# A single slice of the German tokenization data with no MWT in it
NO_MWT_DATA = [[('S', 0), ('e', 0), ('h', 0), ('r', 1), (' ', 0), ('g', 0), ('u', 0), ('t', 0), ('e', 1), (' ', 0), ('B', 0), ('e', 0), ('r', 0), ('a', 0), ('t', 0), ('u', 0), ('n', 0), ('g', 1), (',', 1), (' ', 0), ('s', 0), ('c', 0), ('h', 0), ('n', 0), ('e', 0), ('l', 0), ('l', 0), ('e', 1), (' ', 0), ('B', 0), ('e', 0), ('h', 0), ('e', 0), ('b', 0), ('u', 0), ('n', 0), ('g', 1), (' ', 0), ('d', 0), ('e', 0), ('r', 1), (' ', 0), ('P', 0), ('r', 0), ('o', 0), ('b', 0), ('l', 0), ('e', 0), ('m', 0), ('e', 2)]]

# A single slice of the German tokenization data with an MWT in it
MWT_DATA = [[(' ', 0), ('D', 0), ('i', 0), ('e', 1), (' ', 0), ('K', 0), ('o', 0), ('s', 0), ('t', 0), ('e', 0), ('n', 1), (' ', 0), ('s', 0), ('i', 0), ('n', 0), ('d', 1), (' ', 0), ('d', 0), ('e', 0), ('f', 0), ('i', 0), ('n', 0), ('i', 0), ('t', 0), ('i', 0), ('v', 1), (' ', 0), ('a', 0), ('u', 0), ('c', 0), ('h', 1), (' ', 0), ('i', 0), ('m', 3), (' ', 0), ('R', 0), ('a', 0), ('h', 0), ('m', 0), ('e', 0), ('n', 1), ('.', 2)]]

FAKE_PROPERTIES = {
    "lang":"de",
    'feat_funcs': ("space_before","capitalized"),
    'max_seqlen': 300,
    'use_dictionary': False,
}

def test_has_mwt():
    """
    One dataset has no mwt, the other does
    """
    data = DataLoader(args=FAKE_PROPERTIES, input_data=NO_MWT_DATA)
    assert not data.has_mwt()

    data = DataLoader(args=FAKE_PROPERTIES, input_data=MWT_DATA)
    assert data.has_mwt()

@pytest.fixture(scope="module")
def tokenizer():
    pipeline = Pipeline("en", dir=TEST_MODELS_DIR, download_method=None, processors="tokenize")
    tokenizer = pipeline.processors['tokenize']
    return tokenizer

@pytest.fixture(scope="module")
def zhtok():
    pipeline = Pipeline("zh-hans", dir=TEST_MODELS_DIR, download_method=None, processors="tokenize")
    tokenizer = pipeline.processors['tokenize']
    return tokenizer

EXPECTED_TWO_NL_RAW = [[('T', 0), ('h', 0), ('i', 0), ('s', 0), (' ', 0), ('i', 0), ('s', 0), (' ', 0), ('a', 0), (' ', 0), ('t', 0), ('e', 0), ('s', 0), ('t', 0)], [('f', 0), ('o', 0), ('o', 0)]]
# in this test, the newline after test becomes a space labeled 0
EXPECTED_ONE_NL_RAW = [[('T', 0), ('h', 0), ('i', 0), ('s', 0), (' ', 0), ('i', 0), ('s', 0), (' ', 0), ('a', 0), (' ', 0), ('t', 0), ('e', 0), ('s', 0), ('t', 0), (' ', 0), ('f', 0), ('o', 0), ('o', 0)]]
EXPECTED_SKIP_NL_RAW = [[('T', 0), ('h', 0), ('i', 0), ('s', 0), (' ', 0), ('i', 0), ('s', 0), (' ', 0), ('a', 0), (' ', 0), ('t', 0), ('e', 0), ('s', 0), ('t', 0), ('f', 0), ('o', 0), ('o', 0)]]

def test_convert_units_raw_text(tokenizer):
    """
    Tests converting a couple small segments to units
    """
    raw_text = "This is a      test\n\nfoo"
    batches = DataLoader(tokenizer.config, input_text=raw_text, vocab=tokenizer.vocab, evaluation=True, dictionary=tokenizer.trainer.dictionary)
    assert batches.data == EXPECTED_TWO_NL_RAW

    raw_text = "This is a      test\nfoo"
    batches = DataLoader(tokenizer.config, input_text=raw_text, vocab=tokenizer.vocab, evaluation=True, dictionary=tokenizer.trainer.dictionary)
    assert batches.data == EXPECTED_ONE_NL_RAW

    skip_newline_config = dict(tokenizer.config)
    skip_newline_config['skip_newline'] = True
    batches = DataLoader(skip_newline_config, input_text=raw_text, vocab=tokenizer.vocab, evaluation=True, dictionary=tokenizer.trainer.dictionary)
    assert batches.data == EXPECTED_SKIP_NL_RAW


EXPECTED_TWO_NL_FILE = [[('T', 0), ('h', 0), ('i', 0), ('s', 0), (' ', 0), ('i', 0), ('s', 0), (' ', 0), ('a', 0), (' ', 0), ('t', 0), ('e', 0), ('s', 0), ('t', 0), ('.', 1)], [('f', 0), ('o', 0), ('o', 0)]]
# in this test, the newline after test becomes a space labeled 0
EXPECTED_ONE_NL_FILE = [[('T', 0), ('h', 0), ('i', 0), ('s', 0), (' ', 0), ('i', 0), ('s', 0), (' ', 0), ('a', 0), (' ', 0), ('t', 0), ('e', 0), ('s', 0), ('t', 0), ('.', 1), (' ', 0), ('f', 0), ('o', 0), ('o', 0)]]

def test_convert_units_file(tokenizer):
    """
    Tests reading some text from a file and converting that to units
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        # two nl test case, read from file
        labels   = "00000000000000000001\n\n000\n\n"
        raw_text = "This is a      test.\n\nfoo\n\n"
        txt_file, label_file = write_tokenizer_input(test_dir, raw_text, labels)

        batches = DataLoader(tokenizer.config, input_files={'txt': txt_file, 'label': label_file}, vocab=tokenizer.vocab, evaluation=True, dictionary=tokenizer.trainer.dictionary)
        assert batches.data == EXPECTED_TWO_NL_FILE

        # one nl test case, read from file
        labels   = "000000000000000000010000\n\n"
        raw_text = "This is a      test.\nfoo\n\n"
        txt_file, label_file = write_tokenizer_input(test_dir, raw_text, labels)

        batches = DataLoader(tokenizer.config, input_files={'txt': txt_file, 'label': label_file}, vocab=tokenizer.vocab, evaluation=True, dictionary=tokenizer.trainer.dictionary)
        assert batches.data == EXPECTED_ONE_NL_FILE

def test_dictionary(zhtok):
    """
    Tests some features of the zh tokenizer dictionary

    The expectation is that the Chinese tokenizer will be serialized with a dictionary
    (if it ever gets serialized without, this test will warn us!)
    """
    assert zhtok.trainer.lexicon is not None
    assert zhtok.trainer.dictionary is not None

    assert "老师" in zhtok.trainer.lexicon
    # egg-white-stuff, eg protein
    assert "蛋白质" in zhtok.trainer.lexicon
    # egg-white
    assert "蛋白" in zhtok.trainer.dictionary['prefixes']
    # egg
    assert "蛋" in zhtok.trainer.dictionary['prefixes']
    # white-stuff
    assert "白质" in zhtok.trainer.dictionary['suffixes']
    # stuff
    assert "质" in zhtok.trainer.dictionary['suffixes']

def test_dictionary_feats(zhtok):
    """
    Test the results of running a sentence into the dictionary featurizer
    """
    raw_text = "我想吃蛋白质"
    batches = DataLoader(zhtok.config, input_text=raw_text, vocab=zhtok.vocab, evaluation=True, dictionary=zhtok.trainer.dictionary)
    data = batches.data
    assert len(data) == 1
    assert len(data[0]) == 6

    expected_features = [
        # in our example, the 2-grams made by the one character words at the start
        # don't form any prefixes or suffixes
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
    ]

    for i, expected in enumerate(expected_features):
        dict_features = batches.extract_dict_feat(data[0], i)
        assert dict_features == expected
