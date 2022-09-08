
import pytest
import stanza

from stanza.tests import *

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

# data for testing
TEST_TEXT = "This is a test.  Another sentence.  Are these sorted?"

TEST_TOKENS = [["This", "is", "a", "test", "."], ["Another", "sentence", "."], ["Are", "these", "sorted", "?"]]

def check_results(doc):
    assert len(doc.sentences) == len(TEST_TOKENS)
    for sentence, expected in zip(doc.sentences, TEST_TOKENS):
        assert sentence.constituency.leaf_labels() == expected

def test_sorted_big_batch():
    pipe = stanza.Pipeline("en", model_dir=TEST_MODELS_DIR, processors="tokenize,pos,constituency")
    doc = pipe(TEST_TEXT)
    check_results(doc)

def test_illegal_batch_size():
    stanza.Pipeline("en", model_dir=TEST_MODELS_DIR, processors="tokenize,pos", constituency_batch_size="zzz")
    with pytest.raises(ValueError):
        stanza.Pipeline("en", model_dir=TEST_MODELS_DIR, processors="tokenize,pos,constituency", constituency_batch_size="zzz")

def test_sorted_one_batch():
    pipe = stanza.Pipeline("en", model_dir=TEST_MODELS_DIR, processors="tokenize,pos,constituency", constituency_batch_size=1)
    doc = pipe(TEST_TEXT)
    check_results(doc)

def test_sorted_two_batch():
    pipe = stanza.Pipeline("en", model_dir=TEST_MODELS_DIR, processors="tokenize,pos,constituency", constituency_batch_size=2)
    doc = pipe(TEST_TEXT)
    check_results(doc)

def test_get_constituents():
    pipe = stanza.Pipeline("en", processors="tokenize,pos,constituency")
    assert "SBAR" in pipe.processors["constituency"].get_constituents()
