"""
Check to make sure long tokens are replaced with "UNK" by the tokenization processor
"""
import pytest
import stanza

from stanza.pipeline import tokenize_processor

from stanza.tests import TEST_MODELS_DIR

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def test_replace_long_tokens():
    nlp = stanza.Pipeline(lang="en", download_method=None, model_dir=TEST_MODELS_DIR, processors="tokenize")

    test_str = "foo " + "x" * 10000 + " bar"

    res = nlp(test_str)

    assert res.sentences[0].words[1].text == tokenize_processor.TOKEN_TOO_LONG_REPLACEMENT

def test_set_max_len():
    nlp = stanza.Pipeline(**{'processors': 'tokenize', 'dir': TEST_MODELS_DIR,
                             'lang': 'en',
                             'download_method': None,
                             'tokenize_max_seqlen': 20})
    doc = nlp("This is a doc withaverylongtokenthatshouldbereplaced")
    assert len(doc.sentences) == 1
    assert len(doc.sentences[0].words) == 5
    assert doc.sentences[0].words[-1].text == tokenize_processor.TOKEN_TOO_LONG_REPLACEMENT
