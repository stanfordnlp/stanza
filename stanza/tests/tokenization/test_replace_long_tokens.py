"""
Check to make sure long tokens are replaced with "UNK" by the tokenization processor
"""
import pytest
import stanza

from stanza.tests import TEST_MODELS_DIR

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def test_replace_long_tokens():
    nlp = stanza.Pipeline(lang="en", model_dir=TEST_MODELS_DIR, processors="tokenize")

    test_str = "foo " + "x" * 100000 + " bar"

    res = nlp(test_str)

    assert res.sentences[0].words[1].text == "<UNK>"
