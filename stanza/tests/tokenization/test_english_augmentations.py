import pytest

import stanza
from stanza.tests import TEST_MODELS_DIR

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]


@pytest.fixture(scope="module")
def en_tokenize_pipeline():
    """Tokenize-only English pipeline (no pos, depparse, etc.)."""
    return stanza.Pipeline(
        "en",
        model_dir=TEST_MODELS_DIR,
        download_method=None,
        processors="tokenize",
    )

def test_comma_typos(en_tokenize_pipeline):
    text = "This, tests comma errors"
    doc = en_tokenize_pipeline(text)
    assert len(doc.sentences) == 1
    assert len(doc.sentences[0].words) == 5
    assert doc.sentences[0].words[1].text == ','

    text = "This ,tests comma errors"
    doc = en_tokenize_pipeline(text)
    assert len(doc.sentences) == 1
    assert len(doc.sentences[0].words) == 5
    assert doc.sentences[0].words[1].text == ','

def test_comma_glued(en_tokenize_pipeline):
    text = "This, tests comma gluing"
    doc = en_tokenize_pipeline(text)
    assert len(doc.sentences) == 1
    assert len(doc.sentences[0].words) == 5
    assert doc.sentences[0].words[1].text == ','

    text = "This,tests comma gluing"
    doc = en_tokenize_pipeline(text)
    assert len(doc.sentences) == 1
    assert len(doc.sentences[0].words) == 5
    assert doc.sentences[0].words[1].text == ','
