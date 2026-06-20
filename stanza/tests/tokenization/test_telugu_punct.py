import pytest

import stanza
from stanza.tests import TEST_MODELS_DIR

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]


@pytest.fixture(scope="module")
def te_tokenize_pipeline():
    """Tokenize-only Telugu pipeline (no MWT, pos, etc.)."""
    return stanza.Pipeline(
        "te",
        model_dir=TEST_MODELS_DIR,
        download_method=None,
        processors="tokenize",
    )

def test_sentence_end(te_tokenize_pipeline):
    """
    As the TE UD dataset has all the punct separated from the words,
    there was an augmentation pattern needed to get it to properly tokenize
    examples such as the one in this test
    """
    text = 'సింక్ ముందు ఉన్నారు.'
    assert text[-1] == '.'
    assert text[-2] != ' '
    doc = te_tokenize_pipeline(text)
    assert len(doc.sentences) == 1
    assert len(doc.sentences[0].words) == 4
    assert doc.sentences[0].words[-1].text == '.'

def test_comma(te_tokenize_pipeline):
    """
    Also test the separation of commas which are not separated by whitespace in the text
    """
    text = 'రాము వస్తాడో, రాడో.'
    idx = text.find(',')
    assert idx >= 1
    assert text[idx-1] != ' '

    doc = te_tokenize_pipeline(text)
    assert len(doc.sentences) == 1
    assert len(doc.sentences[0].words) == 5
    assert doc.sentences[0].words[2].text == ','


def test_with_spaces(te_tokenize_pipeline):
    """
    Also check the versions with space separations, just in case
    """
    text = 'రాము వస్తాడో , రాడో .'
    idx = text.find(',')
    assert idx >= 1
    assert text[idx-1] == ' '
    assert text[-1] == '.'
    assert text[-2] == ' '

    doc = te_tokenize_pipeline(text)
    assert len(doc.sentences) == 1
    assert len(doc.sentences[0].words) == 5
    assert doc.sentences[0].words[2].text == ','
    assert doc.sentences[0].words[4].text == '.'
