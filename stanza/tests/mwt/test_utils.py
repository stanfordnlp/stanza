"""
Test the MWT resplitting of preexisting tokens without word splits
"""

import pytest

import stanza
from stanza.models.mwt.utils import resplit_mwt

from stanza.tests import TEST_MODELS_DIR

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

@pytest.fixture(scope="module")
def pipeline():
    """
    A reusable pipeline with the NER module
    """
    return stanza.Pipeline("en", dir=TEST_MODELS_DIR, processors="tokenize,mwt", package="gum")


def test_resplit_keep_tokens(pipeline):
    """
    Test splitting with enforced token boundaries
    """
    tokens = [["I", "can't", "believe", "it"], ["I can't", "sleep"]]
    doc = resplit_mwt(tokens, pipeline)
    assert len(doc.sentences) == 2
    assert len(doc.sentences[0].tokens) == 4
    assert len(doc.sentences[0].tokens[1].words) == 2
    assert doc.sentences[0].tokens[1].words[0].text == "ca"
    assert doc.sentences[0].tokens[1].words[1].text == "n't"

    assert len(doc.sentences[1].tokens) == 2
    # updated GUM MWT splits "I can't" into three segments
    # the way we want, "I - ca - n't"
    # previously it would split "I - can - 't"
    assert len(doc.sentences[1].tokens[0].words) == 3
    assert doc.sentences[1].tokens[0].words[0].text == "I"
    assert doc.sentences[1].tokens[0].words[1].text == "ca"
    assert doc.sentences[1].tokens[0].words[2].text == "n't"


def test_resplit_no_keep_tokens(pipeline):
    """
    Test splitting without enforced token boundaries
    """
    tokens = [["I", "can't", "believe", "it"], ["I can't", "sleep"]]
    doc = resplit_mwt(tokens, pipeline, keep_tokens=False)
    assert len(doc.sentences) == 2
    assert len(doc.sentences[0].tokens) == 4
    assert len(doc.sentences[0].tokens[1].words) == 2
    assert doc.sentences[0].tokens[1].words[0].text == "ca"
    assert doc.sentences[0].tokens[1].words[1].text == "n't"

    assert len(doc.sentences[1].tokens) == 3
    assert len(doc.sentences[1].tokens[1].words) == 2
    assert doc.sentences[1].tokens[1].words[0].text == "ca"
    assert doc.sentences[1].tokens[1].words[1].text == "n't"
