import pytest
from stanza.tests import *

from stanza.models.common.doc import Document
import stanza.server.tokensregex as tokensregex

pytestmark = [pytest.mark.travis, pytest.mark.client]

from stanza.tests.test_semgrex import ONE_SENTENCE_DOC, TWO_SENTENCE_DOC

def test_single_sentence():
    #expected:
    #match {
    #  sentence: 0
    #  match {
    #    text: "Opal"
    #    begin: 2
    #    end: 3
    #  }
    #}

    response = tokensregex.process_doc(ONE_SENTENCE_DOC, "Opal")
    assert len(response.match) == 1
    assert len(response.match[0].match) == 1
    assert response.match[0].match[0].sentence == 0
    assert response.match[0].match[0].match.text == "Opal"
    assert response.match[0].match[0].match.begin == 2
    assert response.match[0].match[0].match.end == 3


def test_ner_sentence():
    #expected:
    #match {
    #  sentence: 0
    #  match {
    #    text: "Opal"
    #    begin: 2
    #    end: 3
    #  }
    #}

    response = tokensregex.process_doc(ONE_SENTENCE_DOC, "[ner: GEM]")
    assert len(response.match) == 1
    assert len(response.match[0].match) == 1
    assert response.match[0].match[0].sentence == 0
    assert response.match[0].match[0].match.text == "Opal"
    assert response.match[0].match[0].match.begin == 2
    assert response.match[0].match[0].match.end == 3
