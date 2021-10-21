
import pytest
import stanza
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document

from stanza.tests import *

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

# data for testing
EN_DOCS = ["Barack Obama was born in Hawaii.", "He was elected president in 2008.", "Obama attended Harvard."]

EXPECTED_ENTS = [[{
    "text": "Barack Obama",
    "type": "PERSON",
    "start_char": 0,
    "end_char": 12
}, {
    "text": "Hawaii",
    "type": "GPE",
    "start_char": 25,
    "end_char": 31
}],
[{
    "text": "2008",
    "type": "DATE",
    "start_char": 28,
    "end_char": 32
}],
[{
    "text": "Obama",
    "type": "PERSON",
    "start_char": 0,
    "end_char": 5
}, {
  "text": "Harvard",
  "type": "ORG",
  "start_char": 15,
  "end_char": 22
}]]


@pytest.fixture(scope="module")
def pipeline():
    """
    A reusable pipeline with the NER module
    """
    return stanza.Pipeline(dir=TEST_MODELS_DIR, processors="tokenize,ner")
    

@pytest.fixture(scope="module")
def processed_doc(pipeline):
    """ Document created by running full English pipeline on a few sentences """
    return [pipeline(text) for text in  EN_DOCS]


@pytest.fixture(scope="module")
def processed_bulk(pipeline):
    """ Document created by running full English pipeline on a few sentences """
    docs = [Document([], text=t) for t in EN_DOCS]
    return pipeline(docs)

def check_entities_equal(doc, expected):
    """
    Checks that the entities of a doc are equal to the given list of maps
    """
    assert len(doc.ents) == len(expected)
    for doc_entity, expected_entity in zip(doc.ents, expected):
        for k in expected_entity:
            assert getattr(doc_entity, k) == expected_entity[k]

def test_bulk_ents(processed_bulk):
    assert len(processed_bulk) == len(EXPECTED_ENTS)
    for doc, expected in zip(processed_bulk, EXPECTED_ENTS):
        check_entities_equal(doc, expected)

def test_ents(processed_doc):
    assert len(processed_doc) == len(EXPECTED_ENTS)
    for doc, expected in zip(processed_doc, EXPECTED_ENTS):
        check_entities_equal(doc, expected)
