
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


def check_entities_equal(doc, expected):
    """
    Checks that the entities of a doc are equal to the given list of maps
    """
    assert len(doc.ents) == len(expected)
    for doc_entity, expected_entity in zip(doc.ents, expected):
        for k in expected_entity:
            assert getattr(doc_entity, k) == expected_entity[k]

class TestNERProcessor:
    @pytest.fixture(scope="class")
    def pipeline(self):
        """
        A reusable pipeline with the NER module
        """
        return stanza.Pipeline(dir=TEST_MODELS_DIR, processors="tokenize,ner")

    @pytest.fixture(scope="class")
    def processed_doc(self, pipeline):
        """ Document created by running full English pipeline on a few sentences """
        return [pipeline(text) for text in EN_DOCS]


    @pytest.fixture(scope="class")
    def processed_bulk(self, pipeline):
        """ Document created by running full English pipeline on a few sentences """
        docs = [Document([], text=t) for t in EN_DOCS]
        return pipeline(docs)

    def test_bulk_ents(self, processed_bulk):
        assert len(processed_bulk) == len(EXPECTED_ENTS)
        for doc, expected in zip(processed_bulk, EXPECTED_ENTS):
            check_entities_equal(doc, expected)

    def test_ents(self, processed_doc):
        assert len(processed_doc) == len(EXPECTED_ENTS)
        for doc, expected in zip(processed_doc, EXPECTED_ENTS):
            check_entities_equal(doc, expected)

EXPECTED_MULTI_ENTS = [{
  "text": "John Bauer",
  "type": "PERSON",
  "start_char": 0,
  "end_char": 10
}, {
  "text": "Stanford",
  "type": "ORG",
  "start_char": 20,
  "end_char": 28
}, {
  "text": "hip arthritis",
  "type": "DISEASE",
  "start_char": 37,
  "end_char": 50
}, {
  "text": "Chris Manning",
  "type": "PERSON",
  "start_char": 66,
  "end_char": 79
}]


EXPECTED_MULTI_NER = [
    [('O', 'B-PERSON'),
     ('O', 'E-PERSON'),
     ('O', 'O'),
     ('O', 'O'),
     ('O', 'S-ORG'),
     ('O', 'O'),
     ('O', 'O'),
     ('B-DISEASE', 'O'),
     ('E-DISEASE', 'O'),
     ('O', 'O')],
    [('O', 'O'),
     ('O', 'O'),
     ('O', 'O'),
     ('O', 'B-PERSON'),
     ('O', 'E-PERSON'),]]



class TestMultiNERProcessor:
    @pytest.fixture(scope="class")
    def pipeline(self):
        """
        A reusable pipeline with TWO ner models
        """
        return stanza.Pipeline(dir=TEST_MODELS_DIR, processors="tokenize,ner", package={"ner": ["ncbi_disease", "ontonotes"]})

    def test_multi_example(self, pipeline):
        doc = pipeline("John Bauer works at Stanford and has hip arthritis.  He works for Chris Manning")
        check_entities_equal(doc, EXPECTED_MULTI_ENTS)

    def test_multi_ner(self, pipeline):
        """
        Test that multiple NER labels are correctly assigned in tuples
        """
        doc = pipeline("John Bauer works at Stanford and has hip arthritis.  He works for Chris Manning")
        multi_ner = [[token.multi_ner for token in sentence.tokens] for sentence in doc.sentences]
        assert multi_ner == EXPECTED_MULTI_NER

    def test_known_tags(self, pipeline):
        assert pipeline.processors["ner"].get_known_tags() == ["DISEASE"]
        assert len(pipeline.processors["ner"].get_known_tags(1)) == 18
