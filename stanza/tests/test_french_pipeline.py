"""
Basic testing of French pipeline

The benefit of this test is to verify that the bulk processing works
for languages with MWT in them
"""

import pytest
import stanza
from stanza.models.common.doc import Document

from stanza.tests import *

pytestmark = pytest.mark.pipeline


FR_MWT_SENTENCE = "Alors encore inconnu du grand public, Emmanuel Macron devient en 2014 ministre de l'Économie, de " \
                  "l'Industrie et du Numérique."

EXPECTED_RESULT = """
[
  [
    {
      "id": 1,
      "text": "Alors",
      "lemma": "alors",
      "upos": "ADV",
      "head": 3,
      "deprel": "advmod",
      "start_char": 0,
      "end_char": 5
    },
    {
      "id": 2,
      "text": "encore",
      "lemma": "encore",
      "upos": "ADV",
      "head": 3,
      "deprel": "advmod",
      "start_char": 6,
      "end_char": 12
    },
    {
      "id": 3,
      "text": "inconnu",
      "lemma": "inconnu",
      "upos": "ADJ",
      "feats": "Gender=Masc|Number=Sing",
      "head": 11,
      "deprel": "advcl",
      "start_char": 13,
      "end_char": 20
    },
    {
      "id": [
        4,
        5
      ],
      "text": "du",
      "start_char": 21,
      "end_char": 23
    },
    {
      "id": 4,
      "text": "de",
      "lemma": "de",
      "upos": "ADP",
      "head": 7,
      "deprel": "case"
    },
    {
      "id": 5,
      "text": "le",
      "lemma": "le",
      "upos": "DET",
      "feats": "Definite=Def|Gender=Masc|Number=Sing|PronType=Art",
      "head": 7,
      "deprel": "det"
    },
    {
      "id": 6,
      "text": "grand",
      "lemma": "grand",
      "upos": "ADJ",
      "feats": "Gender=Masc|Number=Sing",
      "head": 7,
      "deprel": "amod",
      "start_char": 24,
      "end_char": 29
    },
    {
      "id": 7,
      "text": "public",
      "lemma": "public",
      "upos": "NOUN",
      "feats": "Gender=Masc|Number=Sing",
      "head": 3,
      "deprel": "obl:arg",
      "start_char": 30,
      "end_char": 36
    },
    {
      "id": 8,
      "text": ",",
      "lemma": ",",
      "upos": "PUNCT",
      "head": 3,
      "deprel": "punct",
      "start_char": 36,
      "end_char": 37
    },
    {
      "id": 9,
      "text": "Emmanuel",
      "lemma": "Emmanuel",
      "upos": "PROPN",
      "head": 11,
      "deprel": "nsubj",
      "start_char": 38,
      "end_char": 46
    },
    {
      "id": 10,
      "text": "Macron",
      "lemma": "Macron",
      "upos": "PROPN",
      "head": 9,
      "deprel": "flat:name",
      "start_char": 47,
      "end_char": 53
    },
    {
      "id": 11,
      "text": "devient",
      "lemma": "devenir",
      "upos": "VERB",
      "feats": "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin",
      "head": 0,
      "deprel": "root",
      "start_char": 54,
      "end_char": 61
    },
    {
      "id": 12,
      "text": "en",
      "lemma": "en",
      "upos": "ADP",
      "head": 13,
      "deprel": "case",
      "start_char": 62,
      "end_char": 64
    },
    {
      "id": 13,
      "text": "2014",
      "lemma": "2014",
      "upos": "NUM",
      "feats": "Number=Plur",
      "head": 11,
      "deprel": "obl:mod",
      "start_char": 65,
      "end_char": 69
    },
    {
      "id": 14,
      "text": "ministre",
      "lemma": "ministre",
      "upos": "NOUN",
      "feats": "Gender=Masc|Number=Sing",
      "head": 11,
      "deprel": "xcomp:pred",
      "start_char": 70,
      "end_char": 78
    },
    {
      "id": 15,
      "text": "de",
      "lemma": "de",
      "upos": "ADP",
      "head": 17,
      "deprel": "case",
      "start_char": 79,
      "end_char": 81
    },
    {
      "id": 16,
      "text": "l'",
      "lemma": "le",
      "upos": "DET",
      "feats": "Definite=Def|Number=Sing|PronType=Art",
      "head": 17,
      "deprel": "det",
      "start_char": 82,
      "end_char": 84
    },
    {
      "id": 17,
      "text": "Économie",
      "lemma": "économie",
      "upos": "NOUN",
      "feats": "Gender=Fem|Number=Sing",
      "head": 14,
      "deprel": "nmod",
      "start_char": 84,
      "end_char": 92
    },
    {
      "id": 18,
      "text": ",",
      "lemma": ",",
      "upos": "PUNCT",
      "head": 21,
      "deprel": "punct",
      "start_char": 92,
      "end_char": 93
    },
    {
      "id": 19,
      "text": "de",
      "lemma": "de",
      "upos": "ADP",
      "head": 21,
      "deprel": "case",
      "start_char": 94,
      "end_char": 96
    },
    {
      "id": 20,
      "text": "l'",
      "lemma": "le",
      "upos": "DET",
      "feats": "Definite=Def|Number=Sing|PronType=Art",
      "head": 21,
      "deprel": "det",
      "start_char": 97,
      "end_char": 99
    },
    {
      "id": 21,
      "text": "Industrie",
      "lemma": "industrie",
      "upos": "NOUN",
      "feats": "Gender=Fem|Number=Sing",
      "head": 17,
      "deprel": "conj",
      "start_char": 99,
      "end_char": 108
    },
    {
      "id": 22,
      "text": "et",
      "lemma": "et",
      "upos": "CCONJ",
      "head": 25,
      "deprel": "cc",
      "start_char": 109,
      "end_char": 111
    },
    {
      "id": [
        23,
        24
      ],
      "text": "du",
      "start_char": 112,
      "end_char": 114
    },
    {
      "id": 23,
      "text": "de",
      "lemma": "de",
      "upos": "ADP",
      "head": 25,
      "deprel": "case"
    },
    {
      "id": 24,
      "text": "le",
      "lemma": "le",
      "upos": "DET",
      "feats": "Definite=Def|Gender=Masc|Number=Sing|PronType=Art",
      "head": 25,
      "deprel": "det"
    },
    {
      "id": 25,
      "text": "Numérique",
      "lemma": "numérique",
      "upos": "PROPN",
      "feats": "Gender=Masc|Number=Sing",
      "head": 17,
      "deprel": "conj",
      "start_char": 115,
      "end_char": 124
    },
    {
      "id": 26,
      "text": ".",
      "lemma": ".",
      "upos": "PUNCT",
      "head": 11,
      "deprel": "punct",
      "start_char": 124,
      "end_char": 125
    }
  ]
]
"""

@pytest.fixture(scope="module")
def pipeline():
    """ Document created by running full English pipeline on a few sentences """
    pipeline = stanza.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', dir=TEST_MODELS_DIR, lang='fr')
    return pipeline


def test_single(pipeline):
    doc = pipeline(FR_MWT_SENTENCE)
    compare_ignoring_whitespace(str(doc), EXPECTED_RESULT)
    
def test_bulk(pipeline):
    NUM_DOCS = 10
    raw_text = [FR_MWT_SENTENCE] * NUM_DOCS
    raw_doc = [Document([], text=doccontent) for doccontent in raw_text]
    
    result = pipeline(raw_doc)

    assert len(result) == NUM_DOCS
    for doc in result:
        compare_ignoring_whitespace(str(doc), EXPECTED_RESULT)
        assert len(doc.sentences) == 1
        assert doc.num_words == 26
        assert doc.num_tokens == 24


if __name__ == '__main__':
    pipeline = stanza.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', dir=TEST_MODELS_DIR, lang='fr')
    test_single(pipeline)
    test_bulk(pipeline)
