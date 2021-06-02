"""
Basic testing of lemmatization
"""

import pytest
import stanza

from stanza.tests import *

pytestmark = pytest.mark.pipeline

EN_DOC = "Joe Smith was born in California."

EN_DOC_IDENTITY_GOLD = """
Joe Joe
Smith Smith
was was
born born
in in
California California
. .
""".strip()

EN_DOC_LEMMATIZER_MODEL_GOLD = """
Joe Joe
Smith Smith
was be
born bear
in in
California California
. .
""".strip()


def test_identity_lemmatizer():
    nlp = stanza.Pipeline(**{'processors': 'tokenize,lemma', 'dir': TEST_MODELS_DIR, 'lang': 'en',
                                  'lemma_use_identity': True})
    doc = nlp(EN_DOC)
    word_lemma_pairs = []
    for w in doc.iter_words():
        word_lemma_pairs += [f"{w.text} {w.lemma}"]
    assert EN_DOC_IDENTITY_GOLD == "\n".join(word_lemma_pairs)

def test_full_lemmatizer():
    nlp = stanza.Pipeline(**{'processors': 'tokenize,pos,lemma', 'dir': TEST_MODELS_DIR, 'lang': 'en'})
    doc = nlp(EN_DOC)
    word_lemma_pairs = []
    for w in doc.iter_words():
        word_lemma_pairs += [f"{w.text} {w.lemma}"]
    assert EN_DOC_LEMMATIZER_MODEL_GOLD == "\n".join(word_lemma_pairs)

