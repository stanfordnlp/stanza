"""
Basic testing of lemmatization
"""

import pytest
import stanza

from stanza.tests import *
from stanza.models.common.doc import TEXT, UPOS, LEMMA

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

def find_unknown_word(lemmatizer, base):
    for i in range(10):
        base = base + "z"
        if base not in lemmatizer.word_dict and all(x[0] != base for x in lemmatizer.composite_dict.keys()):
            return base
    raise RuntimeError("wtf?")

def test_store_results():
    nlp = stanza.Pipeline(**{'processors': 'tokenize,pos,lemma', 'dir': TEST_MODELS_DIR, 'lang': 'en'}, lemma_store_results=True)
    lemmatizer = nlp.processors["lemma"]._trainer

    az = find_unknown_word(lemmatizer, "a")
    bz = find_unknown_word(lemmatizer, "b")
    cz = find_unknown_word(lemmatizer, "c")

    # try sentences with the order long, short
    doc = nlp("I found an " + az + " in my " + bz + ".  It was a " + cz)
    stuff = doc.get([TEXT, UPOS, LEMMA])
    assert len(stuff) == 12
    assert stuff[3][0] == az
    assert stuff[6][0] == bz
    assert stuff[11][0] == cz

    assert lemmatizer.composite_dict[(az, stuff[3][1])] == stuff[3][2]
    assert lemmatizer.composite_dict[(bz, stuff[6][1])] == stuff[6][2]
    assert lemmatizer.composite_dict[(cz, stuff[11][1])] == stuff[11][2]

    doc2 = nlp("I found an " + az + " in my " + bz + ".  It was a " + cz)
    stuff2 = doc2.get([TEXT, UPOS, LEMMA])

    assert stuff == stuff2

    dz = find_unknown_word(lemmatizer, "d")
    ez = find_unknown_word(lemmatizer, "e")
    fz = find_unknown_word(lemmatizer, "f")

    # try sentences with the order long, short
    doc = nlp("It was a " + dz + ".  I found an " + ez + " in my " + fz)
    stuff = doc.get([TEXT, UPOS, LEMMA])
    assert len(stuff) == 12
    assert stuff[3][0] == dz
    assert stuff[8][0] == ez
    assert stuff[11][0] == fz

    assert lemmatizer.composite_dict[(dz, stuff[3][1])] == stuff[3][2]
    assert lemmatizer.composite_dict[(ez, stuff[8][1])] == stuff[8][2]
    assert lemmatizer.composite_dict[(fz, stuff[11][1])] == stuff[11][2]

    doc2 = nlp("It was a " + dz + ".  I found an " + ez + " in my " + fz)
    stuff2 = doc2.get([TEXT, UPOS, LEMMA])

    assert stuff == stuff2

    assert az not in lemmatizer.word_dict
