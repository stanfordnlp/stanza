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
    nlp = stanza.Pipeline(**{'processors': 'tokenize,lemma', 'dir': TEST_MODELS_DIR, 'lang': 'en', 'lemma_use_identity': True}, download_method=None)
    doc = nlp(EN_DOC)
    word_lemma_pairs = []
    for w in doc.iter_words():
        word_lemma_pairs += [f"{w.text} {w.lemma}"]
    assert EN_DOC_IDENTITY_GOLD == "\n".join(word_lemma_pairs)

def test_full_lemmatizer():
    nlp = stanza.Pipeline(**{'processors': 'tokenize,pos,lemma', 'dir': TEST_MODELS_DIR, 'lang': 'en'}, download_method=None)
    doc = nlp(EN_DOC)
    word_lemma_pairs = []
    for w in doc.iter_words():
        word_lemma_pairs += [f"{w.text} {w.lemma}"]
    assert EN_DOC_LEMMATIZER_MODEL_GOLD == "\n".join(word_lemma_pairs)

def find_unknown_word(lemmatizer, base):
    for i in range(10):
        # pos_dict: pos -> word -> lemma
        # make sure that none of the pos slices contain this word
        base = base + "z"
        if all(base not in x for x in lemmatizer.pos_dict):
            return base
    raise RuntimeError("wtf?")

def test_store_results():
    nlp = stanza.Pipeline(**{'processors': 'tokenize,pos,lemma', 'dir': TEST_MODELS_DIR, 'lang': 'en'}, lemma_store_results=True, download_method=None)
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

    assert lemmatizer.pos_dict[stuff[3][1]][az] == stuff[3][2]
    assert lemmatizer.pos_dict[stuff[6][1]][bz] == stuff[6][2]
    assert lemmatizer.pos_dict[stuff[11][1]][cz] == stuff[11][2]

    doc2 = nlp("I found an " + az + " in my " + bz + ".  It was a " + cz)
    stuff2 = doc2.get([TEXT, UPOS, LEMMA])

    assert stuff == stuff2

    dz = find_unknown_word(lemmatizer, "d")
    ez = find_unknown_word(lemmatizer, "e")
    fz = find_unknown_word(lemmatizer, "f")

    # try sentences with the order short, long
    doc = nlp("It was a " + dz + ".  I found an " + ez + " in my " + fz)
    stuff = doc.get([TEXT, UPOS, LEMMA])
    assert len(stuff) == 12
    assert stuff[3][0] == dz
    assert stuff[8][0] == ez
    assert stuff[11][0] == fz

    assert lemmatizer.pos_dict[stuff[3][1]][dz] == stuff[3][2]
    assert lemmatizer.pos_dict[stuff[8][1]][ez] == stuff[8][2]
    assert lemmatizer.pos_dict[stuff[11][1]][fz] == stuff[11][2]

    doc2 = nlp("It was a " + dz + ".  I found an " + ez + " in my " + fz)
    stuff2 = doc2.get([TEXT, UPOS, LEMMA])

    assert stuff == stuff2

    assert all(az not in x for x in lemmatizer.pos_dict)

def test_caseless_lemmatizer():
    """
    Test that setting the lemmatizer as caseless at Pipeline time lowercases the text
    """
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', model_dir=TEST_MODELS_DIR, download_method=None)
    # the capital letter here should throw off the lemmatizer & it won't remove the plural
    # although weirdly the current English model *does* lowercase the A
    doc = nlp("Here is an Excerpt")
    assert doc.sentences[0].words[-1].lemma == 'excerpt'

    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', model_dir=TEST_MODELS_DIR, download_method=None, lemma_caseless=True)
    # with the model set to lowercasing, the word will be treated as if it were 'antennae'
    doc = nlp("Here is an Excerpt")
    assert doc.sentences[0].words[-1].lemma == 'Excerpt'

def test_latin_caseless_lemmatizer():
    """
    Test the Latin caseless lemmatizer
    """
    nlp = stanza.Pipeline('la', package='ittb', processors='tokenize,pos,lemma', model_dir=TEST_MODELS_DIR, download_method=None)
    lemmatizer = nlp.processors['lemma']
    assert lemmatizer.config['caseless']

    doc = nlp("Quod Erat Demonstrandum")
    expected_lemmas = "qui sum demonstro".split()
    assert len(doc.sentences) == 1
    assert len(doc.sentences[0].words) == 3
    for word, expected in zip(doc.sentences[0].words, expected_lemmas):
        assert word.lemma == expected

def test_contextual_lemmatizer():
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', model_dir=TEST_MODELS_DIR, package={"lemma": "default_accurate"}, download_method=None)
    lemmatizer = nlp.processors['lemma']._trainer
    # the accurate model should have a 's classifier
    assert len(lemmatizer.contextual_lemmatizers) > 0
    doc = nlp("He's added a contextual lemmatizer")
    assert len(doc.sentences) == 1
    assert doc.sentences[0].words[1].text == "'s"
    assert doc.sentences[0].words[1].pos == "AUX"
    # this test should be simple enough that the
    # contextual classifier gets it right,
    # unless it gets retrained really badly
    assert doc.sentences[0].words[1].lemma == "have"

    doc = nlp("He's a little tired")
    assert len(doc.sentences) == 1
    assert doc.sentences[0].words[1].text == "'s"
    assert doc.sentences[0].words[1].pos == "AUX"
    # this test should be simple enough that the
    # contextual classifier gets it right,
    # unless it gets retrained really badly
    assert doc.sentences[0].words[1].lemma == "be"


def test_contextual_lemmatizer_results_are_not_cached():
    """lemma_store_results must not remember a word a contextual lemmatizer decides.

    The seq2seq prediction is made before update_contextual_preds runs, so one
    word, pos answer for 's would stand in for every occurrence: have in "He's
    added a lemmatizer" and be in "He's a little tired". The output itself is
    saved by the contextual pass running on every call, so this asserts on the
    dict rather than on the lemmas.
    """
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', model_dir=TEST_MODELS_DIR,
                          package={"lemma": "default_accurate"}, lemma_store_results=True,
                          download_method=None)
    trainer = nlp.processors['lemma']._trainer
    assert len(trainer.contextual_lemmatizers) > 0

    assert trainer.is_contextual_target("'s", "AUX")
    assert trainer.is_contextual_target("'S", "AUX")      # matched lowercased
    assert not trainer.is_contextual_target("'s", "PART")  # a different tag is not a target
    assert not trainer.is_contextual_target("dog", "NOUN")
    assert trainer.drop_contextual([("'s", "AUX", "be"), ("dog", "NOUN", "dog")]) == [("dog", "NOUN", "dog")]

    # the shipped dict already knows 's, so skip_seq2seq skips it before the cache
    # can see it; dropping those entries exercises the path the filter guards
    for pos_dict in trainer.pos_dict.values():
        pos_dict.pop("'s", None)
    assert trainer.skip_seq2seq([("'s", "AUX")]) == [False]

    doc = nlp("He's added a contextual lemmatizer")
    assert doc.sentences[0].words[1].lemma == "have"
    assert "'s" not in trainer.pos_dict.get("AUX", {})


def test_contextual_lemma_survives_the_results_cache():
    """With lemma_store_results on, the cache must never answer for a word the contextual
    lemmatizer decides, however many times that word has been through the lemmatizer.

    "her" is PRON in both senses, so the upos cannot separate them and only the contextual
    lemmatizer can: she when it is the object, her when it is the possessive. Alternating
    the two senses through one pipeline pins that behaviour.
    """
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', model_dir=TEST_MODELS_DIR,
                          package={"lemma": "default_accurate"}, lemma_store_results=True,
                          download_method=None)
    expected = [
        ("I saw her yesterday.", "she"),
        ("Her book is on the table.", "her"),
        ("They thanked her warmly.", "she"),
        ("Her car broke down.", "her"),
        ("We invited her to dinner.", "she"),
        ("Her sister called.", "her"),
    ]
    for text, lemma in expected:
        words = [word for word in nlp(text).sentences[0].words if word.text.lower() == "her"]
        assert len(words) == 1, text
        assert words[0].pos == "PRON", f"{text}: upos {words[0].pos}, expected PRON"
        assert words[0].lemma == lemma, f"{text}: lemma {words[0].lemma!r}, expected {lemma!r}"
