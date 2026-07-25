"""
Basic testing of lemmatization
"""

from typing import Mapping, Optional, Protocol

import pytest
import stanza

from stanza.models.common.doc import Document
from stanza.pipeline.core import Pipeline
from stanza.pipeline.lemma_processor import LemmaProcessor
from stanza.tests import TEST_MODELS_DIR

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


class _TestLemmaTrainer(Protocol):
    @property
    def pos_dict(
            self,
        ) -> Mapping[Optional[str], Mapping[str, str]]:
        ...

    def has_contextual_lemmatizers(self) -> bool:
        ...


def _process_text(pipeline: Pipeline, text: str) -> Document:
    document = pipeline(text)
    if not isinstance(document, Document):
        raise TypeError("A text pipeline must return a Document")
    return document


def _lemma_trainer(pipeline: Pipeline) -> _TestLemmaTrainer:
    processor = pipeline.processors["lemma"]
    if not isinstance(processor, LemmaProcessor):
        raise TypeError("The pipeline did not load a LemmaProcessor")
    return processor._require_trainer()


def _word_annotations(
        document: Document,
    ) -> list[tuple[str, Optional[str], Optional[str]]]:
    return [
        (word.text, word.upos, word.lemma)
        for word in document.iter_words()
    ]


def test_identity_lemmatizer() -> None:
    nlp = stanza.Pipeline(
        processors='tokenize,lemma',
        dir=TEST_MODELS_DIR,
        lang='en',
        lemma_use_identity=True,
        download_method=None,
    )
    document = _process_text(nlp, EN_DOC)
    word_lemma_pairs: list[str] = []
    for w in document.iter_words():
        word_lemma_pairs += [f"{w.text} {w.lemma}"]
    assert EN_DOC_IDENTITY_GOLD == "\n".join(word_lemma_pairs)

def test_full_lemmatizer() -> None:
    nlp = stanza.Pipeline(
        processors='tokenize,pos,lemma',
        dir=TEST_MODELS_DIR,
        lang='en',
        download_method=None,
    )
    document = _process_text(nlp, EN_DOC)
    word_lemma_pairs: list[str] = []
    for w in document.iter_words():
        word_lemma_pairs += [f"{w.text} {w.lemma}"]
    assert EN_DOC_LEMMATIZER_MODEL_GOLD == "\n".join(word_lemma_pairs)

def find_unknown_word(
        lemmatizer: _TestLemmaTrainer,
        base: str,
    ) -> str:
    for _ in range(10):
        # pos_dict: pos -> word -> lemma
        # make sure that none of the pos slices contain this word
        base = base + "z"
        if all(base not in entries for entries in lemmatizer.pos_dict.values()):
            return base
    raise RuntimeError("wtf?")

def test_store_results() -> None:
    nlp = stanza.Pipeline(
        processors='tokenize,pos,lemma',
        dir=TEST_MODELS_DIR,
        lang='en',
        lemma_store_results=True,
        download_method=None,
    )
    lemmatizer = _lemma_trainer(nlp)

    az = find_unknown_word(lemmatizer, "a")
    bz = find_unknown_word(lemmatizer, "b")
    cz = find_unknown_word(lemmatizer, "c")

    # try sentences with the order long, short
    document = _process_text(
        nlp,
        "I found an " + az + " in my " + bz + ".  It was a " + cz,
    )
    stuff = _word_annotations(document)
    assert len(stuff) == 12
    assert stuff[3][0] == az
    assert stuff[6][0] == bz
    assert stuff[11][0] == cz

    assert lemmatizer.pos_dict[stuff[3][1]][az] == stuff[3][2]
    assert lemmatizer.pos_dict[stuff[6][1]][bz] == stuff[6][2]
    assert lemmatizer.pos_dict[stuff[11][1]][cz] == stuff[11][2]

    second_document = _process_text(
        nlp,
        "I found an " + az + " in my " + bz + ".  It was a " + cz,
    )
    stuff2 = _word_annotations(second_document)

    assert stuff == stuff2

    dz = find_unknown_word(lemmatizer, "d")
    ez = find_unknown_word(lemmatizer, "e")
    fz = find_unknown_word(lemmatizer, "f")

    # try sentences with the order short, long
    document = _process_text(
        nlp,
        "It was a " + dz + ".  I found an " + ez + " in my " + fz,
    )
    stuff = _word_annotations(document)
    assert len(stuff) == 12
    assert stuff[3][0] == dz
    assert stuff[8][0] == ez
    assert stuff[11][0] == fz

    assert lemmatizer.pos_dict[stuff[3][1]][dz] == stuff[3][2]
    assert lemmatizer.pos_dict[stuff[8][1]][ez] == stuff[8][2]
    assert lemmatizer.pos_dict[stuff[11][1]][fz] == stuff[11][2]

    second_document = _process_text(
        nlp,
        "It was a " + dz + ".  I found an " + ez + " in my " + fz,
    )
    stuff2 = _word_annotations(second_document)

    assert stuff == stuff2

    # Runtime updates should stay in the POS-specific dictionary instead of
    # leaking into the POS-independent fallback.
    assert az not in lemmatizer.pos_dict.get("*", {})

def test_caseless_lemmatizer() -> None:
    """
    Test that setting the lemmatizer as caseless at Pipeline time lowercases the text
    """
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', model_dir=TEST_MODELS_DIR, download_method=None)
    # the capital letter here should throw off the lemmatizer & it won't remove the plural
    # although weirdly the current English model *does* lowercase the A
    document = _process_text(nlp, "Here is an Excerpt")
    assert document.sentences[0].words[-1].lemma == 'excerpt'

    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', model_dir=TEST_MODELS_DIR, download_method=None, lemma_caseless=True)
    # with the model set to lowercasing, the word will be treated as if it were 'antennae'
    document = _process_text(nlp, "Here is an Excerpt")
    assert document.sentences[0].words[-1].lemma == 'Excerpt'

def test_latin_caseless_lemmatizer() -> None:
    """
    Test the Latin caseless lemmatizer
    """
    nlp = stanza.Pipeline('la', package='ittb', processors='tokenize,pos,lemma', model_dir=TEST_MODELS_DIR, download_method=None)
    lemmatizer = nlp.processors['lemma']
    assert isinstance(lemmatizer, LemmaProcessor)
    assert lemmatizer.config.get('caseless')

    document = _process_text(nlp, "Quod Erat Demonstrandum")
    expected_lemmas = "qui sum demonstro".split()
    assert len(document.sentences) == 1
    assert len(document.sentences[0].words) == 3
    for word, expected in zip(
            document.sentences[0].words,
            expected_lemmas,
        ):
        assert word.lemma == expected

def test_contextual_lemmatizer() -> None:
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', model_dir=TEST_MODELS_DIR, package={"lemma": "default_accurate"}, download_method=None)
    lemmatizer = _lemma_trainer(nlp)
    # the accurate model should have a 's classifier
    assert lemmatizer.has_contextual_lemmatizers()
    document = _process_text(nlp, "He's added a contextual lemmatizer")
    assert len(document.sentences) == 1
    assert document.sentences[0].words[1].text == "'s"
    assert document.sentences[0].words[1].pos == "AUX"
    # this test should be simple enough that the
    # contextual classifier gets it right,
    # unless it gets retrained really badly
    assert document.sentences[0].words[1].lemma == "have"

    document = _process_text(nlp, "He's a little tired")
    assert len(document.sentences) == 1
    assert document.sentences[0].words[1].text == "'s"
    assert document.sentences[0].words[1].pos == "AUX"
    # this test should be simple enough that the
    # contextual classifier gets it right,
    # unless it gets retrained really badly
    assert document.sentences[0].words[1].lemma == "be"
