"""
Tests for LemmaClassifier base-class behaviour.

Uses a minimal stub subclass to exercise target_indices without needing
pretrained embeddings or a full LSTM/transformer stack.  The multi-word
case (a single classifier handling several surface forms, as in Greek
process_el_gdt) was previously untested.
"""

import pytest

from stanza.models.lemma_classifier.base_model import LemmaClassifier
from stanza.models.lemma_classifier.constants import ModelType

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]


# ---------------------------------------------------------------------------
# Minimal concrete subclass
# ---------------------------------------------------------------------------
# LemmaClassifier is abstract (requires model_type, forward, get_save_dict).
# We implement only what the base-class tests actually call.

class _StubClassifier(LemmaClassifier):
    """Concrete LemmaClassifier for testing base-class logic only."""

    def __init__(self, target_words, target_upos):
        super().__init__(
            label_decoder={0: "label"},
            target_words=set(target_words),
            target_upos=set(target_upos),
        )

    def model_type(self):
        return ModelType.LSTM

    def forward(self, *args, **kwargs):
        raise NotImplementedError("stub")

    def get_save_dict(self):
        raise NotImplementedError("stub")


# ---------------------------------------------------------------------------
# target_indices — single target word
# ---------------------------------------------------------------------------

def test_target_indices_single_word_found():
    clf = _StubClassifier(target_words=["'s"], target_upos=["AUX"])
    words = ["She", "'s", "happy", "."]
    tags  = ["PRON", "AUX", "ADJ", "PUNCT"]
    assert clf.target_indices(words, tags) == [1]


def test_target_indices_single_word_wrong_upos():
    """Word matches but UPOS does not — token must not be returned."""
    clf = _StubClassifier(target_words=["'s"], target_upos=["AUX"])
    words = ["She", "'s", "happy", "."]
    tags  = ["PRON", "NOUN", "ADJ", "PUNCT"]   # 's tagged NOUN, not AUX
    assert clf.target_indices(words, tags) == []


def test_target_indices_single_word_absent():
    clf = _StubClassifier(target_words=["'s"], target_upos=["AUX"])
    words = ["She", "is", "happy", "."]
    tags  = ["PRON", "AUX", "ADJ", "PUNCT"]
    assert clf.target_indices(words, tags) == []


def test_target_indices_case_insensitive():
    """target_indices lowercases the surface form before matching."""
    clf = _StubClassifier(target_words=["her"], target_upos=["PRON"])
    # "Her" capitalised at sentence start
    words = ["Her", "book", "was", "lost", "."]
    tags  = ["PRON", "NOUN", "AUX", "VERB", "PUNCT"]
    assert clf.target_indices(words, tags) == [0]


# ---------------------------------------------------------------------------
# target_indices — multiple target words (the Greek / EL case)
# ---------------------------------------------------------------------------

def test_target_indices_multiple_words_all_present():
    """
    A classifier covering several surface forms (like Greek τους|μας|του|...)
    must return the index of every matching token, not just the first.
    """
    target_words = ["τους", "μας", "του", "της", "σας", "μου"]
    clf = _StubClassifier(target_words=target_words, target_upos=["PRON"])

    words = ["τους", "other", "μας", "word", "του"]
    tags  = ["PRON", "NOUN", "PRON", "NOUN", "PRON"]
    assert clf.target_indices(words, tags) == [0, 2, 4]


def test_target_indices_multiple_words_partial_match():
    """Only words actually present in the sentence are returned."""
    clf = _StubClassifier(target_words=["τους", "μας", "του"], target_upos=["PRON"])
    words = ["μας", "είναι", "εδώ"]
    tags  = ["PRON", "AUX", "ADV"]
    assert clf.target_indices(words, tags) == [0]


def test_target_indices_multiple_words_none_present():
    clf = _StubClassifier(target_words=["τους", "μας", "του"], target_upos=["PRON"])
    words = ["αυτός", "είναι", "εδώ"]
    tags  = ["PRON", "AUX", "ADV"]
    assert clf.target_indices(words, tags) == []


def test_target_indices_multiple_words_wrong_upos():
    """Word matches but UPOS does not — no index should be returned."""
    clf = _StubClassifier(target_words=["τους", "μας"], target_upos=["PRON"])
    words = ["τους", "μας"]
    tags  = ["DET", "DET"]   # both DET, not PRON
    assert clf.target_indices(words, tags) == []


def test_target_indices_word_repeated_in_sentence():
    """The same target word appearing twice must yield two indices."""
    clf = _StubClassifier(target_words=["her"], target_upos=["PRON"])
    words = ["I", "gave", "her", "her", "book", "."]
    tags  = ["PRON", "VERB", "PRON", "PRON", "NOUN", "PUNCT"]
    assert clf.target_indices(words, tags) == [2, 3]
