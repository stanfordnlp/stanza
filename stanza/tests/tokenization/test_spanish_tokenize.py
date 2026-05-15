"""
Tests for Spanish tokenization, including MWT handling and edge cases
that were previously reported as issues.

Each test case is a SpanishTokenizeCase namedtuple:
  text:            the raw input string
  sentences:       list of sentences; each sentence is a list of token strings
                   (for MWT tokens, use the *surface* form, e.g. "ocultándolo")
  words:           list of sentences; each sentence is a list of word strings
                   (MWT tokens are expanded, e.g. "ocultando", "lo")
  mwts:            list of (surface_token, [expanded_words]) for every MWT
                   across the whole document, in order

If `words` is None, it is assumed to equal `sentences` (no MWTs present).
If `mwts` is None, it is assumed to be empty.
"""

import pytest
from collections import namedtuple

import stanza
from stanza.tests import TEST_MODELS_DIR

pytestmark = pytest.mark.pipeline


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

SpanishTokenizeCase = namedtuple(
    "SpanishTokenizeCase",
    ["text", "sentences", "words", "mwts"],
    defaults=[None, None],        # words and mwts default to None
)

CASES = [
    SpanishTokenizeCase(
        text="[Daniel Alarcón]: Esto es Radio Ambulante, desde NPR. Soy Daniel Alarcón.",
        sentences=[
            ["[", "Daniel", "Alarcón", "]", ":", "Esto", "es", "Radio", "Ambulante", ",", "desde", "NPR", "."],
            ["Soy", "Daniel", "Alarcón", "."],
        ],
    ),
    SpanishTokenizeCase(
        text="Ronald ya no pudo seguir ocultándolo… se lo mostró a su mamá.",
        sentences=[
            ["Ronald", "ya", "no", "pudo", "seguir", "ocultándolo", "…", "se", "lo", "mostró", "a", "su", "mamá", "."],
        ],
        words=[
            ["Ronald", "ya", "no", "pudo", "seguir", "ocultando", "lo", "…", "se", "lo", "mostró", "a", "su", "mamá", "."],
        ],
        mwts=[
            ("ocultándolo", ["ocultando", "lo"]),
        ],
    ),
    SpanishTokenizeCase(
        text="Y felices no estaban",
        sentences=[
            ["Y", "felices", "no", "estaban"],
        ],
    ),
    SpanishTokenizeCase(
        text="Y felices no estaban…",
        sentences=[
            ["Y", "felices", "no", "estaban", "…"],
        ],
    ),
]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def es_tokenize_pipeline():
    """Tokenize-only Spanish pipeline (no MWT, pos, etc.)."""
    return stanza.Pipeline(
        "es",
        dir=TEST_MODELS_DIR,
        download_method=None,
        processors="tokenize",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(pipeline, case):
    """Return (doc, resolved_words, resolved_mwts) for a test case."""
    doc = pipeline(case.text)
    resolved_words = case.words if case.words is not None else case.sentences
    resolved_mwts  = case.mwts  if case.mwts  is not None else []
    return doc, resolved_words, resolved_mwts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=[c.text[:40] for c in CASES])
class TestSpanishTokenize:

    def test_sentence_count(self, es_tokenize_pipeline, case):
        doc, _, _ = _run(es_tokenize_pipeline, case)
        assert len(doc.sentences) == len(case.sentences), (
            f"Expected {len(case.sentences)} sentence(s), "
            f"got {len(doc.sentences)}: {[s.text for s in doc.sentences]}"
        )

    def test_token_texts(self, es_tokenize_pipeline, case):
        """Surface token texts match for every sentence."""
        doc, _, _ = _run(es_tokenize_pipeline, case)
        for sent_idx, (sentence, expected_tokens) in enumerate(
            zip(doc.sentences, case.sentences)
        ):
            actual = [token.text for token in sentence.tokens]
            assert actual == expected_tokens, (
                f"Sentence {sent_idx}: token mismatch"
            )

    def test_word_texts(self, es_tokenize_pipeline, case):
        """Word texts (after MWT expansion) match for every sentence."""
        doc, resolved_words, _ = _run(es_tokenize_pipeline, case)
        for sent_idx, (sentence, expected_words) in enumerate(
            zip(doc.sentences, resolved_words)
        ):
            actual = [word.text for word in sentence.words]
            assert actual == expected_words, (
                f"Sentence {sent_idx}: word mismatch"
            )

    def test_mwts(self, es_tokenize_pipeline, case):
        """MWT tokens expand to the expected word sequences."""
        doc, _, resolved_mwts = _run(es_tokenize_pipeline, case)
        actual_mwts = [
            (token.text, [w.text for w in token.words])
            for sentence in doc.sentences
            for token in sentence.tokens
            if len(token.words) > 1
        ]
        assert actual_mwts == resolved_mwts
