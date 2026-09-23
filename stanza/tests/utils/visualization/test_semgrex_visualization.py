"""
Tests for the semgrex visualizer which don't need CoreNLP

The semgrex results are built by hand, so only spacy is needed for
the displacy HTML
"""

import pytest

pytest.importorskip("spacy")
pytest.importorskip("IPython")

from stanza.models.common.doc import Document
from stanza.protobuf import SemgrexResponse
from stanza.utils.visualization.semgrex_visualizer import get_sentences_html, semgrexify_html

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

WORDS = "The quick brown fox jumps over the lazy dog near the old barn".split()

def build_doc():
    sentence = []
    for idx, word in enumerate(WORDS):
        sentence.append({
            "id": idx + 1,
            "text": word,
            "lemma": word.lower(),
            "upos": "VERB" if word == "jumps" else "NOUN",
            "head": 0 if word == "jumps" else 5,
            "deprel": "root" if word == "jumps" else "dep",
        })
    return Document([sentence], " ".join(WORDS))

def build_sentence_result(num_matches):
    """
    One pattern which matched each of the first num_matches words, as {}=word would
    """
    sentence_result = SemgrexResponse.SentenceResult()
    pattern_result = sentence_result.pattern.add()
    for idx in range(num_matches):
        match = pattern_result.match.add()
        match.matchIndex = idx + 1
        node = match.node.add()
        node.name = "word"
        node.matchIndex = idx + 1
    return sentence_result

def test_more_matches_than_colors():
    """
    There are seven colors; more matches than that reuse them in order
    """
    html = get_sentences_html(build_doc(), "en")[0]
    edited = semgrexify_html(html, build_sentence_result(len(WORDS)))

    # each match gets its word bolded and a label
    assert edited.count('class="bolded"') == len(WORDS)
    assert edited.count(">Wor.</tspan>") == len(WORDS)
    # the eighth match is colored the same as the first
    assert edited.count('fill="#4477AA"') == 2 * 2
    assert edited.count('fill="#66CCEE"') == 2 * 2
