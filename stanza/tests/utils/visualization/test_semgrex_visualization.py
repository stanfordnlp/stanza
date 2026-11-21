"""
Tests for the semgrex visualizer which don't need CoreNLP

The semgrex results are built by hand, so only spacy is needed for
the displacy HTML
"""

import re

import pytest

pytest.importorskip("spacy")
pytest.importorskip("IPython")

from stanza.models.common.doc import Document
from stanza.protobuf import SemgrexResponse
from stanza.utils.conll import CoNLL
from stanza.utils.visualization.semgrex_visualizer import get_sentences_html, semgrexify_html

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

WORDS = "The quick brown fox jumps over the lazy dog near the old barn".split()

ENHANCED_SENTENCE = """
# text = Sue likes coffee and Bill tea.
1\tSue\tSue\tPROPN\tNNP\tNumber=Sing\t2\tnsubj\t2:nsubj\t_
2\tlikes\tlike\tVERB\tVBZ\tNumber=Sing|Person=3|Tense=Pres\t0\troot\t0:root\t_
3\tcoffee\tcoffee\tNOUN\tNN\tNumber=Sing\t2\tobj\t2:obj\t_
4\tand\tand\tCCONJ\tCC\t_\t5\tcc\t5.1:cc\t_
5\tBill\tBill\tPROPN\tNNP\tNumber=Sing\t2\tconj\t5.1:nsubj\t_
5.1\tlikes\tlike\tVERB\tVBZ\tNumber=Sing|Person=3|Tense=Pres\t_\t_\t2:conj:and\tCopyOf=2
6\ttea\ttea\tNOUN\tNN\tNumber=Sing\t5\torphan\t5.1:obj\tSpaceAfter=No
7\t.\t.\tPUNCT\t.\t_\t2\tpunct\t2:punct\t_
""".lstrip()

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


def test_empty_word_is_rendered_and_highlighted():
    doc = CoNLL.conll2doc(input_str=ENHANCED_SENTENCE, ignore_gapping=False)
    sentence_result = SemgrexResponse.SentenceResult()
    match = sentence_result.pattern.add().match.add()
    match.matchIndex = 5
    node = match.node.add()
    node.name = "copy"
    node.matchIndex = 5
    node.emptyIndex = 1

    html = get_sentences_html(doc, "en")[0]
    edited = semgrexify_html(html, sentence_result)
    real_word_start = edited.index('<text data-stanza-id="5"')
    empty_word_start = edited.index('<text data-stanza-id="5.1"')
    real_word = edited[real_word_start:edited.index("</text>", real_word_start)]
    empty_word = edited[empty_word_start:edited.index("</text>", empty_word_start)]

    assert "Bill" in real_word
    assert 'class="bolded"' not in real_word
    assert "likes" in empty_word
    assert 'class="bolded"' in empty_word
    assert ">Cop.</tspan>" in empty_word


def test_empty_word_keeps_its_position_in_right_to_left_rendering():
    doc = CoNLL.conll2doc(input_str=ENHANCED_SENTENCE, ignore_gapping=False)
    html = get_sentences_html(doc, "ar")[0]

    assert re.findall(r'data-stanza-id="([^"]+)"', html) == [
        "7",
        "6",
        "5.1",
        "5",
        "4",
        "3",
        "2",
        "1",
    ]
