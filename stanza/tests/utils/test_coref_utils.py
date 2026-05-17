import pytest

from stanza.models.common.doc import Document
from stanza.models.coref.coref_chain import CorefChain, CorefMention
from stanza.utils.coref import coref_utils

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def build_doc_with_coref():
    """
    Build a minimal Document with two sentences and a simple coref chain.
    """
    sentences = [
        [
            {
                "id": 1,
                "text": "Barack",
                "lemma": "Barack",
                "upos": "PROPN",
                "xpos": "NNP",
                "feats": "_",
                "head": 2,
                "deprel": "compound",
                "misc": "start_char=0|end_char=6",
            },
            {
                "id": 2,
                "text": "Obama",
                "lemma": "Obama",
                "upos": "PROPN",
                "xpos": "NNP",
                "feats": "_",
                "head": 0,
                "deprel": "root",
                "misc": "start_char=7|end_char=12",
            },
        ],
        [
            {
                "id": 1,
                "text": "He",
                "lemma": "he",
                "upos": "PRON",
                "xpos": "PRP",
                "feats": "_",
                "head": 0,
                "deprel": "root",
                "misc": "start_char=0|end_char=2",
            }
        ],
    ]
    doc = Document(sentences, text="Barack Obama\nHe")

    # One chain: [Barack Obama] <-> [He]
    mentions = [
        CorefMention(sentence=0, start_word=0, end_word=2),
        CorefMention(sentence=1, start_word=0, end_word=1),
    ]
    chain = CorefChain(index=0, mentions=mentions, representative_text="Barack Obama", representative_index=0)
    doc.coref = [chain]
    return doc


def test_coref_chains_as_dicts():
    doc = build_doc_with_coref()
    chains = coref_utils.coref_chains_as_dicts(doc)

    assert len(chains) == 1
    chain = chains[0]
    assert chain["index"] == 0
    assert chain["representative"] == "Barack Obama"
    assert len(chain["mentions"]) == 2
    first, second = chain["mentions"]
    assert first["sentence"] == 0 and first["start"] == 0 and first["end"] == 2
    assert second["sentence"] == 1 and second["start"] == 0 and second["end"] == 1


def test_resolve_coref_default_strategy():
    doc = build_doc_with_coref()
    tokens = coref_utils.resolve_coref(doc)

    # The pronoun "He" should be replaced with the representative mention
    assert tokens == ["Barack", "Obama", "Barack Obama"]

