"""
Misc tests for the server
"""

import pytest
import re

from stanza.server import CoreNLPClient

pytestmark = pytest.mark.client

# test Italian pretokenized
italian_tokens = [
    "È vero , tutti possiamo essere sostituiti .\n Alcune chiamate partirono da il Quirinale ."
]
italian_tags = [
    [
        ["AUX", "ADJ", "PUNCT", "PRON", "AUX", "AUX", "VERB", "PUNCT"],
        ["DET", "NOUN", "VERB", "ADP", "DET", "PROPN", "PUNCT"],
    ],
]


def test_italian_pretokenized():
    """Test submitting pretokenized Italian text."""
    with CoreNLPClient(
        properties="italian",
        annotators="pos",
        pretokenized=True,
        be_quiet=True,
    ) as client:
        for input_text, gold_tags in zip(italian_tokens, italian_tags):
            ann = client.annotate(input_text)
            for sentence_tags, sentence in zip(gold_tags, ann.sentence):
                result_tags = [tok.pos for tok in sentence.token]
                assert sentence_tags == result_tags


# test French pretokenized
french_tokens = [
    "Les études durent six ans mais leur contenu diffère donc selon les Facultés ."
]
french_tags = [
    [
        [
            "DET",
            "NOUN",
            "VERB",
            "NUM",
            "NOUN",
            "CCONJ",
            "DET",
            "NOUN",
            "VERB",
            "ADV",
            "ADP",
            "DET",
            "PROPN",
            "PUNCT",
        ],
    ],
]


def test_french_pretokenized():
    """Test submitting pretokenized French text."""
    with CoreNLPClient(
        properties="french",
        annotators="pos",
        pretokenized=True,
        be_quiet=True,
    ) as client:
        for input_text, gold_tags in zip(french_tokens, french_tags):
            ann = client.annotate(input_text)
            for sentence_tags, sentence in zip(gold_tags, ann.sentence):
                result_tags = [tok.pos for tok in sentence.token]
                assert sentence_tags == result_tags
