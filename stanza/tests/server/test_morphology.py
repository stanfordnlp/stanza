"""
Test the most basic functionality of the morphology script
"""

import pytest

from stanza.server.morphology import Morphology, process_text

words    = ["Jennifer", "has",  "the", "prettiest", "antennae"]
tags     = ["NNP",      "VBZ",  "DT",  "JJS",       "NNS"]
expected = ["Jennifer", "have", "the", "pretty",    "antenna"]

def test_process_text():
    result = process_text(words, tags)
    lemma = [x.lemma for x in result.words]
    print(lemma)
    assert lemma == expected

def test_basic_morphology():
    with Morphology() as morph:
        result = morph.process(words, tags)
        lemma = [x.lemma for x in result.words]
        assert lemma == expected
