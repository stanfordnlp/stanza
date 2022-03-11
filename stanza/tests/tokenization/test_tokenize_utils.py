"""
Very simple test of the sentence slicing by <PAD> tags

TODO: could add a bunch more simple tests for the tokenization utils
"""

import pytest
import stanza

from stanza.tests import *
from stanza.models.tokenization import utils

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def test_find_spans():
    """
    Test various raw -> span manipulations
    """
    raw = ['u', 'n', 'b', 'a', 'n', ' ', 'm', 'o', 'x', ' ', 'o', 'p', 'a', 'l']
    assert utils.find_spans(raw) == [(0, 14)]

    raw = ['u', 'n', 'b', 'a', 'n', ' ', 'm', 'o', 'x', ' ', 'o', 'p', 'a', 'l', '<PAD>']
    assert utils.find_spans(raw) == [(0, 14)]

    raw = ['<PAD>', 'u', 'n', 'b', 'a', 'n', ' ', 'm', 'o', 'x', ' ', 'o', 'p', 'a', 'l', '<PAD>']
    assert utils.find_spans(raw) == [(1, 15)]

    raw = ['<PAD>', 'u', 'n', 'b', 'a', 'n', ' ', 'm', 'o', 'x', ' ', 'o', 'p', 'a', 'l']
    assert utils.find_spans(raw) == [(1, 15)]

    raw = ['<PAD>', 'u', 'n', 'b', 'a', 'n', '<PAD>', 'm', 'o', 'x', ' ', 'o', 'p', 'a', 'l']
    assert utils.find_spans(raw) == [(1, 6), (7, 15)]

def check_offsets(doc, expected_offsets):
    """
    Compare the start_char and end_char of the tokens in the doc with the given list of list of offsets
    """
    assert len(doc.sentences) == len(expected_offsets)
    for sentence, offsets in zip(doc.sentences, expected_offsets):
        assert len(sentence.tokens) == len(offsets)
        for token, offset in zip(sentence.tokens, offsets):
            assert token.start_char == offset[0]
            assert token.end_char == offset[1]

def test_match_tokens_with_text():
    """
    Test the conversion of pretokenized text to Document
    """
    doc = utils.match_tokens_with_text([["This", "is", "a", "test"]], "Thisisatest")
    expected_offsets = [[(0, 4), (4, 6), (6, 7), (7, 11)]]
    check_offsets(doc, expected_offsets)

    doc = utils.match_tokens_with_text([["This", "is", "a", "test"], ["unban", "mox", "opal", "!"]], "Thisisatest  unban mox  opal!")
    expected_offsets = [[(0, 4), (4, 6), (6, 7), (7, 11)],
                        [(13, 18), (19, 22), (24, 28), (28, 29)]]
    check_offsets(doc, expected_offsets)

    with pytest.raises(ValueError):
        doc = utils.match_tokens_with_text([["This", "is", "a", "test"]], "Thisisatestttt")

    with pytest.raises(ValueError):
        doc = utils.match_tokens_with_text([["This", "is", "a", "test"]], "Thisisates")

    with pytest.raises(ValueError):
        doc = utils.match_tokens_with_text([["This", "iz", "a", "test"]], "Thisisatest")
