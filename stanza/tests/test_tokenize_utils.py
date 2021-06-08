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
