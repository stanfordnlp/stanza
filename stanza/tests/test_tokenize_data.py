"""
Very simple test of the mwt counting functionality in tokenization/data.py

TODO: could add a bunch more simple tests, including tests of reading
the data from a temp file, for example
"""

import pytest
import stanza

from stanza.tests import *
from stanza.models.tokenization.data import DataLoader

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

# A single slice of the German tokenization data with no MWT in it
NO_MWT_DATA = [[('S', 0), ('e', 0), ('h', 0), ('r', 1), (' ', 0), ('g', 0), ('u', 0), ('t', 0), ('e', 1), (' ', 0), ('B', 0), ('e', 0), ('r', 0), ('a', 0), ('t', 0), ('u', 0), ('n', 0), ('g', 1), (',', 1), (' ', 0), ('s', 0), ('c', 0), ('h', 0), ('n', 0), ('e', 0), ('l', 0), ('l', 0), ('e', 1), (' ', 0), ('B', 0), ('e', 0), ('h', 0), ('e', 0), ('b', 0), ('u', 0), ('n', 0), ('g', 1), (' ', 0), ('d', 0), ('e', 0), ('r', 1), (' ', 0), ('P', 0), ('r', 0), ('o', 0), ('b', 0), ('l', 0), ('e', 0), ('m', 0), ('e', 2)]]

# A single slice of the German tokenization data with an MWT in it
MWT_DATA = [[(' ', 0), ('D', 0), ('i', 0), ('e', 1), (' ', 0), ('K', 0), ('o', 0), ('s', 0), ('t', 0), ('e', 0), ('n', 1), (' ', 0), ('s', 0), ('i', 0), ('n', 0), ('d', 1), (' ', 0), ('d', 0), ('e', 0), ('f', 0), ('i', 0), ('n', 0), ('i', 0), ('t', 0), ('i', 0), ('v', 1), (' ', 0), ('a', 0), ('u', 0), ('c', 0), ('h', 1), (' ', 0), ('i', 0), ('m', 3), (' ', 0), ('R', 0), ('a', 0), ('h', 0), ('m', 0), ('e', 0), ('n', 1), ('.', 2)]]

FAKE_PROPERTIES = {
    "lang":"de",
    'feat_funcs': ("space_before","capitalized"),
    'max_seqlen': 300,
}

def test_has_mwt():
    """
    One dataset has no mwt, the other does
    """
    data = DataLoader(args=FAKE_PROPERTIES, input_data=NO_MWT_DATA)
    assert not data.has_mwt()

    data = DataLoader(args=FAKE_PROPERTIES, input_data=MWT_DATA)
    assert data.has_mwt()

