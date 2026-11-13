"""
Test that the text of a coref mention is rebuilt using the whitespace on the tokens
"""

import pytest

from stanza.pipeline.coref_processor import extract_text
from stanza.utils.conll import CoNLL

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

CONLL = """
# text = I like the U.S.,  said Bob.
1	I	I	PRON	_	_	0	_	_	_
2	like	like	VERB	_	_	0	_	_	_
3	the	the	DET	_	_	0	_	_	_
4	U.S.	U.S.	PROPN	_	_	0	_	_	SpaceAfter=No
5	,	,	PUNCT	_	_	0	_	_	SpacesAfter=\\s\\s
6	said	say	VERB	_	_	0	_	_	_
7	Bob	Bob	PROPN	_	_	0	_	_	SpaceAfter=No
8	.	.	PUNCT	_	_	0	_	_	_

# text = Vámonos del parque.
1-2	Vámonos	_	_	_	_	_	_	_	_
1	Vamos	ir	VERB	_	_	0	_	_	_
2	nos	nos	PRON	_	_	0	_	_	_
3-4	del	_	_	_	_	_	_	_	_
3	de	de	ADP	_	_	0	_	_	_
4	el	el	DET	_	_	0	_	_	_
5	parque	parque	NOUN	_	_	0	_	_	SpaceAfter=No
6	.	.	PUNCT	_	_	0	_	_	_

# text = Dámelo ahora.
1-3	Dámelo	_	_	_	_	_	_	_	_
1	Da	dar	VERB	_	_	0	_	_	_
2	me	yo	PRON	_	_	0	_	_	_
3	lo	él	PRON	_	_	0	_	_	_
4	ahora	ahora	ADV	_	_	0	_	_	SpaceAfter=No
5	.	.	PUNCT	_	_	0	_	_	_
""".strip() + "\n\n"

# (sentence, 0-based start word, exclusive end word, expected text)
EXPECTED = [
    (0, 2, 4, "the U.S."),
    (0, 2, 5, "the U.S.,"),
    (0, 3, 8, "U.S.,  said Bob."),
    (0, 6, 8, "Bob."),
    # whole MWT inside the span uses the token text
    (1, 0, 2, "Vámonos"),
    (1, 2, 6, "del parque."),
    # partial MWT: the last word of the MWT gets the token's whitespace
    (1, 3, 6, "el parque."),
    (1, 2, 3, "de"),
    # partial MWT: a word in the middle of an MWT is followed by no space
    (2, 1, 5, "melo ahora."),
]

@pytest.fixture(scope="module")
def doc():
    return CoNLL.conll2doc(input_str=CONLL)

@pytest.mark.parametrize("sent_id, start_word, end_word, expected", EXPECTED)
def test_extract_text(doc, sent_id, start_word, end_word, expected):
    assert extract_text(doc, sent_id, start_word, end_word) == expected
