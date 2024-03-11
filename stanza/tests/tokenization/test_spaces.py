"""
Test that when tokenizing a document, the Space annotations get set the way we expect
"""

import stanza
from stanza.tests import TEST_MODELS_DIR

EXPECTED_NO_MWT = """
# text = Jennifer has nice antennae.
# sent_id = 0
1	Jennifer	_	_	_	_	0	_	_	start_char=2|end_char=10|SpacesBefore=\\s\\s
2	has	_	_	_	_	1	_	_	start_char=11|end_char=14
3	nice	_	_	_	_	2	_	_	start_char=15|end_char=19
4	antennae	_	_	_	_	3	_	_	start_char=20|end_char=28|SpaceAfter=No
5	.	_	_	_	_	4	_	_	start_char=28|end_char=29|SpacesAfter=\\s\\s

# text = Not very nice person, though.
# sent_id = 1
1	Not	_	_	_	_	0	_	_	start_char=31|end_char=34
2	very	_	_	_	_	1	_	_	start_char=35|end_char=39
3	nice	_	_	_	_	2	_	_	start_char=40|end_char=44
4	person	_	_	_	_	3	_	_	start_char=45|end_char=51|SpaceAfter=No
5	,	_	_	_	_	4	_	_	start_char=51|end_char=52
6	though	_	_	_	_	5	_	_	start_char=53|end_char=59|SpaceAfter=No
7	.	_	_	_	_	6	_	_	start_char=59|end_char=60|SpacesAfter=\\s\\s
""".strip()

def test_spaces_no_mwt():
    """
    Test what happens if the words in a document have SpacesBefore and/or After
    """
    nlp = stanza.Pipeline(**{'processors': 'tokenize', 'download_method': None, 'dir': TEST_MODELS_DIR, 'lang': 'en'})
    doc = nlp("  Jennifer has nice antennae.  Not very nice person, though.  ")
    result = "{:C}".format(doc)
    result = result.strip()
    assert EXPECTED_NO_MWT == result

EXPECTED_MWT = """
# text = She's not a nice person.
# sent_id = 0
1-2	She's	_	_	_	_	_	_	_	start_char=2|end_char=7|SpacesBefore=\\s\\s
1	She	_	_	_	_	0	_	_	start_char=2|end_char=5
2	's	_	_	_	_	1	_	_	start_char=5|end_char=7
3	not	_	_	_	_	2	_	_	start_char=8|end_char=11
4	a	_	_	_	_	3	_	_	start_char=12|end_char=13
5	nice	_	_	_	_	4	_	_	start_char=14|end_char=18
6	person	_	_	_	_	5	_	_	start_char=19|end_char=25|SpaceAfter=No
7	.	_	_	_	_	6	_	_	start_char=25|end_char=26|SpacesAfter=\\s\\s

# text = However, the best antennae on the Cerritos are Jennifer's.
# sent_id = 1
1	However	_	_	_	_	0	_	_	start_char=28|end_char=35|SpaceAfter=No
2	,	_	_	_	_	1	_	_	start_char=35|end_char=36
3	the	_	_	_	_	2	_	_	start_char=37|end_char=40
4	best	_	_	_	_	3	_	_	start_char=41|end_char=45
5	antennae	_	_	_	_	4	_	_	start_char=46|end_char=54
6	on	_	_	_	_	5	_	_	start_char=55|end_char=57
7	the	_	_	_	_	6	_	_	start_char=58|end_char=61
8	Cerritos	_	_	_	_	7	_	_	start_char=62|end_char=70
9	are	_	_	_	_	8	_	_	start_char=71|end_char=74
10-11	Jennifer's	_	_	_	_	_	_	_	start_char=75|end_char=85|SpaceAfter=No
10	Jennifer	_	_	_	_	9	_	_	start_char=75|end_char=83
11	's	_	_	_	_	10	_	_	start_char=83|end_char=85
12	.	_	_	_	_	11	_	_	start_char=85|end_char=86|SpacesAfter=\\s\\s
""".strip()

def test_spaces_mwt():
    """
    Similar to the above test, but now we test it with MWT
    """
    nlp = stanza.Pipeline(**{'processors': 'tokenize', 'download_method': None, 'dir': TEST_MODELS_DIR, 'lang': 'en'})
    doc = nlp("  She's not a nice person.  However, the best antennae on the Cerritos are Jennifer's.  ")
    result = "{:C}".format(doc)
    result = result.strip()
    assert EXPECTED_MWT == result
