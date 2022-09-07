"""
Test conllu manipulating routines in stanza/utils/dataset/common.py
"""

import pytest


from stanza.utils.datasets.common import maybe_add_fake_dependencies
# from stanza.tests import *

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

DEPS_EXAMPLE="""
# text = Sh'reyan's antennae are hella thicc
1	Sh'reyan	Sh'reyan	PROPN	NNP	Number=Sing	3	nmod:poss	3:nmod:poss	SpaceAfter=No
2	's	's	PART	POS	_	1	case	1:case	_
3	antennae	antenna	NOUN	NNS	Number=Plur	6	nsubj	6:nsubj	_
4	are	be	VERB	VBP	Mood=Ind|Tense=Pres|VerbForm=Fin	6	cop	6:cop	_
5	hella	hella	ADV	RB	_	6	advmod	6:advmod	_
6	thicc	thicc	ADJ	JJ	Degree=Pos	0	root	0:root	_
""".strip().split("\n")


ONLY_ROOT_EXAMPLE="""
# text = Sh'reyan's antennae are hella thicc
1	Sh'reyan	Sh'reyan	PROPN	NNP	Number=Sing	_	_	_	SpaceAfter=No
2	's	's	PART	POS	_	_	_	_	_
3	antennae	antenna	NOUN	NNS	Number=Plur	_	_	_	_
4	are	be	VERB	VBP	Mood=Ind|Tense=Pres|VerbForm=Fin	_	_	_	_
5	hella	hella	ADV	RB	_	_	_	_	_
6	thicc	thicc	ADJ	JJ	Degree=Pos	0	root	0:root	_
""".strip().split("\n")

ONLY_ROOT_EXPECTED="""
# text = Sh'reyan's antennae are hella thicc
1	Sh'reyan	Sh'reyan	PROPN	NNP	Number=Sing	6	dep	_	SpaceAfter=No
2	's	's	PART	POS	_	1	dep	_	_
3	antennae	antenna	NOUN	NNS	Number=Plur	1	dep	_	_
4	are	be	VERB	VBP	Mood=Ind|Tense=Pres|VerbForm=Fin	1	dep	_	_
5	hella	hella	ADV	RB	_	1	dep	_	_
6	thicc	thicc	ADJ	JJ	Degree=Pos	0	root	0:root	_
""".strip().split("\n")

NO_DEPS_EXAMPLE="""
# text = Sh'reyan's antennae are hella thicc
1	Sh'reyan	Sh'reyan	PROPN	NNP	Number=Sing	_	_	_	SpaceAfter=No
2	's	's	PART	POS	_	_	_	_	_
3	antennae	antenna	NOUN	NNS	Number=Plur	_	_	_	_
4	are	be	VERB	VBP	Mood=Ind|Tense=Pres|VerbForm=Fin	_	_	_	_
5	hella	hella	ADV	RB	_	_	_	_	_
6	thicc	thicc	ADJ	JJ	Degree=Pos	_	_	_	_
""".strip().split("\n")

NO_DEPS_EXPECTED="""
# text = Sh'reyan's antennae are hella thicc
1	Sh'reyan	Sh'reyan	PROPN	NNP	Number=Sing	0	root	_	SpaceAfter=No
2	's	's	PART	POS	_	1	dep	_	_
3	antennae	antenna	NOUN	NNS	Number=Plur	1	dep	_	_
4	are	be	VERB	VBP	Mood=Ind|Tense=Pres|VerbForm=Fin	1	dep	_	_
5	hella	hella	ADV	RB	_	1	dep	_	_
6	thicc	thicc	ADJ	JJ	Degree=Pos	1	dep	_	_
""".strip().split("\n")


def test_fake_deps_no_change():
    result = maybe_add_fake_dependencies(DEPS_EXAMPLE)
    assert result == DEPS_EXAMPLE

def test_fake_deps_all_tokens():
    result = maybe_add_fake_dependencies(NO_DEPS_EXAMPLE)
    assert result == NO_DEPS_EXPECTED


def test_fake_deps_only_root():
    result = maybe_add_fake_dependencies(ONLY_ROOT_EXAMPLE)
    assert result == ONLY_ROOT_EXPECTED
