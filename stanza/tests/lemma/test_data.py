"""
Test a couple basic data functions, such as processing a doc for its lemmas
"""

import pytest

from stanza.models.common.doc import Document
from stanza.models.lemma.data import DataLoader
from stanza.utils.conll import CoNLL

TRAIN_DATA = """
# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0003
# text = DPA: Iraqi authorities announced that they had busted up 3 terrorist cells operating in Baghdad.
1	DPA	DPA	PROPN	NNP	Number=Sing	0	root	0:root	SpaceAfter=No
2	:	:	PUNCT	:	_	1	punct	1:punct	_
3	Iraqi	Iraqi	ADJ	JJ	Degree=Pos	4	amod	4:amod	_
4	authorities	authority	NOUN	NNS	Number=Plur	5	nsubj	5:nsubj	_
5	announced	announce	VERB	VBD	Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin	1	parataxis	1:parataxis	_
6	that	that	SCONJ	IN	_	9	mark	9:mark	_
7	they	they	PRON	PRP	Case=Nom|Number=Plur|Person=3|PronType=Prs	9	nsubj	9:nsubj	_
8	had	have	AUX	VBD	Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin	9	aux	9:aux	_
9	busted	bust	VERB	VBN	Tense=Past|VerbForm=Part	5	ccomp	5:ccomp	_
10	up	up	ADP	RP	_	9	compound:prt	9:compound:prt	_
11	3	3	NUM	CD	NumForm=Digit|NumType=Card	13	nummod	13:nummod	_
12	terrorist	terrorist	ADJ	JJ	Degree=Pos	13	amod	13:amod	_
13	cells	cell	NOUN	NNS	Number=Plur	9	obj	9:obj	_
14	operating	operate	VERB	VBG	VerbForm=Ger	13	acl	13:acl	_
15	in	in	ADP	IN	_	16	case	16:case	_
16	Baghdad	Baghdad	PROPN	NNP	Number=Sing	14	obl	14:obl:in	SpaceAfter=No
17	.	.	PUNCT	.	_	1	punct	1:punct	_

# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0004
# text = Two of them were being run by 2 officials of the Ministry of the Interior!
1	Two	two	NUM	CD	NumForm=Word|NumType=Card	6	nsubj:pass	6:nsubj:pass	_
2	of	of	ADP	IN	_	3	case	3:case	_
3	them	they	PRON	PRP	Case=Acc|Number=Plur|Person=3|PronType=Prs	1	nmod	1:nmod:of	_
4	were	be	AUX	VBD	Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin	6	aux	6:aux	_
5	being	be	AUX	VBG	VerbForm=Ger	6	aux:pass	6:aux:pass	_
6	run	run	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	0	root	0:root	_
7	by	by	ADP	IN	_	9	case	9:case	_
8	2	2	NUM	CD	NumForm=Digit|NumType=Card	9	nummod	9:nummod	_
9	officials	official	NOUN	NNS	Number=Plur	6	obl	6:obl:by	_
10	of	of	ADP	IN	_	12	case	12:case	_
11	the	the	DET	DT	Definite=Def|PronType=Art	12	det	12:det	_
12	Ministry	Ministry	PROPN	NNP	Number=Sing	9	nmod	9:nmod:of	_
13	of	of	ADP	IN	_	15	case	15:case	_
14	the	the	DET	DT	Definite=Def|PronType=Art	15	det	15:det	_
15	Interior	Interior	PROPN	NNP	Number=Sing	12	nmod	12:nmod:of	SpaceAfter=No
16	!	!	PUNCT	.	_	6	punct	6:punct	_

""".lstrip()

GOESWITH_DATA = """
# sent_id = email-enronsent27_01-0041
# newpar id = email-enronsent27_01-p0005
# text = Ken Rice@ENRON COMMUNICATIONS
1	Ken	kenrice@enroncommunications	X	GW	Typo=Yes	0	root	0:root	_
2	Rice@ENRON	_	X	GW	_	1	goeswith	1:goeswith	_
3	COMMUNICATIONS	_	X	ADD	_	1	goeswith	1:goeswith	_

""".lstrip()

CORRECT_FORM_DATA = """
# sent_id = weblog-blogspot.com_healingiraq_20040409053012_ENG_20040409_053012-0019
# text = They are targetting ambulances
1	They	they	PRON	PRP	Case=Nom|Number=Plur|Person=3|PronType=Prs	3	nsubj	3:nsubj	_
2	are	be	AUX	VBP	Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin	3	aux	3:aux	_
3	targetting	target	VERB	VBG	Tense=Pres|Typo=Yes|VerbForm=Part	0	root	0:root	CorrectForm=targeting
4	ambulances	ambulance	NOUN	NNS	Number=Plur	3	obj	3:obj	SpaceAfter=No
"""


def test_load_document():
    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA)
    data = DataLoader.load_doc(train_doc, caseless=False, evaluation=True)
    assert len(data) == 33 # meticulously counted by hand
    assert all(len(x) == 3 for x in data)

    data = DataLoader.load_doc(train_doc, caseless=False, evaluation=False)
    assert len(data) == 33
    assert all(len(x) == 3 for x in data)

def test_load_goeswith():
    raw_data = TRAIN_DATA + GOESWITH_DATA
    train_doc = CoNLL.conll2doc(input_str=raw_data)
    data = DataLoader.load_doc(train_doc, caseless=False, evaluation=True)
    assert len(data) == 36 # will be the same as in test_load_document with three additional words
    assert all(len(x) == 3 for x in data)

    data = DataLoader.load_doc(train_doc, caseless=False, evaluation=False)
    assert len(data) == 33 # will be the same as in test_load_document, but with the trailing 3 GOESWITH removed
    assert all(len(x) == 3 for x in data)

def test_correct_form():
    raw_data = TRAIN_DATA + CORRECT_FORM_DATA
    train_doc = CoNLL.conll2doc(input_str=raw_data)
    data = DataLoader.load_doc(train_doc, caseless=False, evaluation=True)
    assert len(data) == 37
    # the 'targeting' correction should not be applied if evaluation=True
    # when evaluation=False, then the CorrectForms will be applied
    assert not any(x[0] == 'targeting' for x in data)

    data = DataLoader.load_doc(train_doc, caseless=False, evaluation=False)
    assert len(data) == 38 # the same, but with an extra row so the model learns both 'targetting' and 'targeting'
    assert any(x[0] == 'targeting' for x in data)
    assert any(x[0] == 'targetting' for x in data)
