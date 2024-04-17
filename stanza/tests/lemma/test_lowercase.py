import pytest

from stanza.models.lemmatizer import all_lowercase
from stanza.utils.conll import CoNLL

LATIN_CONLLU = """
# sent_id = train-s1
# text = unde et philosophus dicit felicitatem esse operationem perfectam.
# reference = ittb-scg-s4203
1	unde	unde	ADV	O4	AdvType=Loc|PronType=Rel	4	advmod:lmod	_	_
2	et	et	CCONJ	O4	_	3	advmod:emph	_	_
3	philosophus	philosophus	NOUN	B1|grn1|casA|gen1	Case=Nom|Gender=Masc|InflClass=IndEurO|Number=Sing	4	nsubj	_	_
4	dicit	dico	VERB	N3|modA|tem1|gen6	Aspect=Imp|InflClass=LatX|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act	0	root	_	TraditionalMood=Indicativus|TraditionalTense=Praesens
5	felicitatem	felicitas	NOUN	C1|grn1|casD|gen2	Case=Acc|Gender=Fem|InflClass=IndEurX|Number=Sing	7	nsubj	_	_
6	esse	sum	AUX	N3|modH|tem1	Aspect=Imp|Tense=Pres|VerbForm=Inf	7	cop	_	_
7	operationem	operatio	NOUN	C1|grn1|casD|gen2|vgr1	Case=Acc|Gender=Fem|InflClass=IndEurX|Number=Sing	4	ccomp	_	_
8	perfectam	perfectus	ADJ	A1|grn1|casD|gen2	Case=Acc|Gender=Fem|InflClass=IndEurA|Number=Sing	7	amod	_	SpaceAfter=No
9	.	.	PUNCT	Punc	_	4	punct	_	_

# sent_id = train-s2
# text = perfectio autem operationis dependet ex quatuor.
# reference = ittb-scg-s4204
1	perfectio	perfectio	NOUN	C1|grn1|casA|gen2	Case=Nom|Gender=Fem|InflClass=IndEurX|Number=Sing	4	nsubj	_	_
2	autem	autem	PART	O4	_	4	discourse	_	_
3	operationis	operatio	NOUN	C1|grn1|casB|gen2|vgr1	Case=Gen|Gender=Fem|InflClass=IndEurX|Number=Sing	1	nmod	_	_
4	dependet	dependeo	VERB	K3|modA|tem1|gen6	Aspect=Imp|InflClass=LatE|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act	0	root	_	TraditionalMood=Indicativus|TraditionalTense=Praesens
5	ex	ex	ADP	S4|vgr2	_	6	case	_	_
6	quatuor	quattuor	NUM	G1|gen3|vgr1	NumForm=Word|NumType=Card	4	obl:arg	_	SpaceAfter=No
7	.	.	PUNCT	Punc	_	4	punct	_	_
""".lstrip()

ENG_CONLLU = """
# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0007
# text = You wonder if he was manipulating the market with his bombing targets.
1	You	you	PRON	PRP	Case=Nom|Person=2|PronType=Prs	2	nsubj	2:nsubj	_
2	wonder	wonder	VERB	VBP	Mood=Ind|Number=Sing|Person=2|Tense=Pres|VerbForm=Fin	0	root	0:root	_
3	if	if	SCONJ	IN	_	6	mark	6:mark	_
4	he	he	PRON	PRP	Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs	6	nsubj	6:nsubj	_
5	was	be	AUX	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	6	aux	6:aux	_
6	manipulating	manipulate	VERB	VBG	Tense=Pres|VerbForm=Part	2	ccomp	2:ccomp	_
7	the	the	DET	DT	Definite=Def|PronType=Art	8	det	8:det	_
8	market	market	NOUN	NN	Number=Sing	6	obj	6:obj	_
9	with	with	ADP	IN	_	12	case	12:case	_
10	his	his	PRON	PRP$	Case=Gen|Gender=Masc|Number=Sing|Person=3|Poss=Yes|PronType=Prs	12	nmod:poss	12:nmod:poss	_
11	bombing	bombing	NOUN	NN	Number=Sing	12	compound	12:compound	_
12	targets	target	NOUN	NNS	Number=Plur	6	obl	6:obl:with	SpaceAfter=No
13	.	.	PUNCT	.	_	2	punct	2:punct	_
""".lstrip()


def test_all_lowercase():
    doc = CoNLL.conll2doc(input_str=LATIN_CONLLU)
    assert all_lowercase(doc)

def test_not_all_lowercase():
    doc = CoNLL.conll2doc(input_str=ENG_CONLLU)
    assert not all_lowercase(doc)
