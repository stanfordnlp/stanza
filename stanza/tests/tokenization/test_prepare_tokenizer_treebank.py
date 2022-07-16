import pytest
import stanza
from stanza.tests import *

from stanza.utils.datasets import prepare_tokenizer_treebank

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def test_has_space_after_no():
    assert prepare_tokenizer_treebank.has_space_after_no("SpaceAfter=No")
    assert prepare_tokenizer_treebank.has_space_after_no("UnbanMoxOpal=Yes|SpaceAfter=No")
    assert prepare_tokenizer_treebank.has_space_after_no("SpaceAfter=No|UnbanMoxOpal=Yes")
    assert not prepare_tokenizer_treebank.has_space_after_no("SpaceAfter=Yes")
    assert not prepare_tokenizer_treebank.has_space_after_no("CorrectSpaceAfter=No")
    assert not prepare_tokenizer_treebank.has_space_after_no("_")


def test_add_space_after_no():
    assert prepare_tokenizer_treebank.add_space_after_no("_") == "SpaceAfter=No"
    assert prepare_tokenizer_treebank.add_space_after_no("MoxOpal=Unban") == "MoxOpal=Unban|SpaceAfter=No"
    with pytest.raises(ValueError):
        prepare_tokenizer_treebank.add_space_after_no("SpaceAfter=No")

def test_remove_space_after_no():
    assert prepare_tokenizer_treebank.remove_space_after_no("SpaceAfter=No") == "_"
    assert prepare_tokenizer_treebank.remove_space_after_no("SpaceAfter=No|MoxOpal=Unban") == "MoxOpal=Unban"
    assert prepare_tokenizer_treebank.remove_space_after_no("MoxOpal=Unban|SpaceAfter=No") == "MoxOpal=Unban"
    with pytest.raises(ValueError):
        prepare_tokenizer_treebank.remove_space_after_no("_")

def read_test_doc(doc):
    sentences = [x.strip().split("\n") for x in doc.split("\n\n")]
    return sentences


SPANISH_QM_TEST_CASE = """
# sent_id = train-s7914
# text = ¿Cómo explicarles entonces que el mar tiene varios dueños y que a partir de la frontera de aquella ola el pescado ya no es tuyo?.
# orig_file_sentence 080#14
# this sentence will have the intiial ¿ removed.  an MWT should be preserved
1	¿	¿	PUNCT	_	PunctSide=Ini|PunctType=Qest	3	punct	_	SpaceAfter=No
2	Cómo	cómo	PRON	_	PronType=Ind	3	obl	_	_
3-4	explicarles	_	_	_	_	_	_	_	_
3	explicar	explicar	VERB	_	VerbForm=Inf	0	root	_	_
4	les	él	PRON	_	Case=Dat|Number=Plur|Person=3|PronType=Prs	3	obj	_	_
5	entonces	entonces	ADV	_	_	3	advmod	_	_
6	que	que	SCONJ	_	_	9	mark	_	_
7	el	el	DET	_	Definite=Def|Gender=Masc|Number=Sing|PronType=Art	8	det	_	_
8	mar	mar	NOUN	_	Number=Sing	9	nsubj	_	_
9	tiene	tener	VERB	_	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	3	ccomp	_	_
10	varios	varios	DET	_	Gender=Masc|Number=Plur|PronType=Ind	11	det	_	_
11	dueños	dueño	NOUN	_	Gender=Masc|Number=Plur	9	obj	_	_
12	y	y	CCONJ	_	_	27	cc	_	_
13	que	que	SCONJ	_	_	27	mark	_	_
14	a	a	ADP	_	_	18	case	_	MWE=a_partir_de|MWEPOS=ADP
15	partir	partir	NOUN	_	_	14	fixed	_	_
16	de	de	ADP	_	_	14	fixed	_	_
17	la	el	DET	_	Definite=Def|Gender=Fem|Number=Sing|PronType=Art	18	det	_	_
18	frontera	frontera	NOUN	_	Gender=Fem|Number=Sing	27	obl	_	_
19	de	de	ADP	_	_	21	case	_	_
20	aquella	aquel	DET	_	Gender=Fem|Number=Sing|PronType=Dem	21	det	_	_
21	ola	ola	NOUN	_	Gender=Fem|Number=Sing	18	nmod	_	_
22	el	el	DET	_	Definite=Def|Gender=Masc|Number=Sing|PronType=Art	23	det	_	_
23	pescado	pescado	NOUN	_	Gender=Masc|Number=Sing	27	nsubj	_	_
24	ya	ya	ADV	_	_	27	advmod	_	_
25	no	no	ADV	_	Polarity=Neg	27	advmod	_	_
26	es	ser	AUX	_	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	27	cop	_	_
27	tuyo	tuyo	PRON	_	Gender=Masc|Number=Sing|Number[psor]=Sing|Person=2|Poss=Yes|PronType=Ind	9	conj	_	SpaceAfter=No
28	?	?	PUNCT	_	PunctSide=Fin|PunctType=Qest	3	punct	_	SpaceAfter=No
29	.	.	PUNCT	_	PunctType=Peri	3	punct	_	_

# sent_id = train-s8516
# text = ¿ Pero es divertido en la vida real? - -.
# orig_file_sentence 086#16
# this sentence will have the ¿ removed even with no SpaceAfter=No
1	¿	¿	PUNCT	_	PunctSide=Ini|PunctType=Qest	4	punct	_	_
2	Pero	pero	CCONJ	_	_	4	advmod	_	_
3	es	ser	AUX	_	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	4	cop	_	_
4	divertido	divertido	ADJ	_	Gender=Masc|Number=Sing|VerbForm=Part	0	root	_	_
5	en	en	ADP	_	_	7	case	_	_
6	la	el	DET	_	Definite=Def|Gender=Fem|Number=Sing|PronType=Art	7	det	_	_
7	vida	vida	NOUN	_	Gender=Fem|Number=Sing	4	obl	_	_
8	real	real	ADJ	_	Number=Sing	7	amod	_	SpaceAfter=No
9	?	?	PUNCT	_	PunctSide=Fin|PunctType=Qest	4	punct	_	_
10	-	-	PUNCT	_	PunctType=Dash	4	punct	_	_
11	-	-	PUNCT	_	PunctType=Dash	4	punct	_	SpaceAfter=No
12	.	.	PUNCT	_	PunctType=Peri	4	punct	_	_

# sent_id = train-s2337
# text = Es imposible.
# orig_file_sentence 024#37
# Also included is a sentence which should be skipped (note that it does not show up in the expected result)
1	Es	ser	AUX	_	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	2	cop	_	_
2	imposible	imposible	ADJ	_	Number=Sing	0	root	_	SpaceAfter=No
3	.	.	PUNCT	_	PunctType=Peri	2	punct	_	_

# sent_id = 3LB-CAST-a1-2-s6
# text = ¿Para qué seguir?
# orig_file_sentence 006#22
# The treebank now includes basic dependencies in the additional dependencies column
1	¿	¿	PUNCT	fia	PunctSide=Ini|PunctType=Qest	4	punct	4:punct	SpaceAfter=No
2	Para	para	ADP	sps00	_	3	case	3:case	_
3	qué	qué	PRON	pt0cs000	Number=Sing|PronType=Int,Rel	4	obl	4:obl	_
4	seguir	seguir	VERB	vmn0000	VerbForm=Inf	0	root	0:root	SpaceAfter=No
5	?	?	PUNCT	fit	PunctSide=Fin|PunctType=Qest	4	punct	4:punct	_

# sent_id = CESS-CAST-P-19990901-16-s19
# text = ¿Estará fingiendo?.
# orig_file_sentence 097#24
# also it includes some copy nodes
1	¿	¿	PUNCT	fia	PunctSide=Ini|PunctType=Qest	3	punct	3:punct	SpaceAfter=No
2	Estará	estar	AUX	vmif3s0	Mood=Ind|Number=Sing|Person=3|Tense=Fut|VerbForm=Fin	3	aux	3:aux	_
3	fingiendo	fingir	VERB	vmg0000	VerbForm=Ger	0	root	0:root	SpaceAfter=No
3.1	_	_	PRON	p	_	_	_	3:nsubj	Entity=(CESSCASTP1999090116c2-person-1-CorefType:ident,gstype:spec)
4	?	?	PUNCT	fit	PunctSide=Fin|PunctType=Qest	3	punct	3:punct	SpaceAfter=No
5	.	.	PUNCT	fp	PunctType=Peri	3	punct	3:punct	_

# sent_id = CESS-CAST-P-20000401-126-s31
# text = ¿Qué pensó cuando se quedó
# orig_file_sentence 087#37
# this one has a colon in the dependency name
1	¿	¿	PUNCT	fia	PunctSide=Ini|PunctType=Qest	3	punct	3:punct	SpaceAfter=No|Entity=(CESSCASTP20000401126c27--3
2	Qué	qué	PRON	pt0cs000	Number=Sing|PronType=Int,Rel	3	obj	3:obj	_
3	pensó	pensar	VERB	vmis3s0	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	0	root	0:root	_
3.1	_	_	PRON	p	_	_	_	3:nsubj	Entity=(CESSCASTP20000401126c1-person-1-CorefType:ident,gstype:spec)
4	cuando	cuando	SCONJ	cs	_	6	mark	6:mark	_
4.1	_	_	PRON	p	_	_	_	6:nsubj	Entity=(CESSCASTP20000401126c1-person-1-CorefType:ident,gstype:spec)
5	se	él	PRON	p0300000	Case=Acc|Person=3|PrepCase=Npr|PronType=Prs|Reflex=Yes	6	expl:pv	6:expl:pv	_
6	quedó	quedar	VERB	vmis3s0	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	3	advcl	3:advcl	_
"""

SPANISH_QM_RESULT = """
# sent_id = train-s7914
# text = Cómo explicarles entonces que el mar tiene varios dueños y que a partir de la frontera de aquella ola el pescado ya no es tuyo?.
# orig_file_sentence 080#14
# this sentence will have the intiial ¿ removed.  an MWT should be preserved
1	Cómo	cómo	PRON	_	PronType=Ind	2	obl	_	_
2-3	explicarles	_	_	_	_	_	_	_	_
2	explicar	explicar	VERB	_	VerbForm=Inf	0	root	_	_
3	les	él	PRON	_	Case=Dat|Number=Plur|Person=3|PronType=Prs	2	obj	_	_
4	entonces	entonces	ADV	_	_	2	advmod	_	_
5	que	que	SCONJ	_	_	8	mark	_	_
6	el	el	DET	_	Definite=Def|Gender=Masc|Number=Sing|PronType=Art	7	det	_	_
7	mar	mar	NOUN	_	Number=Sing	8	nsubj	_	_
8	tiene	tener	VERB	_	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	2	ccomp	_	_
9	varios	varios	DET	_	Gender=Masc|Number=Plur|PronType=Ind	10	det	_	_
10	dueños	dueño	NOUN	_	Gender=Masc|Number=Plur	8	obj	_	_
11	y	y	CCONJ	_	_	26	cc	_	_
12	que	que	SCONJ	_	_	26	mark	_	_
13	a	a	ADP	_	_	17	case	_	MWE=a_partir_de|MWEPOS=ADP
14	partir	partir	NOUN	_	_	13	fixed	_	_
15	de	de	ADP	_	_	13	fixed	_	_
16	la	el	DET	_	Definite=Def|Gender=Fem|Number=Sing|PronType=Art	17	det	_	_
17	frontera	frontera	NOUN	_	Gender=Fem|Number=Sing	26	obl	_	_
18	de	de	ADP	_	_	20	case	_	_
19	aquella	aquel	DET	_	Gender=Fem|Number=Sing|PronType=Dem	20	det	_	_
20	ola	ola	NOUN	_	Gender=Fem|Number=Sing	17	nmod	_	_
21	el	el	DET	_	Definite=Def|Gender=Masc|Number=Sing|PronType=Art	22	det	_	_
22	pescado	pescado	NOUN	_	Gender=Masc|Number=Sing	26	nsubj	_	_
23	ya	ya	ADV	_	_	26	advmod	_	_
24	no	no	ADV	_	Polarity=Neg	26	advmod	_	_
25	es	ser	AUX	_	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	26	cop	_	_
26	tuyo	tuyo	PRON	_	Gender=Masc|Number=Sing|Number[psor]=Sing|Person=2|Poss=Yes|PronType=Ind	8	conj	_	SpaceAfter=No
27	?	?	PUNCT	_	PunctSide=Fin|PunctType=Qest	2	punct	_	SpaceAfter=No
28	.	.	PUNCT	_	PunctType=Peri	2	punct	_	_

# sent_id = train-s8516
# text = Pero es divertido en la vida real? - -.
# orig_file_sentence 086#16
# this sentence will have the ¿ removed even with no SpaceAfter=No
1	Pero	pero	CCONJ	_	_	3	advmod	_	_
2	es	ser	AUX	_	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	3	cop	_	_
3	divertido	divertido	ADJ	_	Gender=Masc|Number=Sing|VerbForm=Part	0	root	_	_
4	en	en	ADP	_	_	6	case	_	_
5	la	el	DET	_	Definite=Def|Gender=Fem|Number=Sing|PronType=Art	6	det	_	_
6	vida	vida	NOUN	_	Gender=Fem|Number=Sing	3	obl	_	_
7	real	real	ADJ	_	Number=Sing	6	amod	_	SpaceAfter=No
8	?	?	PUNCT	_	PunctSide=Fin|PunctType=Qest	3	punct	_	_
9	-	-	PUNCT	_	PunctType=Dash	3	punct	_	_
10	-	-	PUNCT	_	PunctType=Dash	3	punct	_	SpaceAfter=No
11	.	.	PUNCT	_	PunctType=Peri	3	punct	_	_

# sent_id = 3LB-CAST-a1-2-s6
# text = Para qué seguir?
# orig_file_sentence 006#22
# The treebank now includes basic dependencies in the additional dependencies column
1	Para	para	ADP	sps00	_	2	case	2:case	_
2	qué	qué	PRON	pt0cs000	Number=Sing|PronType=Int,Rel	3	obl	3:obl	_
3	seguir	seguir	VERB	vmn0000	VerbForm=Inf	0	root	0:root	SpaceAfter=No
4	?	?	PUNCT	fit	PunctSide=Fin|PunctType=Qest	3	punct	3:punct	_

# sent_id = CESS-CAST-P-19990901-16-s19
# text = Estará fingiendo?.
# orig_file_sentence 097#24
# also it includes some copy nodes
1	Estará	estar	AUX	vmif3s0	Mood=Ind|Number=Sing|Person=3|Tense=Fut|VerbForm=Fin	2	aux	2:aux	_
2	fingiendo	fingir	VERB	vmg0000	VerbForm=Ger	0	root	0:root	SpaceAfter=No
2.1	_	_	PRON	p	_	_	_	2:nsubj	Entity=(CESSCASTP1999090116c2-person-1-CorefType:ident,gstype:spec)
3	?	?	PUNCT	fit	PunctSide=Fin|PunctType=Qest	2	punct	2:punct	SpaceAfter=No
4	.	.	PUNCT	fp	PunctType=Peri	2	punct	2:punct	_

# sent_id = CESS-CAST-P-20000401-126-s31
# text = Qué pensó cuando se quedó
# orig_file_sentence 087#37
# this one has a colon in the dependency name
1	Qué	qué	PRON	pt0cs000	Number=Sing|PronType=Int,Rel	2	obj	2:obj	_
2	pensó	pensar	VERB	vmis3s0	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	0	root	0:root	_
2.1	_	_	PRON	p	_	_	_	2:nsubj	Entity=(CESSCASTP20000401126c1-person-1-CorefType:ident,gstype:spec)
3	cuando	cuando	SCONJ	cs	_	5	mark	5:mark	_
3.1	_	_	PRON	p	_	_	_	5:nsubj	Entity=(CESSCASTP20000401126c1-person-1-CorefType:ident,gstype:spec)
4	se	él	PRON	p0300000	Case=Acc|Person=3|PrepCase=Npr|PronType=Prs|Reflex=Yes	5	expl:pv	5:expl:pv	_
5	quedó	quedar	VERB	vmis3s0	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	2	advcl	2:advcl	_
"""

def test_augment_initial_punct():
    doc = read_test_doc(SPANISH_QM_TEST_CASE)
    doc2 = prepare_tokenizer_treebank.augment_initial_punct(doc, ratio=1.0)
    expected = doc + read_test_doc(SPANISH_QM_RESULT)
    assert doc2 == expected

SPANISH_SHOULD_THROW = """
# sent_id = 3LB-CAST-a1-2-s6
# text = ¿Para qué seguir?
# orig_file_sentence 006#22
# multiple heads are not handled yet in the augmented dependencies column
1	¿	¿	PUNCT	fia	PunctSide=Ini|PunctType=Qest	4	punct	4:punct	SpaceAfter=No
2	Para	para	ADP	sps00	_	3	case	3:case	_
3	qué	qué	PRON	pt0cs000	Number=Sing|PronType=Int,Rel	4	obl	4:obl,3:foo	_
4	seguir	seguir	VERB	vmn0000	VerbForm=Inf	0	root	0:root	SpaceAfter=No
5	?	?	PUNCT	fit	PunctSide=Fin|PunctType=Qest	4	punct	4:punct	_
"""

def test_augment_initial_punct_error():
    """
    The augment script should protect against the single dependency assumption changing in the future
    """
    doc = read_test_doc(SPANISH_SHOULD_THROW)
    with pytest.raises(NotImplementedError):
        doc2 = prepare_tokenizer_treebank.augment_initial_punct(doc, ratio=1.0)

# first sentence should have the space added
# second sentence should be unchanged
ARABIC_SPACE_AFTER_TEST_CASE = """
# newpar id = afp.20000815.0079:p6
# sent_id = afp.20000815.0079:p6u1
# text = وتتميز امسية الاربعاء الدولية باقامة 16 مباراة ودية.
# orig_file_sentence AFP_ARB_20000815.0079#6
1-2	وتتميز	_	_	_	_	_	_	_	_
1	و	وَ	CCONJ	C---------	_	0	root	0:root	Vform=وَ|Gloss=and|Root=wa|Translit=wa|LTranslit=wa
2	تتميز	تَمَيَّز	VERB	VIIA-3FS--	Aspect=Imp|Gender=Fem|Mood=Ind|Number=Sing|Person=3|VerbForm=Fin|Voice=Act	1	parataxis	1:parataxis	Vform=تَتَمَيَّزُ|Gloss=be_distinguished,stand_out,discern,distinguish|Root=m_y_z|Translit=tatamayyazu|LTranslit=tamayyaz
3	امسية	أُمسِيَّة	NOUN	N------S1R	Case=Nom|Definite=Cons|Number=Sing	2	nsubj	2:nsubj	Vform=أُمسِيَّةُ|Gloss=evening,soiree|Root=m_s_w|Translit=ʾumsīyatu|LTranslit=ʾumsīyat
4	الاربعاء	أَربِعَاء	NOUN	N------S2D	Case=Gen|Definite=Def|Number=Sing	3	nmod	3:nmod:gen	Vform=اَلأَربِعَاءِ|Gloss=Wednesday|Root=r_b_`|Translit=al-ʾarbiʿāʾi|LTranslit=ʾarbiʿāʾ
5	الدولية	دُوَلِيّ	ADJ	A-----FS1D	Case=Nom|Definite=Def|Gender=Fem|Number=Sing	3	amod	3:amod	Vform=اَلدُّوَلِيَّةُ|Gloss=international,world|Root=d_w_l|Translit=ad-duwalīyatu|LTranslit=duwalīy
6-7	باقامة	_	_	_	_	_	_	_	_
6	ب	بِ	ADP	P---------	AdpType=Prep	7	case	7:case	Vform=بِ|Gloss=by,with|Root=bi|Translit=bi|LTranslit=bi
7	إقامة	إِقَامَة	NOUN	N------S2R	Case=Gen|Definite=Cons|Number=Sing	2	obl	2:obl:بِ:gen	Vform=إِقَامَةِ|Gloss=residency,setting_up|Root=q_w_m|Translit=ʾiqāmati|LTranslit=ʾiqāmat
8	16	16	NUM	Q---------	NumForm=Digit	7	nummod	7:nummod	Vform=١٦|Translit=16
9	مباراة	مُبَارَاة	NOUN	N------S4I	Case=Acc|Definite=Ind|Number=Sing	8	nmod	8:nmod:acc	Vform=مُبَارَاةً|Gloss=match,game,competition|Root=b_r_y|Translit=mubārātan|LTranslit=mubārāt
10	ودية	وُدِّيّ	ADJ	A-----FS4I	Case=Acc|Definite=Ind|Gender=Fem|Number=Sing	9	amod	9:amod	SpaceAfter=No|Vform=وُدِّيَّةً|Gloss=friendly,amicable|Root=w_d_d|Translit=wuddīyatan|LTranslit=wuddīy
11	.	.	PUNCT	G---------	_	1	punct	1:punct	Vform=.|Translit=.

# newdoc id = afp.20000715.0075
# newpar id = afp.20000715.0075:p1
# sent_id = afp.20000715.0075:p1u1
# text = برلين ترفض حصول شركة اميركية على رخصة تصنيع دبابة "ليوبارد" الالمانية
# orig_file_sentence AFP_ARB_20000715.0075#1
1	برلين	بَرلِين	X	X---------	Foreign=Yes	2	nsubj	2:nsubj	Vform=بَرلِين|Gloss=Berlin|Root=barlIn|Translit=barlīn|LTranslit=barlīn
2	ترفض	رَفَض	VERB	VIIA-3FS--	Aspect=Imp|Gender=Fem|Mood=Ind|Number=Sing|Person=3|VerbForm=Fin|Voice=Act	0	root	0:root	Vform=تَرفُضُ|Gloss=reject,refuse|Root=r_f_.d|Translit=tarfuḍu|LTranslit=rafaḍ
3	حصول	حُصُول	NOUN	N------S4R	Case=Acc|Definite=Cons|Number=Sing	2	obj	2:obj	Vform=حُصُولَ|Gloss=acquisition,obtaining,occurrence,happening|Root=.h_.s_l|Translit=ḥuṣūla|LTranslit=ḥuṣūl
4	شركة	شَرِكَة	NOUN	N------S2I	Case=Gen|Definite=Ind|Number=Sing	3	nmod	3:nmod:gen	Vform=شَرِكَةٍ|Gloss=company,corporation|Root=^s_r_k|Translit=šarikatin|LTranslit=šarikat
5	اميركية	أَمِيرِكِيّ	ADJ	A-----FS2I	Case=Gen|Definite=Ind|Gender=Fem|Number=Sing	4	amod	4:amod	Vform=أَمِيرِكِيَّةٍ|Gloss=American|Root='amIrik|Translit=ʾamīrikīyatin|LTranslit=ʾamīrikīy
6	على	عَلَى	ADP	P---------	AdpType=Prep	7	case	7:case	Vform=عَلَى|Gloss=on,above|Root=`_l_w|Translit=ʿalā|LTranslit=ʿalā
7	رخصة	رُخصَة	NOUN	N------S2R	Case=Gen|Definite=Cons|Number=Sing	3	obl:arg	3:obl:arg:عَلَى:gen	Vform=رُخصَةِ|Gloss=license,permit|Root=r__h_.s|Translit=ruḫṣati|LTranslit=ruḫṣat
8	تصنيع	تَصنِيع	NOUN	N------S2R	Case=Gen|Definite=Cons|Number=Sing	7	nmod	7:nmod:gen	Vform=تَصنِيعِ|Gloss=fabrication,industrialization,processing|Root=.s_n_`|Translit=taṣnīʿi|LTranslit=taṣnīʿ
9	دبابة	دَبَّابَة	NOUN	N------S2R	Case=Gen|Definite=Cons|Number=Sing	8	nmod	8:nmod:gen	Vform=دَبَّابَةِ|Gloss=tank|Root=d_b_b|Translit=dabbābati|LTranslit=dabbābat
10	"	"	PUNCT	G---------	_	11	punct	11:punct	SpaceAfter=No|Vform="|Translit="
11	ليوبارد	لِيُوبَارد	X	X---------	Foreign=Yes	9	nmod	9:nmod	SpaceAfter=No|Vform=لِيُوبَارد|Gloss=Leopard|Root=liyUbArd|Translit=liyūbārd|LTranslit=liyūbārd
12	"	"	PUNCT	G---------	_	11	punct	11:punct	Vform="|Translit="
13	الالمانية	أَلمَانِيّ	ADJ	A-----FS2D	Case=Gen|Definite=Def|Gender=Fem|Number=Sing	9	amod	9:amod	Vform=اَلأَلمَانِيَّةِ|Gloss=German|Root='almAn|Translit=al-ʾalmānīyati|LTranslit=ʾalmānīy
"""

ARABIC_SPACE_AFTER_RESULT = """
# newpar id = afp.20000815.0079:p6
# sent_id = afp.20000815.0079:p6u1
# text = وتتميز امسية الاربعاء الدولية باقامة 16 مباراة ودية .
# orig_file_sentence AFP_ARB_20000815.0079#6
1-2	وتتميز	_	_	_	_	_	_	_	_
1	و	وَ	CCONJ	C---------	_	0	root	0:root	Vform=وَ|Gloss=and|Root=wa|Translit=wa|LTranslit=wa
2	تتميز	تَمَيَّز	VERB	VIIA-3FS--	Aspect=Imp|Gender=Fem|Mood=Ind|Number=Sing|Person=3|VerbForm=Fin|Voice=Act	1	parataxis	1:parataxis	Vform=تَتَمَيَّزُ|Gloss=be_distinguished,stand_out,discern,distinguish|Root=m_y_z|Translit=tatamayyazu|LTranslit=tamayyaz
3	امسية	أُمسِيَّة	NOUN	N------S1R	Case=Nom|Definite=Cons|Number=Sing	2	nsubj	2:nsubj	Vform=أُمسِيَّةُ|Gloss=evening,soiree|Root=m_s_w|Translit=ʾumsīyatu|LTranslit=ʾumsīyat
4	الاربعاء	أَربِعَاء	NOUN	N------S2D	Case=Gen|Definite=Def|Number=Sing	3	nmod	3:nmod:gen	Vform=اَلأَربِعَاءِ|Gloss=Wednesday|Root=r_b_`|Translit=al-ʾarbiʿāʾi|LTranslit=ʾarbiʿāʾ
5	الدولية	دُوَلِيّ	ADJ	A-----FS1D	Case=Nom|Definite=Def|Gender=Fem|Number=Sing	3	amod	3:amod	Vform=اَلدُّوَلِيَّةُ|Gloss=international,world|Root=d_w_l|Translit=ad-duwalīyatu|LTranslit=duwalīy
6-7	باقامة	_	_	_	_	_	_	_	_
6	ب	بِ	ADP	P---------	AdpType=Prep	7	case	7:case	Vform=بِ|Gloss=by,with|Root=bi|Translit=bi|LTranslit=bi
7	إقامة	إِقَامَة	NOUN	N------S2R	Case=Gen|Definite=Cons|Number=Sing	2	obl	2:obl:بِ:gen	Vform=إِقَامَةِ|Gloss=residency,setting_up|Root=q_w_m|Translit=ʾiqāmati|LTranslit=ʾiqāmat
8	16	16	NUM	Q---------	NumForm=Digit	7	nummod	7:nummod	Vform=١٦|Translit=16
9	مباراة	مُبَارَاة	NOUN	N------S4I	Case=Acc|Definite=Ind|Number=Sing	8	nmod	8:nmod:acc	Vform=مُبَارَاةً|Gloss=match,game,competition|Root=b_r_y|Translit=mubārātan|LTranslit=mubārāt
10	ودية	وُدِّيّ	ADJ	A-----FS4I	Case=Acc|Definite=Ind|Gender=Fem|Number=Sing	9	amod	9:amod	Vform=وُدِّيَّةً|Gloss=friendly,amicable|Root=w_d_d|Translit=wuddīyatan|LTranslit=wuddīy
11	.	.	PUNCT	G---------	_	1	punct	1:punct	Vform=.|Translit=.
"""

def test_augment_space_final_punct():
    doc = read_test_doc(ARABIC_SPACE_AFTER_TEST_CASE)
    doc2 = prepare_tokenizer_treebank.augment_arabic_padt(doc, ratio=1.0)
    expected = doc + read_test_doc(ARABIC_SPACE_AFTER_RESULT)
    assert doc2 == expected

ENGLISH_COMMA_SWAP_TEST_CASE="""
# sent_id = reviews-086839-0004
# text = Approx 4 months later, the compressor went out.
1	Approx	approx	ADV	RB	_	3	advmod	3:advmod	_
2	4	4	NUM	CD	NumType=Card	3	nummod	3:nummod	_
3	months	month	NOUN	NNS	Number=Plur	4	obl:npmod	4:obl:npmod	_
4	later	late	ADV	RBR	Degree=Cmp	8	advmod	8:advmod	SpaceAfter=No
5	,	,	PUNCT	,	_	8	punct	8:punct	_
6	the	the	DET	DT	Definite=Def|PronType=Art	7	det	7:det	_
7	compressor	compressor	NOUN	NN	Number=Sing	8	nsubj	8:nsubj	_
8	went	go	VERB	VBD	Mood=Ind|Tense=Past|VerbForm=Fin	0	root	0:root	_
9	out	out	ADP	RP	_	8	compound:prt	8:compound:prt	SpaceAfter=No
10	.	.	PUNCT	.	_	8	punct	8:punct	_

# sent_id = reviews-086839-0004b
# text = Approx 4 months later , the compressor went out.
1	Approx	approx	ADV	RB	_	3	advmod	3:advmod	_
2	4	4	NUM	CD	NumType=Card	3	nummod	3:nummod	_
3	months	month	NOUN	NNS	Number=Plur	4	obl:npmod	4:obl:npmod	_
4	later	late	ADV	RBR	Degree=Cmp	8	advmod	8:advmod	_
5	,	,	PUNCT	,	_	8	punct	8:punct	_
6	the	the	DET	DT	Definite=Def|PronType=Art	7	det	7:det	_
7	compressor	compressor	NOUN	NN	Number=Sing	8	nsubj	8:nsubj	_
8	went	go	VERB	VBD	Mood=Ind|Tense=Past|VerbForm=Fin	0	root	0:root	_
9	out	out	ADP	RP	_	8	compound:prt	8:compound:prt	SpaceAfter=No
10	.	.	PUNCT	.	_	8	punct	8:punct	_
"""

ENGLISH_COMMA_SWAP_RESULT="""
# sent_id = reviews-086839-0004
# text = Approx 4 months later ,the compressor went out.
1	Approx	approx	ADV	RB	_	3	advmod	3:advmod	_
2	4	4	NUM	CD	NumType=Card	3	nummod	3:nummod	_
3	months	month	NOUN	NNS	Number=Plur	4	obl:npmod	4:obl:npmod	_
4	later	late	ADV	RBR	Degree=Cmp	8	advmod	8:advmod	_
5	,	,	PUNCT	,	_	8	punct	8:punct	SpaceAfter=No
6	the	the	DET	DT	Definite=Def|PronType=Art	7	det	7:det	_
7	compressor	compressor	NOUN	NN	Number=Sing	8	nsubj	8:nsubj	_
8	went	go	VERB	VBD	Mood=Ind|Tense=Past|VerbForm=Fin	0	root	0:root	_
9	out	out	ADP	RP	_	8	compound:prt	8:compound:prt	SpaceAfter=No
10	.	.	PUNCT	.	_	8	punct	8:punct	_

# sent_id = reviews-086839-0004b
# text = Approx 4 months later , the compressor went out.
1	Approx	approx	ADV	RB	_	3	advmod	3:advmod	_
2	4	4	NUM	CD	NumType=Card	3	nummod	3:nummod	_
3	months	month	NOUN	NNS	Number=Plur	4	obl:npmod	4:obl:npmod	_
4	later	late	ADV	RBR	Degree=Cmp	8	advmod	8:advmod	_
5	,	,	PUNCT	,	_	8	punct	8:punct	_
6	the	the	DET	DT	Definite=Def|PronType=Art	7	det	7:det	_
7	compressor	compressor	NOUN	NN	Number=Sing	8	nsubj	8:nsubj	_
8	went	go	VERB	VBD	Mood=Ind|Tense=Past|VerbForm=Fin	0	root	0:root	_
9	out	out	ADP	RP	_	8	compound:prt	8:compound:prt	SpaceAfter=No
10	.	.	PUNCT	.	_	8	punct	8:punct	_
"""

def test_augment_space_final_punct():
    doc = read_test_doc(ENGLISH_COMMA_SWAP_TEST_CASE)
    doc2 = prepare_tokenizer_treebank.augment_move_comma(doc, ratio=1.0)
    expected = read_test_doc(ENGLISH_COMMA_SWAP_RESULT)
    assert doc2 == expected
