import pytest
import stanza
from tests import *

from stanza.utils.datasets import prepare_tokenizer_treebank

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

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
"""

def test_augment_initial_punct():
    doc = read_test_doc(SPANISH_QM_TEST_CASE)
    doc2 = prepare_tokenizer_treebank.augment_initial_punct(doc, ratio=1.0)
    expected = doc + read_test_doc(SPANISH_QM_RESULT)
    assert doc2 == expected
