"""
Test a couple different classes of trees to check the output of the VIT conversion

A couple representative trees are included, but hopefully not enough
to be a problem in terms of our license.

One of the tests is currently disabled as it relies on tregex & tsurgeon features
not yet released
"""

import io
import os
import tempfile

import pytest

from stanza.server import tsurgeon
from stanza.utils.conll import CoNLL
from stanza.utils.datasets.constituency import convert_it_vit

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

# just a sample!  don't sue us please
CON_SAMPLE = """
#ID=sent_00002	cp-[sp-[part-negli, sn-[sa-[ag-ultimi], nt-anni]], f-[sn-[art-la, n-dinamica, spd-[partd-dei, sn-[n-polo_di_attrazione]]], ibar-[ause-è, ausep-stata, savv-[savv-[avv-sempre], avv-più], vppt-caratterizzata], compin-[spda-[partda-dall, sn-[n-emergere, spd-[pd-di, sn-[art-una, sa-[ag-crescente], n-concorrenza, f2-[rel-che, f-[ibar-[clit-si, ause-è, avv-progressivamente, vppin-spostata], compin-[spda-[partda-dalle, sn-[sa-[ag-singole], n-imprese]], sp-[part-ai, sn-[n-sistemi, sa-[coord-[ag-economici, cong-e, ag-territoriali]]]], fp-[punt-',', sv5-[vgt-determinando, compt-[sn-[art-l_, nf-esigenza, spd-[pd-di, sn-[art-una, n-riconsiderazione, spd-[partd-dei, sn-[n-rapporti, sv3-[ppre-esistenti, compin-[sp-[p-tra, sn-[n-soggetti, sa-[ag-produttivi]]], cong-e, sn-[n-ambiente, f2-[sp-[p-in, sn-[relob-cui]], f-[sn-[deit-questi], ibar-[vin-operano, punto-.]]]]]]]]]]]]]]]]]]]]]]]]

#ID=sent_00318	dirsp-[fc-[congf-tuttavia, f-[sn-[sq-[ind-qualche], n-problema], ir_infl-[vsupir-potrebbe, vcl-esserci], compc-[clit-ci, sp-[p-per, sn-[art-la, n-commissione, sa-[ag-esteri], f2-[sp-[part-alla, relob-cui, sn-[n-presidenza]], f-[ibar-[vc-è], compc-[sn-[n-candidato], sn-[art-l, n-esponente, spd-[pd-di, sn-[mw-Alleanza, npro-Nazionale]], sn-[mw-Mirko, nh-Tremaglia]]]]]]]]]], dirs-':', f3-[sn-[art-una, n-candidatura, sc-[q-più, sa-[ppas-subìta], sc-[ccong-che, sa-[ppas-gradita]], compt-[spda-[partda-dalla, sn-[mw-Lega, npro-Nord, punt-',', f2-[rel-che, fc-[congf-tuttavia, f-[ir_infl-[vsupir-dovrebbe, vit-rispettare], compt-[sn-[art-gli, n-accordi]]]]]]]]]], punto-.]]

#ID=sent_00589	f-[sn-[art-l, n-ottimismo, spd-[pd-di, sn-[nh-Kantor]]], ir_infl-[vsupir-potrebbe, congf-però, vcl-rivelarsi], compc-[sn-[in-ancora, art-una, nt-volta], sa-[ag-prematuro]], punto-.]
"""

UD_SAMPLE = """
# sent_id = VIT-2
# text = Negli ultimi anni la dinamica dei polo di attrazione è stata sempre più caratterizzata dall'emergere di una crescente concorrenza che si è progressivamente spostata dalle singole imprese ai sistemi economici e territoriali, determinando l'esigenza di una riconsiderazione dei rapporti esistenti tra soggetti produttivi e ambiente in cui questi operano.
1-2	Negli	_	_	_	_	_	_	_	_
1	In	in	ADP	E	_	4	case	_	_
2	gli	il	DET	RD	Definite=Def|Gender=Masc|Number=Plur|PronType=Art	4	det	_	_
3	ultimi	ultimo	ADJ	A	Gender=Masc|Number=Plur	4	amod	_	_
4	anni	anno	NOUN	S	Gender=Masc|Number=Plur	16	obl	_	_
5	la	il	DET	RD	Definite=Def|Gender=Fem|Number=Sing|PronType=Art	6	det	_	_
6	dinamica	dinamica	NOUN	S	Gender=Fem|Number=Sing	16	nsubj:pass	_	_
7-8	dei	_	_	_	_	_	_	_	_
7	di	di	ADP	E	_	9	case	_	_
8	i	il	DET	RD	Definite=Def|Gender=Masc|Number=Plur|PronType=Art	9	det	_	_
9	polo	polo	NOUN	S	Gender=Masc|Number=Sing	6	nmod	_	_
10	di	di	ADP	E	_	11	case	_	_
11	attrazione	attrazione	NOUN	S	Gender=Fem|Number=Sing	9	nmod	_	_
12	è	essere	AUX	VA	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	16	aux	_	_
13	stata	essere	AUX	VA	Gender=Fem|Number=Sing|Tense=Past|VerbForm=Part	16	aux:pass	_	_
14	sempre	sempre	ADV	B	_	15	advmod	_	_
15	più	più	ADV	B	_	16	advmod	_	_
16	caratterizzata	caratterizzare	VERB	V	Gender=Fem|Number=Sing|Tense=Past|VerbForm=Part	0	root	_	_
17-18	dall'	_	_	_	_	_	_	_	SpaceAfter=No
17	da	da	ADP	E	_	19	case	_	_
18	l'	il	DET	RD	Definite=Def|Number=Sing|PronType=Art	19	det	_	_
19	emergere	emergere	NOUN	S	Gender=Masc|Number=Sing	16	obl	_	_
20	di	di	ADP	E	_	23	case	_	_
21	una	uno	DET	RI	Definite=Ind|Gender=Fem|Number=Sing|PronType=Art	23	det	_	_
22	crescente	crescente	ADJ	A	Number=Sing	23	amod	_	_
23	concorrenza	concorrenza	NOUN	S	Gender=Fem|Number=Sing	19	nmod	_	_
24	che	che	PRON	PR	PronType=Rel	28	nsubj	_	_
25	si	si	PRON	PC	Clitic=Yes|Person=3|PronType=Prs	28	expl	_	_
26	è	essere	AUX	VA	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	28	aux	_	_
27	progressivamente	progressivamente	ADV	B	_	28	advmod	_	_
28	spostata	spostare	VERB	V	Gender=Fem|Number=Sing|Tense=Past|VerbForm=Part	23	acl:relcl	_	_
29-30	dalle	_	_	_	_	_	_	_	_
29	da	da	ADP	E	_	32	case	_	_
30	le	il	DET	RD	Definite=Def|Gender=Fem|Number=Plur|PronType=Art	32	det	_	_
31	singole	singolo	ADJ	A	Gender=Fem|Number=Plur	32	amod	_	_
32	imprese	impresa	NOUN	S	Gender=Fem|Number=Plur	28	obl	_	_
33-34	ai	_	_	_	_	_	_	_	_
33	a	a	ADP	E	_	35	case	_	_
34	i	il	DET	RD	Definite=Def|Gender=Masc|Number=Plur|PronType=Art	35	det	_	_
35	sistemi	sistema	NOUN	S	Gender=Masc|Number=Plur	28	obl	_	_
36	economici	economico	ADJ	A	Gender=Masc|Number=Plur	35	amod	_	_
37	e	e	CCONJ	CC	_	38	cc	_	_
38	territoriali	territoriale	ADJ	A	Number=Plur	36	conj	_	SpaceAfter=No
39	,	,	PUNCT	FF	_	28	punct	_	_
40	determinando	determinare	VERB	V	VerbForm=Ger	28	advcl	_	_
41	l'	il	DET	RD	Definite=Def|Number=Sing|PronType=Art	42	det	_	SpaceAfter=No
42	esigenza	esigenza	NOUN	S	Gender=Fem|Number=Sing	40	obj	_	_
43	di	di	ADP	E	_	45	case	_	_
44	una	uno	DET	RI	Definite=Ind|Gender=Fem|Number=Sing|PronType=Art	45	det	_	_
45	riconsiderazione	riconsiderazione	NOUN	S	Gender=Fem|Number=Sing	42	nmod	_	_
46-47	dei	_	_	_	_	_	_	_	_
46	di	di	ADP	E	_	48	case	_	_
47	i	il	DET	RD	Definite=Def|Gender=Masc|Number=Plur|PronType=Art	48	det	_	_
48	rapporti	rapporto	NOUN	S	Gender=Masc|Number=Plur	45	nmod	_	_
49	esistenti	esistente	VERB	V	Number=Plur	48	acl	_	_
50	tra	tra	ADP	E	_	51	case	_	_
51	soggetti	soggetto	NOUN	S	Gender=Masc|Number=Plur	49	obl	_	_
52	produttivi	produttivo	ADJ	A	Gender=Masc|Number=Plur	51	amod	_	_
53	e	e	CCONJ	CC	_	54	cc	_	_
54	ambiente	ambiente	NOUN	S	Gender=Masc|Number=Sing	51	conj	_	_
55	in	in	ADP	E	_	56	case	_	_
56	cui	cui	PRON	PR	PronType=Rel	58	obl	_	_
57	questi	questo	PRON	PD	Gender=Masc|Number=Plur|PronType=Dem	58	nsubj	_	_
58	operano	operare	VERB	V	Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin	54	acl:relcl	_	SpaceAfter=No
59	.	.	PUNCT	FS	_	16	punct	_	_

# sent_id = VIT-318
# text = Tuttavia qualche problema potrebbe esserci per la commissione esteri alla cui presidenza è candidato l'esponente di Alleanza Nazionale Mirko Tremaglia: una candidatura più subìta che gradita dalla Lega Nord, che tuttavia dovrebbe rispettare gli accordi.
1	Tuttavia	tuttavia	CCONJ	CC	_	5	cc	_	_
2	qualche	qualche	DET	DI	Number=Sing|PronType=Ind	3	det	_	_
3	problema	problema	NOUN	S	Gender=Masc|Number=Sing	5	nsubj	_	_
4	potrebbe	potere	AUX	VA	Mood=Cnd|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	5	aux	_	_
5-6	esserci	_	_	_	_	_	_	_	_
5	esser	essere	VERB	V	VerbForm=Inf	0	root	_	_
6	ci	ci	PRON	PC	Clitic=Yes|Number=Plur|Person=1|PronType=Prs	5	expl	_	_
7	per	per	ADP	E	_	9	case	_	_
8	la	il	DET	RD	Definite=Def|Gender=Fem|Number=Sing|PronType=Art	9	det	_	_
9	commissione	commissione	NOUN	S	Gender=Fem|Number=Sing	5	obl	_	_
10	esteri	estero	ADJ	A	Gender=Masc|Number=Plur	9	amod	_	_
11-12	alla	_	_	_	_	_	_	_	_
11	a	a	ADP	E	_	14	case	_	_
12	la	il	DET	RD	Definite=Def|Gender=Fem|Number=Sing|PronType=Art	14	det	_	_
13	cui	cui	DET	DR	PronType=Rel	14	det:poss	_	_
14	presidenza	presidenza	NOUN	S	Gender=Fem|Number=Sing	16	obl	_	_
15	è	essere	AUX	VA	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	16	aux:pass	_	_
16	candidato	candidare	VERB	V	Gender=Masc|Number=Sing|Tense=Past|VerbForm=Part	9	acl:relcl	_	_
17	l'	il	DET	RD	Definite=Def|Number=Sing|PronType=Art	18	det	_	SpaceAfter=No
18	esponente	esponente	NOUN	S	Number=Sing	16	nsubj:pass	_	_
19	di	di	ADP	E	_	20	case	_	_
20	Alleanza	Alleanza	PROPN	SP	_	18	nmod	_	_
21	Nazionale	Nazionale	PROPN	SP	_	20	flat:name	_	_
22	Mirko	Mirko	PROPN	SP	_	18	nmod	_	_
23	Tremaglia	Tremaglia	PROPN	SP	_	22	flat:name	_	SpaceAfter=No
24	:	:	PUNCT	FC	_	22	punct	_	_
25	una	uno	DET	RI	Definite=Ind|Gender=Fem|Number=Sing|PronType=Art	26	det	_	_
26	candidatura	candidatura	NOUN	S	Gender=Fem|Number=Sing	22	appos	_	_
27	più	più	ADV	B	_	28	advmod	_	_
28	subìta	subire	VERB	V	Gender=Fem|Number=Sing|Tense=Past|VerbForm=Part	26	advcl	_	_
29	che	che	CCONJ	CC	_	30	cc	_	_
30	gradita	gradito	ADJ	A	Gender=Fem|Number=Sing	28	amod	_	_
31-32	dalla	_	_	_	_	_	_	_	_
31	da	da	ADP	E	_	33	case	_	_
32	la	il	DET	RD	Definite=Def|Gender=Fem|Number=Sing|PronType=Art	33	det	_	_
33	Lega	Lega	PROPN	SP	_	28	obl:agent	_	_
34	Nord	Nord	PROPN	SP	_	33	flat:name	_	SpaceAfter=No
35	,	,	PUNCT	FC	_	33	punct	_	_
36	che	che	PRON	PR	PronType=Rel	39	nsubj	_	_
37	tuttavia	tuttavia	CCONJ	CC	_	39	cc	_	_
38	dovrebbe	dovere	AUX	VM	Mood=Cnd|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	39	aux	_	_
39	rispettare	rispettare	VERB	V	VerbForm=Inf	33	acl:relcl	_	_
40	gli	il	DET	RD	Definite=Def|Gender=Masc|Number=Plur|PronType=Art	41	det	_	_
41	accordi	accordio	NOUN	S	Gender=Masc|Number=Plur	39	obj	_	SpaceAfter=No
42	.	.	PUNCT	FS	_	5	punct	_	_

# sent_id = VIT-591
# text = L'ottimismo di Kantor potrebbe però rivelarsi ancora una volta prematuro.
1	L'	il	DET	RD	Definite=Def|Number=Sing|PronType=Art	2	det	_	SpaceAfter=No
2	ottimismo	ottimismo	NOUN	S	Gender=Masc|Number=Sing	7	nsubj	_	_
3	di	di	ADP	E	_	4	case	_	_
4	Kantor	Kantor	PROPN	SP	_	2	nmod	_	_
5	potrebbe	potere	AUX	VM	Mood=Cnd|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	7	aux	_	_
6	però	però	ADV	B	_	7	advmod	_	_
7-8	rivelarsi	_	_	_	_	_	_	_	_
7	rivelar	rivelare	VERB	V	VerbForm=Inf	0	root	_	_
8	si	si	PRON	PC	Clitic=Yes|Person=3|PronType=Prs	7	expl	_	_
9	ancora	ancora	ADV	B	_	7	advmod	_	_
10	una	uno	DET	RI	Definite=Ind|Gender=Fem|Number=Sing|PronType=Art	11	det	_	_
11	volta	volta	NOUN	S	Gender=Fem|Number=Sing	7	obl	_	_
12	prematuro	prematuro	ADJ	A	Gender=Masc|Number=Sing	7	xcomp	_	SpaceAfter=No
13	.	.	PUNCT	FS	_	7	punct	_	_
"""


def test_process_mwts():
    # dei appears multiple times
    # the verb/pron esserci will be ignored
    expected_mwts = {'Negli': ('In', 'gli'), 'dei': ('di', 'i'), "dall'": ('da', "l'"), 'dalle': ('da', 'le'), 'ai': ('a', 'i'), 'alla': ('a', 'la'), 'dalla': ('da', 'la')}

    ud_train_data = CoNLL.conll2doc(input_str=UD_SAMPLE)

    mwts = convert_it_vit.get_mwt(ud_train_data)
    assert expected_mwts == mwts

def test_raw_tree():
    con_sentences = convert_it_vit.read_constituency_sentences(io.StringIO(CON_SAMPLE))
    expected_ids = ["#ID=sent_00002", "#ID=sent_00318", "#ID=sent_00589"]
    expected_trees = ["(ROOT (cp (sp (part negli) (sn (sa (ag ultimi)) (nt anni))) (f (sn (art la) (n dinamica) (spd (partd dei) (sn (n polo) (n di) (n attrazione)))) (ibar (ause è) (ausep stata) (savv (savv (avv sempre)) (avv più)) (vppt caratterizzata)) (compin (spda (partda dall) (sn (n emergere) (spd (pd di) (sn (art una) (sa (ag crescente)) (n concorrenza) (f2 (rel che) (f (ibar (clit si) (ause è) (avv progressivamente) (vppin spostata)) (compin (spda (partda dalle) (sn (sa (ag singole)) (n imprese))) (sp (part ai) (sn (n sistemi) (sa (coord (ag economici) (cong e) (ag territoriali))))) (fp (punt ,) (sv5 (vgt determinando) (compt (sn (art l') (nf esigenza) (spd (pd di) (sn (art una) (n riconsiderazione) (spd (partd dei) (sn (n rapporti) (sv3 (ppre esistenti) (compin (sp (p tra) (sn (n soggetti) (sa (ag produttivi)))) (cong e) (sn (n ambiente) (f2 (sp (p in) (sn (relob cui))) (f (sn (deit questi)) (ibar (vin operano) (punto .))))))))))))))))))))))))))",
                      "(ROOT (dirsp (fc (congf tuttavia) (f (sn (sq (ind qualche)) (n problema)) (ir_infl (vsupir potrebbe) (vcl esserci)) (compc (clit ci) (sp (p per) (sn (art la) (n commissione) (sa (ag esteri)) (f2 (sp (part alla) (relob cui) (sn (n presidenza))) (f (ibar (vc è)) (compc (sn (n candidato)) (sn (art l) (n esponente) (spd (pd di) (sn (mw Alleanza) (npro Nazionale))) (sn (mw Mirko) (nh Tremaglia))))))))))) (dirs :) (f3 (sn (art una) (n candidatura) (sc (q più) (sa (ppas subìta)) (sc (ccong che) (sa (ppas gradita))) (compt (spda (partda dalla) (sn (mw Lega) (npro Nord) (punt ,) (f2 (rel che) (fc (congf tuttavia) (f (ir_infl (vsupir dovrebbe) (vit rispettare)) (compt (sn (art gli) (n accordi))))))))))) (punto .))))",
                      "(ROOT (f (sn (art l) (n ottimismo) (spd (pd di) (sn (nh Kantor)))) (ir_infl (vsupir potrebbe) (congf però) (vcl rivelarsi)) (compc (sn (in ancora) (art una) (nt volta)) (sa (ag prematuro))) (punto .)))"]
    assert len(con_sentences) == 3
    for sentence, expected_id, expected_tree in zip(con_sentences, expected_ids, expected_trees):
        assert sentence[0] == expected_id
        tree = convert_it_vit.raw_tree(sentence[1])
        assert str(tree) == expected_tree

def test_update_mwts():
    con_sentences = convert_it_vit.read_constituency_sentences(io.StringIO(CON_SAMPLE))
    ud_train_data = CoNLL.conll2doc(input_str=UD_SAMPLE)
    mwt_map = convert_it_vit.get_mwt(ud_train_data)
    expected_trees=["(ROOT (cp (sp (part In) (sn (art gli) (sa (ag ultimi)) (nt anni))) (f (sn (art la) (n dinamica) (spd (partd di) (sn (art i) (n polo) (n di) (n attrazione)))) (ibar (ause è) (ausep stata) (savv (savv (avv sempre)) (avv più)) (vppt caratterizzata)) (compin (spda (partda da) (sn (art l') (n emergere) (spd (pd di) (sn (art una) (sa (ag crescente)) (n concorrenza) (f2 (rel che) (f (ibar (clit si) (ause è) (avv progressivamente) (vppin spostata)) (compin (spda (partda da) (sn (art le) (sa (ag singole)) (n imprese))) (sp (part a) (sn (art i) (n sistemi) (sa (coord (ag economici) (cong e) (ag territoriali))))) (fp (punt ,) (sv5 (vgt determinando) (compt (sn (art l') (nf esigenza) (spd (pd di) (sn (art una) (n riconsiderazione) (spd (partd di) (sn (art i) (n rapporti) (sv3 (ppre esistenti) (compin (sp (p tra) (sn (n soggetti) (sa (ag produttivi)))) (cong e) (sn (n ambiente) (f2 (sp (p in) (sn (relob cui))) (f (sn (deit questi)) (ibar (vin operano) (punto .))))))))))))))))))))))))))",
                    "(ROOT (dirsp (fc (congf tuttavia) (f (sn (sq (ind qualche)) (n problema)) (ir_infl (vsupir potrebbe) (vcl esserci)) (compc (clit ci) (sp (p per) (sn (art la) (n commissione) (sa (ag esteri)) (f2 (sp (part a) (art la) (relob cui) (sn (n presidenza))) (f (ibar (vc è)) (compc (sn (n candidato)) (sn (art l) (n esponente) (spd (pd di) (sn (mw Alleanza) (npro Nazionale))) (sn (mw Mirko) (nh Tremaglia))))))))))) (dirs :) (f3 (sn (art una) (n candidatura) (sc (q più) (sa (ppas subìta)) (sc (ccong che) (sa (ppas gradita))) (compt (spda (partda da) (sn (art la) (mw Lega) (npro Nord) (punt ,) (f2 (rel che) (fc (congf tuttavia) (f (ir_infl (vsupir dovrebbe) (vit rispettare)) (compt (sn (art gli) (n accordi))))))))))) (punto .))))",
                    "(ROOT (f (sn (art l) (n ottimismo) (spd (pd di) (sn (nh Kantor)))) (ir_infl (vsupir potrebbe) (congf però) (vcl rivelarsi)) (compc (clit si) (sn (in ancora) (art una) (nt volta)) (sa (ag prematuro))) (punto .)))"]
    with tsurgeon.Tsurgeon(classpath="$CLASSPATH") as tsurgeon_processor:
        for con_sentence, ud_sentence, expected_tree in zip(con_sentences, ud_train_data.sentences, expected_trees):
            con_tree = convert_it_vit.raw_tree(con_sentence[1])
            # the moveprune feature requires corenlp 4.5.0 or later
            updated_tree, _ = convert_it_vit.update_mwts_and_special_cases(con_tree, ud_sentence, mwt_map, tsurgeon_processor)
            assert str(updated_tree) == expected_tree
