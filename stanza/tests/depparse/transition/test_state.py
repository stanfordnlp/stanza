import pytest

from collections import Counter

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

from stanza.utils.conll import CoNLL
from stanza.models.depparse.data import DataLoader
from stanza.models.depparse.trainer import unpack_batch
from stanza.models.depparse.transition.state import from_gold, states_from_data_batch
from stanza.models.depparse.transition.transitions import ProjectiveRight, NonprojectiveRight, ProjectiveLeft, NonprojectiveLeft, Shift

sample_sentence = """
# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0006
# text = The third was being run by the head of an investment firm.
1	The	the	DET	DT	Definite=Def|PronType=Art	2	det	2:det	_
2	third	third	ADJ	JJ	Degree=Pos|NumForm=Word|NumType=Ord	5	nsubj:pass	5:nsubj:pass	_
3	was	be	AUX	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	5	aux	5:aux	_
4	being	be	AUX	VBG	Tense=Pres|VerbForm=Part	5	aux:pass	5:aux:pass	_
5	run	run	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	0	root	0:root	_
6	by	by	ADP	IN	_	8	case	8:case	_
7	the	the	DET	DT	Definite=Def|PronType=Art	8	det	8:det	_
8	head	head	NOUN	NN	Number=Sing	5	obl:agent	5:obl:agent	_
9	of	of	ADP	IN	_	12	case	12:case	_
10	an	a	DET	DT	Definite=Ind|PronType=Art	12	det	12:det	_
11	investment	investment	NOUN	NN	Number=Sing	12	compound	12:compound	_
12	firm	firm	NOUN	NN	Number=Sing	8	nmod	8:nmod:of	SpaceAfter=No
13	.	.	PUNCT	.	_	5	punct	5:punct	_
""".lstrip()

nonproj_sentence = """
# newdoc id = n01010
# sent_id = n01010042
# parallel_id = pud/n01010042
# text = There was a time, Mr Panvalkar said, when he felt that they should leave the building.
1	There	there	PRON	EX	_	2	expl	2:expl	_
2	was	be	VERB	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	0	root	0:root	_
3	a	a	DET	DT	Definite=Ind|PronType=Art	4	det	4:det	_
4	time	time	NOUN	NN	Number=Sing	2	nsubj	2:nsubj	SpaceAfter=No
5	,	,	PUNCT	,	_	8	punct	8:punct	_
6	Mr	Mr.	PROPN	NNP	Number=Sing	7	nmod:desc	7:nmod:desc	_
7	Panvalkar	Panvalkar	PROPN	NNP	Number=Sing	8	nsubj	8:nsubj	_
8	said	say	VERB	VBD	Mood=Ind|Tense=Past|VerbForm=Fin	2	parataxis	2:parataxis	SpaceAfter=No
9	,	,	PUNCT	,	_	8	punct	8:punct	_
10	when	when	ADV	WRB	PronType=Rel	12	advmod	12:advmod	_
11	he	he	PRON	PRP	Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs	12	nsubj	12:nsubj	_
12	felt	feel	VERB	VBD	Mood=Ind|Tense=Past|VerbForm=Fin	4	acl:relcl	4:acl:relcl	_
13	that	that	SCONJ	IN	_	16	mark	16:mark	_
14	they	they	PRON	PRP	Case=Nom|Number=Plur|Person=3|PronType=Prs	16	nsubj	16:nsubj	_
15	should	should	AUX	MD	VerbForm=Fin	16	aux	16:aux	_
16	leave	leave	VERB	VB	VerbForm=Inf	12	ccomp	12:ccomp	_
17	the	the	DET	DT	Definite=Def|PronType=Art	18	det	18:det	_
18	building	building	NOUN	NN	Number=Sing	16	obj	16:obj	SpaceAfter=No
19	.	.	PUNCT	.	_	2	punct	2:punct	_
""".lstrip()

nonproj_right_sentence = """
# sent_id = n01002058
# parallel_id = pud/n01002058
# text = What she’s saying and what she’s doing, it — actually, it’s unbelievable.
1	What	what	PRON	WP	PronType=Int	4	obj	4:obj	_
2	she	she	PRON	PRP	Case=Nom|Gender=Fem|Number=Sing|Person=3|PronType=Prs	4	nsubj	4:nsubj	SpaceAfter=No
3	’s	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	4	aux	4:aux	_
4	saying	say	VERB	VBG	VerbForm=Ger	17	dislocated	17:dislocated	_
5	and	and	CCONJ	CC	_	9	cc	9:cc	_
6	what	what	PRON	WP	PronType=Int	9	obj	9:obj	_
7	she	she	PRON	PRP	Case=Nom|Gender=Fem|Number=Sing|Person=3|PronType=Prs	9	nsubj	9:nsubj	SpaceAfter=No
8	’s	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	9	aux	9:aux	_
9	doing	do	VERB	VBG	VerbForm=Ger	4	conj	4:conj:and	SpaceAfter=No
10	,	,	PUNCT	,	_	17	punct	17:punct	_
11	it	it	PRON	PRP	Case=Nom|Gender=Neut|Number=Sing|Person=3|PronType=Prs	15	reparandum	15:reparandum	_
12	—	—	PUNCT	:	_	11	punct	11:punct	_
13	actually	actually	ADV	RB	_	17	advmod	17:advmod	SpaceAfter=No
14	,	,	PUNCT	,	_	13	punct	13:punct	_
15	it	it	PRON	PRP	Case=Nom|Gender=Neut|Number=Sing|Person=3|PronType=Prs	17	nsubj	17:nsubj	SpaceAfter=No
16	’s	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	17	cop	17:cop	_
17	unbelievable	unbelievable	ADJ	JJ	Degree=Pos	0	root	0:root	SpaceAfter=No
18	.	.	PUNCT	.	_	17	punct	17:punct	_
""".lstrip()

# the next two sentences both uncovered bugs in the non-projective implementation
difficult_nonproj_sentence = """
# sent_id = w01111093
# parallel_id = pud/w01111093
# text = Winstone was declared bankrupt on 4 October 1988 and again on 19 March 1993.
1	Winstone	Winstone	PROPN	NNP	Number=Sing	3	nsubj:pass	3:nsubj:pass|4:nsubj:xsubj|13:nsubj:pass	_
2	was	be	AUX	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	3	aux:pass	3:aux:pass	_
3	declared	declare	VERB	VBN	Tense=Past|VerbForm=Part	0	root	0:root	_
4	bankrupt	bankrupt	ADJ	JJ	Degree=Pos	3	xcomp	3:xcomp	_
5	on	on	ADP	IN	_	6	case	6:case	_
6	4	4	NUM	CD	NumForm=Digit|NumType=Card	3	obl	3:obl:on	_
7	October	October	PROPN	NNP	Number=Sing	6	nmod:unmarked	6:nmod:unmarked	_
8	1988	1988	NUM	CD	NumForm=Digit|NumType=Card	6	nmod:unmarked	6:nmod:unmarked	_
9	and	and	CCONJ	CC	_	13	cc	13:cc	_
10	again	again	ADV	RB	_	13	advmod	13:advmod	_
11	on	on	ADP	IN	_	12	case	12:case	_
12	19	19	NUM	CD	NumForm=Digit|NumType=Card	3	conj	3:conj:and	_
13	March	March	PROPN	NNP	Number=Sing	12	nmod:unmarked	12:nmod:unmarked	_
14	1993	1993	NUM	CD	NumForm=Digit|NumType=Card	12	nmod:unmarked	12:nmod:unmarked	SpaceAfter=No
15	.	.	PUNCT	.	_	3	punct	3:punct	_
""".lstrip()

difficult_nonproj_p2_sentence = """
# sent_id = w01053067
# parallel_id = pud/w01053067
# text = It is possible to establish the phase of the moon on a particular day two thousand years ago but not whether it was obscured by clouds or haze.
1	It	it	PRON	PRP	Case=Nom|Gender=Neut|Number=Sing|Person=3|PronType=Prs	3	expl	3:expl	_
2	is	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	3	cop	3:cop	_
3	possible	possible	ADJ	JJ	Degree=Pos	0	root	0:root	_
4	to	to	PART	TO	_	5	mark	5:mark	_
5	establish	establish	VERB	VB	VerbForm=Inf	3	csubj	3:csubj	_
6	the	the	DET	DT	Definite=Def|PronType=Art	7	det	7:det	_
7	phase	phase	NOUN	NN	Number=Sing	5	obj	5:obj	_
8	of	of	ADP	IN	_	10	case	10:case	_
9	the	the	DET	DT	Definite=Def|PronType=Art	10	det	10:det	_
10	moon	moon	NOUN	NN	Number=Sing	7	nmod	7:nmod:of	_
11	on	on	ADP	IN	_	14	case	14:case	_
12	a	a	DET	DT	Definite=Ind|PronType=Art	14	det	14:det	_
13	particular	particular	ADJ	JJ	Degree=Pos	14	amod	14:amod	_
14	day	day	NOUN	NN	Number=Sing	5	obl	5:obl:on	_
15	two	two	NUM	CD	NumForm=Word|NumType=Card	16	compound	16:compound	_
16	thousand	thousand	NUM	CD	NumForm=Word|NumType=Card	17	nummod	17:nummod	_
17	years	year	NOUN	NNS	Number=Plur	18	obl:unmarked	18:obl:unmarked	_
18	ago	ago	ADV	RB	_	14	advmod	14:advmod	_
19	but	but	CCONJ	CC	_	24	cc	24:cc	_
20	not	not	PART	RB	Polarity=Neg	24	advmod	24:advmod	_
21	whether	whether	SCONJ	IN	_	24	mark	24:mark	_
22	it	it	PRON	PRP	Case=Nom|Gender=Neut|Number=Sing|Person=3|PronType=Prs	24	nsubj:pass	24:nsubj:pass	_
23	was	be	AUX	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	24	aux:pass	24:aux:pass	_
24	obscured	obscure	VERB	VBN	Tense=Past|VerbForm=Part	7	conj	5:obj|7:conj:but	_
25	by	by	ADP	IN	_	26	case	26:case	_
26	clouds	cloud	NOUN	NNS	Number=Plur	24	obl	24:obl:by	_
27	or	or	CCONJ	CC	_	28	cc	28:cc	_
28	haze	haze	NOUN	NN	Number=Sing	26	conj	24:obl:by|26:conj:or	SpaceAfter=No
29	.	.	PUNCT	.	_	3	punct	3:punct	_
""".lstrip()

difficult_legal_transitions_sentence = """
# sent_id = email-enronsent41_01-0038
# text = F.O.B. refers to the following terms: Seller is responsible for freight, unloading and storage up to and including delivery in warehouse, Buyer is responsible for storage, loading and freight after delivery.
1	F.O.B.	F.O.B.	PROPN	NNP	Number=Sing	2	nsubj	2:nsubj	_
2	refers	refer	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	0:root	_
3	to	to	ADP	IN	_	6	case	6:case	_
4	the	the	DET	DT	Definite=Def|PronType=Art	6	det	6:det	_
5	following	follow	VERB	VBG	VerbForm=Ger	6	amod	6:amod	_
6	terms	term	NOUN	NNS	Number=Plur	2	obl	2:obl:to	SpaceAfter=No
7	:	:	PUNCT	:	_	10	punct	10:punct	_
8	Seller	seller	NOUN	NN	Number=Sing	10	nsubj	10:nsubj	_
9	is	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	10	cop	10:cop	_
10	responsible	responsible	ADJ	JJ	Degree=Pos	2	parataxis	2:parataxis	_
11	for	for	ADP	IN	_	12	case	12:case	_
12	freight	freight	NOUN	NN	Number=Sing	10	obl	10:obl:for	SpaceAfter=No
13	,	,	PUNCT	,	_	14	punct	14:punct	_
14	unloading	unloading	NOUN	NN	Number=Sing	12	conj	10:obl:for|12:conj:and	_
15	and	and	CCONJ	CC	_	16	cc	16:cc	_
16	storage	storage	NOUN	NN	Number=Sing	12	conj	10:obl:for|12:conj:and	_
17	up	up	ADP	IN	_	21	case	21:case	_
18	to	to	ADP	IN	_	21	case	21:case	_
19	and	and	CCONJ	CC	_	20	cc	20:cc	_
20	including	include	VERB	VBG	Tense=Pres|VerbForm=Part	17	conj	17:conj:and|21:case	_
21	delivery	delivery	NOUN	NN	Number=Sing	10	obl	10:obl:up_to	_
22	in	in	ADP	IN	_	23	case	23:case	_
23	warehouse	warehouse	NOUN	NN	Number=Sing	21	nmod	21:nmod:in	SpaceAfter=No
24	,	,	PUNCT	,	_	27	punct	27:punct	_
25	Buyer	buyer	NOUN	NN	Number=Sing	27	nsubj	27:nsubj	_
26	is	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	27	cop	27:cop	_
27	responsible	responsible	ADJ	JJ	Degree=Pos	10	parataxis	10:parataxis	_
28	for	for	ADP	IN	_	29	case	29:case	_
29	storage	storage	NOUN	NN	Number=Sing	27	obl	27:obl:for	SpaceAfter=No
30	,	,	PUNCT	,	_	31	punct	31:punct	_
31	loading	loading	NOUN	NN	Number=Sing	29	conj	27:obl:for|29:conj:and	_
32	and	and	CCONJ	CC	_	33	cc	33:cc	_
33	freight	freight	NOUN	NN	Number=Sing	29	conj	27:obl:for|29:conj:and	_
34	after	after	ADP	IN	_	35	case	35:case	_
35	delivery	delivery	NOUN	NN	Number=Sing	27	obl	27:obl:after	SpaceAfter=No
36	.	.	PUNCT	.	_	2	punct	2:punct	_
""".lstrip()

# this sentence uncovered an error parsing in reverse
difficult_reversed_sentence = """
# sent_id = answers-20111107175720AAlb2TB_ans-0026
# text = Waterproof clothing and footwear are essential plus an umbrella..
1	Waterproof	waterproof	ADJ	JJ	Degree=Pos	2	amod	2:amod	_
2	clothing	clothing	NOUN	NN	Number=Sing	6	nsubj	6:nsubj	_
3	and	and	CCONJ	CC	_	4	cc	4:cc	_
4	footwear	footwear	NOUN	NN	Number=Sing	2	conj	2:conj:and|6:nsubj	_
5	are	be	AUX	VBP	Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin	6	cop	6:cop	_
6	essential	essential	ADJ	JJ	Degree=Pos	0	root	0:root	_
7	plus	plus	CCONJ	CC	_	9	cc	9:cc	_
8	an	a	DET	DT	Definite=Ind|PronType=Art	9	det	9:det	_
9	umbrella	umbrella	NOUN	NN	Number=Sing	4	conj	4:conj:plus	SpaceAfter=No
10	..	..	PUNCT	.	_	6	punct	6:punct	_
""".lstrip()


def check_rebuilt_graph(state):
    gold_sequence = state.gold_sequence
    for transition in gold_sequence:
        assert transition.is_legal(state)
        state = transition.apply(state)
    assert state.gold_graph.nodes() == state.parsed_graph.nodes()
    assert sorted(state.gold_graph.edges(data=True)) == sorted(state.parsed_graph.edges(data=True))
    assert state.transitions == state.gold_sequence

    for transition in gold_sequence:
        assert not transition.is_legal(state)

def test_build_basic_state():
    sample_doc = CoNLL.conll2doc(input_str=sample_sentence)
    assert len(sample_doc.sentences) == 1

    state = from_gold(sample_doc.sentences[0])
    check_rebuilt_graph(state)

def test_build_nonproj_state():
    sample_doc = CoNLL.conll2doc(input_str=nonproj_sentence)
    assert len(sample_doc.sentences) == 1

    state = from_gold(sample_doc.sentences[0])
    check_rebuilt_graph(state)

def test_difficult_nonproj():
    sample_doc = CoNLL.conll2doc(input_str=difficult_nonproj_sentence)
    assert len(sample_doc.sentences) == 1

    state = from_gold(sample_doc.sentences[0])
    check_rebuilt_graph(state)

def test_difficult_nonproj_p2():
    sample_doc = CoNLL.conll2doc(input_str=difficult_nonproj_p2_sentence)
    assert len(sample_doc.sentences) == 1

    state = from_gold(sample_doc.sentences[0])
    check_rebuilt_graph(state)

def test_difficult_transition():
    sample_doc = CoNLL.conll2doc(input_str=difficult_legal_transitions_sentence)
    assert len(sample_doc.sentences) == 1

    state = from_gold(sample_doc.sentences[0])
    check_rebuilt_graph(state)

def test_reversed():
    """
    Test the conversion in both the regular and reversed direction for this sentence using the DataLoader
    """
    batch_size = 1
    device = "cpu"

    sample_doc = CoNLL.conll2doc(input_str=difficult_reversed_sentence)

    # test in the regular direction
    args = {"shorthand": "en"}
    data = DataLoader(sample_doc, batch_size, args, None)
    for batch in data:
        inputs, orig_idx, word_orig_idx, sentlens, wordlens, text = unpack_batch(batch, device)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel = inputs

        states = states_from_data_batch(data.vocab['deprel'], head, deprel, text, sentlens)
        for state in states:
            check_rebuilt_graph(state)

    # test in the reversed direction
    args = {"shorthand": "en", "reversed": True}
    data = DataLoader(sample_doc, batch_size, args, None)
    for batch in data:
        inputs, orig_idx, word_orig_idx, sentlens, wordlens, text = unpack_batch(batch, device)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel = inputs

        states = states_from_data_batch(data.vocab['deprel'], head, deprel, text, sentlens)
        for state in states:
            check_rebuilt_graph(state)

def ztest_pud():
    doc = CoNLL.conll2doc("../data/ud2/UD_English-PUD/en_pud-ud-test.conllu")
    for sentence in doc.sentences:
        try:
            state = from_gold(sentence)
        except AssertionError:
            print(sentence.text)

def ztest_ewt():
    doc = CoNLL.conll2doc("extern_data/ud2/git/UD_English-EWT/en_ewt-ud-train.conllu")
    transition_counts = Counter()
    for sentence in doc.sentences:
        try:
            state = from_gold(sentence)
        except AssertionError:
            print(sentence.text)

        transitions = state.gold_sequence

        # sanity check - some bug in the parser indexing code a while back
        for t in transitions:
            if isinstance(t, NonprojectiveLeft):
                assert t.word_idx != 0

        transitions = [x.simplify() for x in transitions]
        transition_counts.update(transitions)

        try:
            check_rebuilt_graph(state)
        except:
            print(sentence)
            raise
    for key, count in transition_counts.most_common():
        print("  %s: %d" % (key, count))

def test_nonproj_left_illegal():
    """
    Test that a nonprojective left going to a node in its own subtree is illegal
    """
    sample_doc = CoNLL.conll2doc(input_str=nonproj_sentence)
    assert len(sample_doc.sentences) == 1

    state = from_gold(sample_doc.sentences[0])
    for npl_idx, npl_transition in enumerate(state.gold_sequence):
        if isinstance(npl_transition, NonprojectiveLeft):
            break
    else:
        raise AssertionError("This sequence should have a NonprojectiveLeft")

    gold_sequence = state.gold_sequence
    for transition in gold_sequence[:npl_idx]:
        assert transition.is_legal(state)
        state = transition.apply(state)
    assert len(state.transitions) == npl_idx

    assert npl_transition.is_legal(state)
    fake_transition = NonprojectiveLeft(npl_transition.deprel, state.current_heads[-1])
    assert not fake_transition.is_legal(state)

    num_shallow = 0
    num_deep = 0
    for pred in state.parsed_graph.predecessors(state.current_heads[-1]):
        fake_transition = NonprojectiveLeft(npl_transition.deprel, pred)
        assert not fake_transition.is_legal(state)
        num_shallow += 1

        for pred2 in state.parsed_graph.predecessors(pred):
            fake_transition = NonprojectiveLeft(npl_transition.deprel, pred2)
            assert not fake_transition.is_legal(state)
            num_deep += 1

    # check that we actually tested some of the fake transitions
    assert num_shallow > 0
    assert num_deep > 0
