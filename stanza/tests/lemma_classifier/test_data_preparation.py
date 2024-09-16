import os

import pytest

import stanza.models.lemma_classifier.utils as utils
import stanza.utils.datasets.prepare_lemma_classifier as prepare_lemma_classifier

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

EWT_ONE_SENTENCE = """
# sent_id = weblog-blogspot.com_rigorousintuition_20050518101500_ENG_20050518_101500-0002
# newpar id = weblog-blogspot.com_rigorousintuition_20050518101500_ENG_20050518_101500-p0002
# text = Here's a Miami Herald interview
1-2	Here's	_	_	_	_	_	_	_	_
1	Here	here	ADV	RB	PronType=Dem	0	root	0:root	_
2	's	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	1	cop	1:cop	_
3	a	a	DET	DT	Definite=Ind|PronType=Art	6	det	6:det	_
4	Miami	Miami	PROPN	NNP	Number=Sing	5	compound	5:compound	_
5	Herald	Herald	PROPN	NNP	Number=Sing	6	compound	6:compound	_
6	interview	interview	NOUN	NN	Number=Sing	1	nsubj	1:nsubj	_
""".lstrip()


EWT_TRAIN_SENTENCES = """
# sent_id = weblog-blogspot.com_rigorousintuition_20050518101500_ENG_20050518_101500-0002
# newpar id = weblog-blogspot.com_rigorousintuition_20050518101500_ENG_20050518_101500-p0002
# text = Here's a Miami Herald interview
1-2	Here's	_	_	_	_	_	_	_	_
1	Here	here	ADV	RB	PronType=Dem	0	root	0:root	_
2	's	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	1	cop	1:cop	_
3	a	a	DET	DT	Definite=Ind|PronType=Art	6	det	6:det	_
4	Miami	Miami	PROPN	NNP	Number=Sing	5	compound	5:compound	_
5	Herald	Herald	PROPN	NNP	Number=Sing	6	compound	6:compound	_
6	interview	interview	NOUN	NN	Number=Sing	1	nsubj	1:nsubj	_

# sent_id = weblog-blogspot.com_rigorousintuition_20050518101500_ENG_20050518_101500-0027
# text = But Posada's nearly 80 years old
1	But	but	CCONJ	CC	_	7	cc	7:cc	_
2-3	Posada's	_	_	_	_	_	_	_	_
2	Posada	Posada	PROPN	NNP	Number=Sing	7	nsubj	7:nsubj	_
3	's	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	7	cop	7:cop	_
4	nearly	nearly	ADV	RB	_	5	advmod	5:advmod	_
5	80	80	NUM	CD	NumForm=Digit|NumType=Card	6	nummod	6:nummod	_
6	years	year	NOUN	NNS	Number=Plur	7	obl:npmod	7:obl:npmod	_
7	old	old	ADJ	JJ	Degree=Pos	0	root	0:root	SpaceAfter=No

# sent_id = weblog-blogspot.com_rigorousintuition_20050518101500_ENG_20050518_101500-0067
# newpar id = weblog-blogspot.com_rigorousintuition_20050518101500_ENG_20050518_101500-p0011
# text = Now that's a post I can relate to.
1	Now	now	ADV	RB	_	5	advmod	5:advmod	_
2-3	that's	_	_	_	_	_	_	_	_
2	that	that	PRON	DT	Number=Sing|PronType=Dem	5	nsubj	5:nsubj	_
3	's	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	5	cop	5:cop	_
4	a	a	DET	DT	Definite=Ind|PronType=Art	5	det	5:det	_
5	post	post	NOUN	NN	Number=Sing	0	root	0:root	_
6	I	I	PRON	PRP	Case=Nom|Number=Sing|Person=1|PronType=Prs	8	nsubj	8:nsubj	_
7	can	can	AUX	MD	VerbForm=Fin	8	aux	8:aux	_
8	relate	relate	VERB	VB	VerbForm=Inf	5	acl:relcl	5:acl:relcl	_
9	to	to	ADP	IN	_	8	obl	8:obl	SpaceAfter=No
10	.	.	PUNCT	.	_	5	punct	5:punct	_

# sent_id = weblog-blogspot.com_rigorousintuition_20050518101500_ENG_20050518_101500-0073
# newpar id = weblog-blogspot.com_rigorousintuition_20050518101500_ENG_20050518_101500-p0012
# text = hey that's a great blog
1	hey	hey	INTJ	UH	_	6	discourse	6:discourse	_
2-3	that's	_	_	_	_	_	_	_	_
2	that	that	PRON	DT	Number=Sing|PronType=Dem	6	nsubj	6:nsubj	_
3	's	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	6	cop	6:cop	_
4	a	a	DET	DT	Definite=Ind|PronType=Art	6	det	6:det	_
5	great	great	ADJ	JJ	Degree=Pos	6	amod	6:amod	_
6	blog	blog	NOUN	NN	Number=Sing	0	root	0:root	SpaceAfter=No

# sent_id = weblog-blogspot.com_rigorousintuition_20050518101500_ENG_20050518_101500-0089
# text = And It's Not Hard To Do
1	And	and	CCONJ	CC	_	5	cc	5:cc	_
2-3	It's	_	_	_	_	_	_	_	_
2	It	it	PRON	PRP	Case=Nom|Gender=Neut|Number=Sing|Person=3|PronType=Prs	5	expl	5:expl	_
3	's	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	5	cop	5:cop	_
4	Not	not	PART	RB	_	5	advmod	5:advmod	_
5	Hard	hard	ADJ	JJ	Degree=Pos	0	root	0:root	_
6	To	to	PART	TO	_	7	mark	7:mark	_
7	Do	do	VERB	VB	VerbForm=Inf	5	csubj	5:csubj	SpaceAfter=No

# sent_id = weblog-blogspot.com_rigorousintuition_20060511134300_ENG_20060511_134300-0029
# text = Meanwhile, a decision's been reached
1	Meanwhile	meanwhile	ADV	RB	_	7	advmod	7:advmod	SpaceAfter=No
2	,	,	PUNCT	,	_	1	punct	1:punct	_
3	a	a	DET	DT	Definite=Ind|PronType=Art	4	det	4:det	_
4-5	decision's	_	_	_	_	_	_	_	_
4	decision	decision	NOUN	NN	Number=Sing	7	nsubj:pass	7:nsubj:pass	_
5	's	have	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	7	aux	7:aux	_
6	been	be	AUX	VBN	Tense=Past|VerbForm=Part	7	aux:pass	7:aux:pass	_
7	reached	reach	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	0	root	0:root	_

# sent_id = weblog-blogspot.com_rigorousintuition_20060511134300_ENG_20060511_134300-0138
# text = It's become a guardian of morality
1-2	It's	_	_	_	_	_	_	_	_
1	It	it	PRON	PRP	Case=Nom|Gender=Neut|Number=Sing|Person=3|PronType=Prs	3	nsubj	3:nsubj|5:nsubj:xsubj	_
2	's	have	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	3	aux	3:aux	_
3	become	become	VERB	VBN	Tense=Past|VerbForm=Part	0	root	0:root	_
4	a	a	DET	DT	Definite=Ind|PronType=Art	5	det	5:det	_
5	guardian	guardian	NOUN	NN	Number=Sing	3	xcomp	3:xcomp	_
6	of	of	ADP	IN	_	7	case	7:case	_
7	morality	morality	NOUN	NN	Number=Sing	5	nmod	5:nmod:of	_

# sent_id = email-enronsent15_01-0018
# text = It's got its own bathroom and tv
1-2	It's	_	_	_	_	_	_	_	_
1	It	it	PRON	PRP	Case=Nom|Gender=Neut|Number=Sing|Person=3|PronType=Prs	3	nsubj	3:nsubj|13:nsubj	_
2	's	have	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	3	aux	3:aux	_
3	got	get	VERB	VBN	Tense=Past|VerbForm=Part	0	root	0:root	_
4	its	its	PRON	PRP$	Case=Gen|Gender=Neut|Number=Sing|Person=3|Poss=Yes|PronType=Prs	6	nmod:poss	6:nmod:poss	_
5	own	own	ADJ	JJ	Degree=Pos	6	amod	6:amod	_
6	bathroom	bathroom	NOUN	NN	Number=Sing	3	obj	3:obj	_
7	and	and	CCONJ	CC	_	8	cc	8:cc	_
8	tv	TV	NOUN	NN	Number=Sing	6	conj	3:obj|6:conj:and	SpaceAfter=No

# sent_id = newsgroup-groups.google.com_alt.animals.cat_01ff709c4bf2c60c_ENG_20040418_040100-0022
# text = It's also got the website
1-2	It's	_	_	_	_	_	_	_	_
1	It	it	PRON	PRP	Case=Nom|Gender=Neut|Number=Sing|Person=3|PronType=Prs	4	nsubj	4:nsubj	_
2	's	have	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	4	aux	4:aux	_
3	also	also	ADV	RB	_	4	advmod	4:advmod	_
4	got	get	VERB	VBN	Tense=Past|VerbForm=Part	0	root	0:root	_
5	the	the	DET	DT	Definite=Def|PronType=Art	6	det	6:det	_
6	website	website	NOUN	NN	Number=Sing	4	obj	4:obj|12:obl	_
""".lstrip()


# from the train set, actually
EWT_DEV_SENTENCES = """
# sent_id = answers-20111108104724AAuBUR7_ans-0044
# text = He's only exhibited weight loss and some muscle atrophy
1-2	He's	_	_	_	_	_	_	_	_
1	He	he	PRON	PRP	Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs	4	nsubj	4:nsubj	_
2	's	have	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	4	aux	4:aux	_
3	only	only	ADV	RB	_	4	advmod	4:advmod	_
4	exhibited	exhibit	VERB	VBN	Tense=Past|VerbForm=Part	0	root	0:root	_
5	weight	weight	NOUN	NN	Number=Sing	6	compound	6:compound	_
6	loss	loss	NOUN	NN	Number=Sing	4	obj	4:obj	_
7	and	and	CCONJ	CC	_	10	cc	10:cc	_
8	some	some	DET	DT	PronType=Ind	10	det	10:det	_
9	muscle	muscle	NOUN	NN	Number=Sing	10	compound	10:compound	_
10	atrophy	atrophy	NOUN	NN	Number=Sing	6	conj	4:obj|6:conj:and	SpaceAfter=No

# sent_id = weblog-blogspot.com_rigorousintuition_20060511134300_ENG_20060511_134300-0097
# text = It's a good thing too.
1-2	It's	_	_	_	_	_	_	_	_
1	It	it	PRON	PRP	Case=Nom|Gender=Neut|Number=Sing|Person=3|PronType=Prs	5	nsubj	5:nsubj	_
2	's	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	5	cop	5:cop	_
3	a	a	DET	DT	Definite=Ind|PronType=Art	5	det	5:det	_
4	good	good	ADJ	JJ	Degree=Pos	5	amod	5:amod	_
5	thing	thing	NOUN	NN	Number=Sing	0	root	0:root	_
6	too	too	ADV	RB	_	5	advmod	5:advmod	SpaceAfter=No
7	.	.	PUNCT	.	_	5	punct	5:punct	_
""".lstrip()

# from the train set, actually
EWT_TEST_SENTENCES = """
# sent_id = reviews-162422-0015
# text = He said he's had a long and bad day.
1	He	he	PRON	PRP	Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs	2	nsubj	2:nsubj	_
2	said	say	VERB	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	0	root	0:root	_
3-4	he's	_	_	_	_	_	_	_	_
3	he	he	PRON	PRP	Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs	5	nsubj	5:nsubj	_
4	's	have	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	5	aux	5:aux	_
5	had	have	VERB	VBN	Tense=Past|VerbForm=Part	2	ccomp	2:ccomp	_
6	a	a	DET	DT	Definite=Ind|PronType=Art	10	det	10:det	_
7	long	long	ADJ	JJ	Degree=Pos	10	amod	10:amod	_
8	and	and	CCONJ	CC	_	9	cc	9:cc	_
9	bad	bad	ADJ	JJ	Degree=Pos	7	conj	7:conj:and|10:amod	_
10	day	day	NOUN	NN	Number=Sing	5	obj	5:obj	SpaceAfter=No
11	.	.	PUNCT	.	_	2	punct	2:punct	_

# sent_id = weblog-blogspot.com_rigorousintuition_20060511134300_ENG_20060511_134300-0100
# text = What's a few dead soldiers
1-2	What's	_	_	_	_	_	_	_	_
1	What	what	PRON	WP	PronType=Int	6	nsubj	6:nsubj	_
2	's	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	6	cop	6:cop	_
3	a	a	DET	DT	Definite=Ind|PronType=Art	6	det	6:det	_
4	few	few	ADJ	JJ	Degree=Pos	6	amod	6:amod	_
5	dead	dead	ADJ	JJ	Degree=Pos	6	amod	6:amod	_
6	soldiers	soldier	NOUN	NNS	Number=Plur	0	root	0:root	_
"""

def write_test_dataset(tmp_path, texts, datasets):
    ud_path = tmp_path / "ud"
    input_path = ud_path / "UD_English-EWT"
    output_path = tmp_path / "data" / "lemma_classifier"

    os.makedirs(input_path, exist_ok=True)

    for text, dataset in zip(texts, datasets):
        sample_file = input_path / ("en_ewt-ud-%s.conllu" % dataset)
        with open(sample_file, "w", encoding="utf-8") as fout:
            fout.write(text)

    paths = {"UDBASE": ud_path,
             "LEMMA_CLASSIFIER_DATA_DIR": output_path}

    return paths

def write_english_test_dataset(tmp_path):
    texts = (EWT_TRAIN_SENTENCES, EWT_DEV_SENTENCES, EWT_TEST_SENTENCES)
    datasets = prepare_lemma_classifier.SECTIONS
    return write_test_dataset(tmp_path, texts, datasets)

def convert_english_dataset(tmp_path):
    paths = write_english_test_dataset(tmp_path)
    converted_files = prepare_lemma_classifier.process_treebank(paths, "en_ewt", "'s", "AUX", "be|have")
    assert len(converted_files) == 3

    return converted_files

def test_convert_one_sentence(tmp_path):
    texts = [EWT_ONE_SENTENCE]
    datasets = ["train"]
    paths = write_test_dataset(tmp_path, texts, datasets)

    converted_files = prepare_lemma_classifier.process_treebank(paths, "en_ewt", "'s", "AUX", "be|have", ["train"])
    assert len(converted_files) == 1

    dataset = utils.Dataset(converted_files[0], get_counts=True, batch_size=10, shuffle=False)

    assert len(dataset) == 1
    assert dataset.label_decoder == {'be': 0}
    id_to_upos = {y: x for x, y in dataset.upos_to_id.items()}

    for text_batches, _, upos_batches, _ in dataset:
        assert text_batches == [['Here', "'s", 'a', 'Miami', 'Herald', 'interview']]
        upos = [id_to_upos[x] for x in upos_batches[0]]
        assert upos == ['ADV', 'AUX', 'DET', 'PROPN', 'PROPN', 'NOUN']

def test_convert_dataset(tmp_path):
    converted_files = convert_english_dataset(tmp_path)

    dataset = utils.Dataset(converted_files[0], get_counts=True, batch_size=10, shuffle=False)

    assert len(dataset) == 1
    label_decoder = dataset.label_decoder
    assert len(label_decoder) == 2
    assert "be" in label_decoder
    assert "have" in label_decoder
    for text_batches, _, _, _ in dataset:
        assert len(text_batches) == 9

    dataset = utils.Dataset(converted_files[1], get_counts=True, batch_size=10, shuffle=False)
    assert len(dataset) == 1
    for text_batches, _, _, _ in dataset:
        assert len(text_batches) == 2

    dataset = utils.Dataset(converted_files[2], get_counts=True, batch_size=10, shuffle=False)
    assert len(dataset) == 1
    for text_batches, _, _, _ in dataset:
        assert len(text_batches) == 2

