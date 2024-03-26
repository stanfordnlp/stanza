"""
A few tests of specific operations from the Dataset
"""

import os
import pytest

from stanza.models.common.doc import *
from stanza.models import tagger
from stanza.models.pos.data import Dataset, ShuffledDataset
from stanza.utils.conll import CoNLL

from stanza.tests.pos.test_tagger import TRAIN_DATA, TRAIN_DATA_NO_XPOS, TRAIN_DATA_NO_UPOS, TRAIN_DATA_NO_FEATS

def test_basic_reading():
    """
    Test that a dataset with no xpos is detected by the Dataset
    """
    # empty args for building the data object
    args = tagger.parse_args(args=[])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA)

    data = Dataset(train_doc, args, None)
    assert data.has_upos
    assert data.has_xpos
    assert data.has_feats

def test_no_xpos():
    """
    Test that a dataset with no xpos is detected by the Dataset
    """
    # empty args for building the data object
    args = tagger.parse_args(args=[])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA_NO_XPOS)

    data = Dataset(train_doc, args, None)
    assert data.has_upos
    assert not data.has_xpos
    assert data.has_feats

def test_no_upos():
    """
    Test that a dataset with no upos is detected by the Dataset
    """
    # empty args for building the data object
    args = tagger.parse_args(args=[])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA_NO_UPOS)

    data = Dataset(train_doc, args, None)
    assert not data.has_upos
    assert data.has_xpos
    assert data.has_feats

def test_no_feats():
    """
    Test that a dataset with no feats is detected by the Dataset
    """
    # empty args for building the data object
    args = tagger.parse_args(args=[])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA_NO_FEATS)

    data = Dataset(train_doc, args, None)
    assert data.has_upos
    assert data.has_xpos
    assert not data.has_feats

def test_no_augment():
    """
    Test that with no punct removing augmentation, the doc always has punct at the end
    """
    args = tagger.parse_args(args=["--shorthand", "en_test", "--augment_nopunct", "0.0"])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA)
    data = Dataset(train_doc, args, None)
    data = data.to_loader(batch_size=2)

    for i in range(50):
        for batch in data:
            for text in batch.text:
                assert text[-1] in (".", "!")

def test_augment():
    """
    Test that with 100% punct removing augmentation, the doc never has punct at the end
    """
    args = tagger.parse_args(args=["--shorthand", "en_test", "--augment_nopunct", "1.0"])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA)
    data = Dataset(train_doc, args, None)
    data = data.to_loader(batch_size=2)

    for i in range(50):
        for batch in data:
            for text in batch.text:
                assert text[-1] not in (".", "!")

def test_sometimes_augment():
    """
    Test 50% punct removing augmentation

    With this frequency, we should get a reasonable number of docs
    with a punct at the end and a reasonable without.
    """
    args = tagger.parse_args(args=["--shorthand", "en_test", "--augment_nopunct", "0.5"])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA)
    data = Dataset(train_doc, args, None)
    data = data.to_loader(batch_size=2)

    count_with = 0
    count_without = 0
    for i in range(50):
        for batch in data:
            for text in batch.text:
                if text[-1] in (".", "!"):
                    count_with += 1
                else:
                    count_without += 1

    # this should never happen
    # literally less than 1 in 10^20th odds
    assert count_with > 5
    assert count_without > 5


NO_XPOS_TEMPLATE = """
# text = Noxpos {indexp}
# sent_id = {index}
1	Noxpos	noxpos	NOUN	_	Number=Sing	0	root	_	start_char=0|end_char=8|ner=O
2	{indexp}	{indexp}	NUM	_	NumForm=Digit|NumType=Card	1	dep	_	start_char=9|end_char=10|ner=S-CARDINAL
""".strip()

YES_XPOS_TEMPLATE = """
# text = Yesxpos {indexp}
# sent_id = {index}
1	Yesxpos	yesxpos	NOUN	NN	Number=Sing	0	root	_	start_char=0|end_char=8|ner=O
2	{indexp}	{indexp}	NUM	CD	NumForm=Digit|NumType=Card	1	dep	_	start_char=9|end_char=10|ner=S-CARDINAL
""".strip()

def test_shuffle(tmp_path):
    args = tagger.parse_args(args=["--batch_size", "10", "--shorthand", "en_test", "--augment_nopunct", "0.0"])

    # 100 looked nice but was actually a 1/1000000 chance of the test failing
    # so let's crank it up to 1000 and make it 1/10^58
    no_xpos = [NO_XPOS_TEMPLATE.format(index=idx, indexp=idx+1) for idx in range(1000)]
    no_doc = CoNLL.conll2doc(input_str="\n\n".join(no_xpos))
    no_data = Dataset(no_doc, args, None)

    yes_xpos = [YES_XPOS_TEMPLATE.format(index=idx, indexp=idx+101) for idx in range(1000)]
    yes_doc = CoNLL.conll2doc(input_str="\n\n".join(yes_xpos))
    yes_data = Dataset(yes_doc, args, None)

    shuffled = ShuffledDataset([no_data, yes_data], 10)

    assert sum(1 for _ in shuffled) == 200

    num_with = 0
    num_without = 0
    for batch in shuffled:
        if batch.xpos is not None:
            num_with += 1
        else:
            num_without += 1
        # at the halfway point of the iteration, there should be at
        # least one in each category
        # for example, if we had forgotten to shuffle, this assertion would fail
        if num_with + num_without == 100:
            assert num_with > 1
            assert num_without > 1

    assert num_with == 100
    assert num_without == 100


EWT_SAMPLE = """
# sent_id = weblog-blogspot.com_alaindewitt_20040929103700_ENG_20040929_103700-0048
# text = Bush asked for permission to go to Alabama to work on a Senate campaign.
1	Bush	Bush	PROPN	NNP	Number=Sing	2	nsubj	2:nsubj	_
2	asked	ask	VERB	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	0	root	0:root	_
3	for	for	ADP	IN	_	4	case	4:case	_
4	permission	permission	NOUN	NN	Number=Sing	2	obl	2:obl:for	_
5	to	to	PART	TO	_	6	mark	6:mark	_
6	go	go	VERB	VB	VerbForm=Inf	4	acl	4:acl:to	_
7	to	to	ADP	IN	_	8	case	8:case	_
8	Alabama	Alabama	PROPN	NNP	Number=Sing	6	obl	6:obl:to	_
9	to	to	PART	TO	_	10	mark	10:mark	_
10	work	work	VERB	VB	VerbForm=Inf	6	advcl	6:advcl:to	_
11	on	on	ADP	IN	_	14	case	14:case	_
12	a	a	DET	DT	Definite=Ind|PronType=Art	14	det	14:det	_
13	Senate	Senate	PROPN	NNP	Number=Sing	14	compound	14:compound	_
14	campaign	campaign	NOUN	NN	Number=Sing	10	obl	10:obl:on	SpaceAfter=No
15	.	.	PUNCT	.	_	2	punct	2:punct	_

# sent_id = weblog-blogspot.com_alaindewitt_20040929103700_ENG_20040929_103700-0049
# text = His superior officers said OK.
1	His	his	PRON	PRP$	Case=Gen|Gender=Masc|Number=Sing|Person=3|Poss=Yes|PronType=Prs	3	nmod:poss	3:nmod:poss	_
2	superior	superior	ADJ	JJ	Degree=Pos	3	amod	3:amod	_
3	officers	officer	NOUN	NNS	Number=Plur	4	nsubj	4:nsubj	_
4	said	say	VERB	VBD	Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin	0	root	0:root	_
5	OK	ok	INTJ	UH	_	4	obj	4:obj	SpaceAfter=No
6	.	.	PUNCT	.	_	4	punct	4:punct	_

# sent_id = weblog-blogspot.com_alaindewitt_20040929103700_ENG_20040929_103700-0053
# text = In ’72 or ’73, if you were a pilot, active or Guard, and you had an obligation and wanted to get out, no problem.
1	In	in	ADP	IN	_	2	case	2:case	_
2	’72	'72	NUM	CD	NumForm=Digit|NumType=Card	10	obl	10:obl:in	_
3	or	or	CCONJ	CC	_	4	cc	4:cc	_
4	’73	'73	NUM	CD	NumForm=Digit|NumType=Card	2	conj	2:conj:or|10:obl:in	SpaceAfter=No
5	,	,	PUNCT	,	_	2	punct	2:punct	_
6	if	if	SCONJ	IN	_	10	mark	10:mark	_
7	you	you	PRON	PRP	Case=Nom|Person=2|PronType=Prs	10	nsubj	10:nsubj	_
8	were	be	AUX	VBD	Mood=Ind|Number=Sing|Person=2|Tense=Past|VerbForm=Fin	10	cop	10:cop	_
9	a	a	DET	DT	Definite=Ind|PronType=Art	10	det	10:det	_
10	pilot	pilot	NOUN	NN	Number=Sing	28	advcl	28:advcl:if	SpaceAfter=No
11	,	,	PUNCT	,	_	12	punct	12:punct	_
12	active	active	ADJ	JJ	Degree=Pos	10	amod	10:amod	_
13	or	or	CCONJ	CC	_	14	cc	14:cc	_
14	Guard	Guard	PROPN	NNP	Number=Sing	12	conj	10:amod|12:conj:or	SpaceAfter=No
15	,	,	PUNCT	,	_	18	punct	18:punct	_
16	and	and	CCONJ	CC	_	18	cc	18:cc	_
17	you	you	PRON	PRP	Case=Nom|Person=2|PronType=Prs	18	nsubj	18:nsubj|22:nsubj|24:nsubj:xsubj	_
18	had	have	VERB	VBD	Mood=Ind|Number=Sing|Person=2|Tense=Past|VerbForm=Fin	10	conj	10:conj:and|28:advcl:if	_
19	an	a	DET	DT	Definite=Ind|PronType=Art	20	det	20:det	_
20	obligation	obligation	NOUN	NN	Number=Sing	18	obj	18:obj	_
21	and	and	CCONJ	CC	_	22	cc	22:cc	_
22	wanted	want	VERB	VBD	Mood=Ind|Number=Sing|Person=2|Tense=Past|VerbForm=Fin	18	conj	18:conj:and	_
23	to	to	PART	TO	_	24	mark	24:mark	_
24	get	get	VERB	VB	VerbForm=Inf	22	xcomp	22:xcomp	_
25	out	out	ADV	RB	_	24	advmod	24:advmod	SpaceAfter=No
26	,	,	PUNCT	,	_	10	punct	10:punct	_
27	no	no	DET	DT	PronType=Neg	28	det	28:det	_
28	problem	problem	NOUN	NN	Number=Sing	0	root	0:root	SpaceAfter=No
29	.	.	PUNCT	.	_	28	punct	28:punct	_

# sent_id = weblog-blogspot.com_alaindewitt_20040929103700_ENG_20040929_103700-0054
# text = In fact, you were helping them solve their problem.”
1	In	in	ADP	IN	_	2	case	2:case	_
2	fact	fact	NOUN	NN	Number=Sing	6	obl	6:obl:in	SpaceAfter=No
3	,	,	PUNCT	,	_	2	punct	2:punct	_
4	you	you	PRON	PRP	Case=Nom|Person=2|PronType=Prs	6	nsubj	6:nsubj	_
5	were	be	AUX	VBD	Mood=Ind|Number=Sing|Person=2|Tense=Past|VerbForm=Fin	6	aux	6:aux	_
6	helping	help	VERB	VBG	Tense=Pres|VerbForm=Part	0	root	0:root	_
7	them	they	PRON	PRP	Case=Acc|Number=Plur|Person=3|PronType=Prs	6	obj	6:obj|8:nsubj:xsubj	_
8	solve	solve	VERB	VB	VerbForm=Inf	6	xcomp	6:xcomp	_
9	their	their	PRON	PRP$	Case=Gen|Number=Plur|Person=3|Poss=Yes|PronType=Prs	10	nmod:poss	10:nmod:poss	_
10	problem	problem	NOUN	NN	Number=Sing	8	obj	8:obj	SpaceAfter=No
11	.	.	PUNCT	.	_	6	punct	6:punct	SpaceAfter=No
12	”	"	PUNCT	''	_	6	punct	6:punct	_

# sent_id = weblog-blogspot.com_alaindewitt_20040929103700_ENG_20040929_103700-0055
# text = So Bush stopped flying.
1	So	so	ADV	RB	_	3	advmod	3:advmod	_
2	Bush	Bush	PROPN	NNP	Number=Sing	3	nsubj	3:nsubj|4:nsubj:xsubj	_
3	stopped	stop	VERB	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	0	root	0:root	_
4	flying	fly	VERB	VBG	VerbForm=Ger	3	xcomp	3:xcomp	SpaceAfter=No
5	.	.	PUNCT	.	_	3	punct	3:punct	_
""".lstrip()

def test_length_limited_dataloader():
    sample = CoNLL.conll2doc(input_str=EWT_SAMPLE)

    args = tagger.parse_args(args=["--batch_size", "10", "--shorthand", "en_test", "--augment_nopunct", "0.0"])
    data = Dataset(sample, args, None)

    # this should read the whole dataset
    dl = data.to_length_limited_loader(5, 1000)
    batches = [batch.idx for batch in dl]
    assert batches == [(0, 1, 2, 3, 4)]

    dl = data.to_length_limited_loader(4, 1000)
    batches = [batch.idx for batch in dl]
    assert batches == [(0, 1, 2, 3), (4,)]

    dl = data.to_length_limited_loader(2, 1000)
    batches = [batch.idx for batch in dl]
    assert batches == [(0, 1), (2, 3), (4,)]

    # the first three sentences should reach this limit
    dl = data.to_length_limited_loader(5, 55)
    batches = [batch.idx for batch in dl]
    assert batches == [(0, 1, 2), (3, 4)]

    # the third sentence (2) is already past this limit by itself
    dl = data.to_length_limited_loader(5, 25)
    batches = [batch.idx for batch in dl]
    assert batches == [(0, 1), (2,), (3, 4)]
