"""
Test some pieces of the depparse dataloader
"""
import pytest
from stanza.models import parser
from stanza.models.depparse.data import data_to_batches, DataLoader
from stanza.utils.conll import CoNLL

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def make_fake_data(*lengths):
    data = []
    for i, length in enumerate(lengths):
        word = chr(ord('A') + i)
        chunk = [[word] * length]
        data.append(chunk)
    return data

def check_batches(batched_data, expected_sizes, expected_order):
    for chunk, size in zip(batched_data, expected_sizes):
        assert sum(len(x[0]) for x in chunk) == size
    word_order = []
    for chunk in batched_data:
        for sentence in chunk:
            word_order.append(sentence[0][0])
    assert word_order == expected_order

def test_data_to_batches_eval_mode():
    """
    Tests the chunking of batches in eval_mode

    A few options are tested, such as whether or not to sort and the maximum sentence size
    """
    data = make_fake_data(1, 2, 3)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=True, min_length_to_batch_separately=None)
    check_batches(batched_data[0], [5, 1], ['C', 'B', 'A'])

    data = make_fake_data(1, 2, 6)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=True, min_length_to_batch_separately=None)
    check_batches(batched_data[0], [6, 3], ['C', 'B', 'A'])

    data = make_fake_data(3, 2, 1)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=True, min_length_to_batch_separately=None)
    check_batches(batched_data[0], [5, 1], ['A', 'B', 'C'])

    data = make_fake_data(3, 5, 2)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=True, min_length_to_batch_separately=None)
    check_batches(batched_data[0], [5, 5], ['B', 'A', 'C'])

    data = make_fake_data(3, 5, 2)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=False, min_length_to_batch_separately=3)
    check_batches(batched_data[0], [3, 5, 2], ['A', 'B', 'C'])

    data = make_fake_data(4, 1, 1)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=False, min_length_to_batch_separately=3)
    check_batches(batched_data[0], [4, 2], ['A', 'B', 'C'])

    data = make_fake_data(1, 4, 1)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=False, min_length_to_batch_separately=3)
    check_batches(batched_data[0], [1, 4, 1], ['A', 'B', 'C'])


EWT_PUNCT_SAMPLE = """
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
15	!!!!!	!	PUNCT	.	_	2	punct	2:punct	_

# sent_id = weblog-blogspot.com_alaindewitt_20040929103700_ENG_20040929_103700-0049
# text = His superior officers said OK.
1	His	his	PRON	PRP$	Case=Gen|Gender=Masc|Number=Sing|Person=3|Poss=Yes|PronType=Prs	3	nmod:poss	3:nmod:poss	_
2	superior	superior	ADJ	JJ	Degree=Pos	3	amod	3:amod	_
3	officers	officer	NOUN	NNS	Number=Plur	4	nsubj	4:nsubj	_
4	said	say	VERB	VBD	Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin	0	root	0:root	_
5	OK	ok	INTJ	UH	_	4	obj	4:obj	SpaceAfter=No
6	?????	?	PUNCT	.	_	4	punct	4:punct	_
"""


def test_punct_simplification():
    """
    Test a punctuation simplification that should make it so unexpected
    question/exclamation marks types are processed into ? and !
    """
    sample = CoNLL.conll2doc(input_str=EWT_PUNCT_SAMPLE)

    args = parser.parse_args(args=["--batch_size", "1000", "--shorthand", "en_test"])
    data = DataLoader(sample, 5000, args, None)

    batches = [batch for batch in data]
    assert batches[0][-1] == [['Bush', 'asked', 'for', 'permission', 'to', 'go', 'to', 'Alabama', 'to', 'work', 'on', 'a', 'Senate', 'campaign', '!'],
                              ['His', 'superior', 'officers', 'said', 'OK', '?']]


if __name__ == '__main__':
    test_data_to_batches()

