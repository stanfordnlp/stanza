"""
Very simple test of the sentence slicing by <PAD> tags

TODO: could add a bunch more simple tests for the tokenization utils
"""

import pytest
import stanza

from stanza import Pipeline
from stanza.tests import *
from stanza.models.common import doc
from stanza.models.tokenization import data
from stanza.models.tokenization import utils

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def test_find_spans():
    """
    Test various raw -> span manipulations
    """
    raw = ['u', 'n', 'b', 'a', 'n', ' ', 'm', 'o', 'x', ' ', 'o', 'p', 'a', 'l']
    assert utils.find_spans(raw) == [(0, 14)]

    raw = ['u', 'n', 'b', 'a', 'n', ' ', 'm', 'o', 'x', ' ', 'o', 'p', 'a', 'l', '<PAD>']
    assert utils.find_spans(raw) == [(0, 14)]

    raw = ['<PAD>', 'u', 'n', 'b', 'a', 'n', ' ', 'm', 'o', 'x', ' ', 'o', 'p', 'a', 'l', '<PAD>']
    assert utils.find_spans(raw) == [(1, 15)]

    raw = ['<PAD>', 'u', 'n', 'b', 'a', 'n', ' ', 'm', 'o', 'x', ' ', 'o', 'p', 'a', 'l']
    assert utils.find_spans(raw) == [(1, 15)]

    raw = ['<PAD>', 'u', 'n', 'b', 'a', 'n', '<PAD>', 'm', 'o', 'x', ' ', 'o', 'p', 'a', 'l']
    assert utils.find_spans(raw) == [(1, 6), (7, 15)]

def check_offsets(doc, expected_offsets):
    """
    Compare the start_char and end_char of the tokens in the doc with the given list of list of offsets
    """
    assert len(doc.sentences) == len(expected_offsets)
    for sentence, offsets in zip(doc.sentences, expected_offsets):
        assert len(sentence.tokens) == len(offsets)
        for token, offset in zip(sentence.tokens, offsets):
            assert token.start_char == offset[0]
            assert token.end_char == offset[1]

def test_match_tokens_with_text():
    """
    Test the conversion of pretokenized text to Document
    """
    doc = utils.match_tokens_with_text([["This", "is", "a", "test"]], "Thisisatest")
    expected_offsets = [[(0, 4), (4, 6), (6, 7), (7, 11)]]
    check_offsets(doc, expected_offsets)

    doc = utils.match_tokens_with_text([["This", "is", "a", "test"], ["unban", "mox", "opal", "!"]], "Thisisatest  unban mox  opal!")
    expected_offsets = [[(0, 4), (4, 6), (6, 7), (7, 11)],
                        [(13, 18), (19, 22), (24, 28), (28, 29)]]
    check_offsets(doc, expected_offsets)

    with pytest.raises(ValueError):
        doc = utils.match_tokens_with_text([["This", "is", "a", "test"]], "Thisisatestttt")

    with pytest.raises(ValueError):
        doc = utils.match_tokens_with_text([["This", "is", "a", "test"]], "Thisisates")

    with pytest.raises(ValueError):
        doc = utils.match_tokens_with_text([["This", "iz", "a", "test"]], "Thisisatest")

def test_long_paragraph():
    """
    Test the tokenizer's capacity to break text up into smaller chunks
    """
    pipeline = Pipeline("en", dir=TEST_MODELS_DIR, processors="tokenize")
    tokenizer = pipeline.processors['tokenize']

    raw_text = "TIL not to ask a date to dress up as Smurfette on a first date.  " * 100

    # run a test to make sure the chunk operation is called
    # if not, the test isn't actually testing what we need to test
    batches = data.DataLoader(tokenizer.config, input_text=raw_text, vocab=tokenizer.vocab, evaluation=True, dictionary=tokenizer.trainer.dictionary)
    batches.advance_old_batch = None
    with pytest.raises(TypeError):
        _, _, _, document = utils.output_predictions(None, tokenizer.trainer, batches, tokenizer.vocab, None, 3000,
                                                     orig_text=raw_text,
                                                     no_ssplit=tokenizer.config.get('no_ssplit', False))

    # a new DataLoader should not be crippled as the above one was
    batches = data.DataLoader(tokenizer.config, input_text=raw_text, vocab=tokenizer.vocab, evaluation=True, dictionary=tokenizer.trainer.dictionary)
    _, _, _, document = utils.output_predictions(None, tokenizer.trainer, batches, tokenizer.vocab, None, 3000,
                                                 orig_text=raw_text,
                                                 no_ssplit=tokenizer.config.get('no_ssplit', False))

    document = doc.Document(document, raw_text)
    assert len(document.sentences) == 100

def test_postprocessor_application():
    """
    Check that the postprocessor behaves correctly by applying the identity postprocessor and hoping that it does indeed return correctly.
    """

    good_tokenization = [['I', 'am', 'Joe.', '⭆⊱⇞', 'Hi', '.'], ["I'm", 'a', 'chicken', '.']]
    text = "I am Joe. ⭆⊱⇞ Hi. I'm a chicken."

    target_doc = [[{'id': 1, 'text': 'I', 'start_char': 0, 'end_char': 1}, {'id': 2, 'text': 'am', 'start_char': 2, 'end_char': 4}, {'id': 3, 'text': 'Joe.', 'start_char': 5, 'end_char': 9}, {'id': 4, 'text': '⭆⊱⇞', 'start_char': 10, 'end_char': 13}, {'id': 5, 'text': 'Hi', 'start_char': 14, 'end_char': 16, 'misc': 'SpaceAfter=No'}, {'id': 6, 'text': '.', 'start_char': 16, 'end_char': 17}], [{'id': 1, 'text': "I'm", 'start_char': 18, 'end_char': 21}, {'id': 2, 'text': 'a', 'start_char': 22, 'end_char': 23}, {'id': 3, 'text': 'chicken', 'start_char': 24, 'end_char': 31, 'misc': 'SpaceAfter=No'}, {'id': 4, 'text': '.', 'start_char': 31, 'end_char': 32, 'misc': 'SpaceAfter=No'}]]

    def postprocesor(_):
        return good_tokenization

    res = utils.postprocess_doc(target_doc, postprocesor, text)

    assert res == target_doc

def test_reassembly_indexing():
    """
    Check that the reassembly code counts the indicies correctly, and including OOV chars.
    """

    good_tokenization = [['I', 'am', 'Joe.', '⭆⊱⇞', 'Hi', '.'], ["I'm", 'a', 'chicken', '.']]
    good_mwts = [[False for _ in range(len(i))] for i in good_tokenization]
    good_expansions = [[None for _ in range(len(i))] for i in good_tokenization]

    text = "I am Joe. ⭆⊱⇞ Hi. I'm a chicken."

    target_doc = [[{'id': 1, 'text': 'I', 'start_char': 0, 'end_char': 1}, {'id': 2, 'text': 'am', 'start_char': 2, 'end_char': 4}, {'id': 3, 'text': 'Joe.', 'start_char': 5, 'end_char': 9}, {'id': 4, 'text': '⭆⊱⇞', 'start_char': 10, 'end_char': 13}, {'id': 5, 'text': 'Hi', 'start_char': 14, 'end_char': 16, 'misc': 'SpaceAfter=No'}, {'id': 6, 'text': '.', 'start_char': 16, 'end_char': 17}], [{'id': 1, 'text': "I'm", 'start_char': 18, 'end_char': 21}, {'id': 2, 'text': 'a', 'start_char': 22, 'end_char': 23}, {'id': 3, 'text': 'chicken', 'start_char': 24, 'end_char': 31, 'misc': 'SpaceAfter=No'}, {'id': 4, 'text': '.', 'start_char': 31, 'end_char': 32, 'misc': 'SpaceAfter=No'}]]

    res = utils.reassemble_doc_from_tokens(good_tokenization, good_mwts, good_expansions, text)

    assert res == target_doc

def test_reassembly_reference_failures():
    """
    Check that the reassembly code complains correctly when the user adds tokens that doesn't exist
    """

    bad_addition_tokenization = [['Joe', 'Smith', 'lives', 'in', 'Southern', 'California', '.']]
    bad_addition_mwts = [[False for _ in range(len(bad_addition_tokenization[0]))]]
    bad_addition_expansions = [[None for _ in range(len(bad_addition_tokenization[0]))]]

    bad_inline_tokenization = [['Joe', 'Smith', 'lives', 'in', 'Californiaa', '.']]
    bad_inline_mwts = [[False for _ in range(len(bad_inline_tokenization[0]))]]
    bad_inline_expansions = [[None for _ in range(len(bad_inline_tokenization[0]))]]

    good_tokenization = [['Joe', 'Smith', 'lives', 'in', 'California', '.']]
    good_mwts = [[False for _ in range(len(good_tokenization[0]))]]
    good_expansions = [[None for _ in range(len(good_tokenization[0]))]]

    text = "Joe Smith lives in California."

    with pytest.raises(ValueError):
        utils.reassemble_doc_from_tokens(bad_addition_tokenization, bad_addition_mwts, bad_addition_expansions, text)

    with pytest.raises(ValueError):
        utils.reassemble_doc_from_tokens(bad_inline_tokenization, bad_inline_mwts, bad_inline_mwts, text)

    utils.reassemble_doc_from_tokens(good_tokenization, good_mwts, good_expansions, text)



TRAIN_DATA = """
# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0003
# text = DPA: Iraqi authorities announced that they'd busted up three terrorist cells operating in Baghdad.
1	DPA	DPA	PROPN	NNP	Number=Sing	0	root	0:root	SpaceAfter=No
2	:	:	PUNCT	:	_	1	punct	1:punct	_
3	Iraqi	Iraqi	ADJ	JJ	Degree=Pos	4	amod	4:amod	_
4	authorities	authority	NOUN	NNS	Number=Plur	5	nsubj	5:nsubj	_
5	announced	announce	VERB	VBD	Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin	1	parataxis	1:parataxis	_
6	that	that	SCONJ	IN	_	9	mark	9:mark	_
7-8	they'd	_	_	_	_	_	_	_	_
7	they	they	PRON	PRP	Case=Nom|Number=Plur|Person=3|PronType=Prs	9	nsubj	9:nsubj	_
8	'd	have	AUX	VBD	Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin	9	aux	9:aux	_
9	busted	bust	VERB	VBN	Tense=Past|VerbForm=Part	5	ccomp	5:ccomp	_
10	up	up	ADP	RP	_	9	compound:prt	9:compound:prt	_
11	three	three	NUM	CD	NumForm=Digit|NumType=Card	13	nummod	13:nummod	_
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

def test_lexicon_from_training_data(tmp_path):
    """
    Test a couple aspects of building a lexicon from training data

    expected number of words eliminated for being too long
    duplicate words counted once
    numbers eliminated
    """
    conllu_file = str(tmp_path / "train.conllu")
    with open(conllu_file, "w", encoding="utf-8") as fout:
        fout.write(TRAIN_DATA)

    lexicon, num_dict_feat = utils.create_lexicon("en_test", conllu_file)
    lexicon = sorted(lexicon)
    expected_lexicon = ["'d", 'announced', 'baghdad', 'being', 'busted', 'by', 'cells', 'dpa', 'in', 'interior', 'iraqi', 'ministry', 'of', 'officials', 'operating', 'run', 'terrorist', 'that', 'the', 'them', 'they', "they'd", 'three', 'two', 'up', 'were']
    assert lexicon == expected_lexicon
    assert num_dict_feat == max(len(x) for x in lexicon)

