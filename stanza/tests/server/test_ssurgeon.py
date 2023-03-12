import pytest

from stanza.tests import compare_ignoring_whitespace

pytestmark = [pytest.mark.travis, pytest.mark.client]

from stanza.utils.conll import CoNLL
import stanza.server.ssurgeon as ssurgeon

SAMPLE_DOC_INPUT = """
# sent_id = 271
# text = Hers is easy to clean.
# previous = What did the dealer like about Alex's car?
# comment = extraction/raising via "tough extraction" and clausal subject
1	Hers	hers	PRON	PRP	Gender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs	3	nsubj	_	_
2	is	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	3	cop	_	_
3	easy	easy	ADJ	JJ	Degree=Pos	0	root	_	_
4	to	to	PART	TO	_	5	mark	_	_
5	clean	clean	VERB	VB	VerbForm=Inf	3	csubj	_	SpaceAfter=No
6	.	.	PUNCT	.	_	5	punct	_	_
"""

SAMPLE_DOC_EXPECTED = """
# sent_id = 271
# text = Hers is easy to clean.
# previous = What did the dealer like about Alex's car?
# comment = extraction/raising via "tough extraction" and clausal subject
1	Hers	hers	PRON	PRP	Gender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs	3	nsubj	_	_
2	is	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	3	cop	_	_
3	easy	easy	ADJ	JJ	Degree=Pos	0	root	_	_
4	to	to	PART	TO	_	5	mark	_	_
5	clean	clean	VERB	VB	VerbForm=Inf	3	advcl	_	SpaceAfter=No
6	.	.	PUNCT	.	_	5	punct	_	_
"""


def test_ssurgeon_same_length():
    semgrex_pattern = "{}=source >nsubj {} >csubj=bad {}"
    ssurgeon_edits = ["relabelNamedEdge -edge bad -reln advcl"]

    doc = CoNLL.conll2doc(input_str=SAMPLE_DOC_INPUT)

    ssurgeon_response = ssurgeon.process_doc_one_operation(doc, semgrex_pattern, ssurgeon_edits)
    updated_doc = ssurgeon.convert_response_to_doc(doc, ssurgeon_response)

    result = "{:C}".format(updated_doc)
    #print(result)
    #print(SAMPLE_DOC_EXPECTED)
    compare_ignoring_whitespace(result, SAMPLE_DOC_EXPECTED)


ADD_WORD_DOC_INPUT = """
# text = Jennifer has lovely antennae.
# sent_id = 12
# comment = if you're in to that kind of thing
1	Jennifer	Jennifer	PROPN	NNP	Number=Sing	2	nsubj	_	start_char=0|end_char=8|ner=S-PERSON
2	has	have	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	_	start_char=9|end_char=12|ner=O
3	lovely	lovely	ADJ	JJ	Degree=Pos	4	amod	_	start_char=13|end_char=19|ner=O
4	antennae	antenna	NOUN	NNS	Number=Plur	2	obj	_	start_char=20|end_char=28|ner=O|SpaceAfter=No
5	.	.	PUNCT	.	_	2	punct	_	start_char=28|end_char=29|ner=O
"""

ADD_WORD_DOC_EXPECTED = """
# text = Jennifer has lovely blue antennae.
# sent_id = 12
# comment = if you're in to that kind of thing
1	Jennifer	Jennifer	PROPN	NNP	Number=Sing	2	nsubj	_	ner=S-PERSON
2	has	have	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	_	ner=O
3	lovely	lovely	ADJ	JJ	Degree=Pos	5	amod	_	ner=O
4	blue	blue	ADJ	JJ	_	5	amod	_	ner=O
5	antennae	antenna	NOUN	NNS	Number=Plur	2	obj	_	SpaceAfter=No|ner=O
6	.	.	PUNCT	.	_	2	punct	_	ner=O
"""


def test_ssurgeon_different_length():
    semgrex_pattern = "{word:antennae}=antennae !> {word:blue}"
    ssurgeon_edits = ["addDep -gov antennae -reln amod -word blue -lemma blue -cpos ADJ -pos JJ -ner O -position -antennae -after \" \""]

    doc = CoNLL.conll2doc(input_str=ADD_WORD_DOC_INPUT)
    #print()
    #print("{:C}".format(doc))

    ssurgeon_response = ssurgeon.process_doc_one_operation(doc, semgrex_pattern, ssurgeon_edits)
    updated_doc = ssurgeon.convert_response_to_doc(doc, ssurgeon_response)

    result = "{:C}".format(updated_doc)
    #print(result)
    #print(ADD_WORD_DOC_EXPECTED)

    compare_ignoring_whitespace(result, ADD_WORD_DOC_EXPECTED)

BECOME_MWT_DOC_INPUT = """
# sent_id = 25
# text = It's not yours!
# comment = negation 
1	It	it	PRON	PRP	Number=Sing|Person=2|PronType=Prs	4	nsubj	_	SpaceAfter=No
2	's	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	4	cop	_	_
3	not	not	PART	RB	Polarity=Neg	4	advmod	_	_
4	yours	yours	PRON	PRP	Gender=Neut|Number=Sing|Person=2|Poss=Yes|PronType=Prs	0	root	_	SpaceAfter=No
5	!	!	PUNCT	.	_	4	punct	_	_
"""

BECOME_MWT_DOC_EXPECTED = """
# sent_id = 25
# text = It's not yours!
# comment = negation
1-2	It's	_	_	_	_	_	_	_	_
1	It	it	PRON	PRP	Number=Sing|Person=2|PronType=Prs	4	nsubj	_	_
2	's	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	4	cop	_	_
3	not	not	PART	RB	Polarity=Neg	4	advmod	_	_
4	yours	yours	PRON	PRP	Gender=Neut|Number=Sing|Person=2|Poss=Yes|PronType=Prs	0	root	_	SpaceAfter=No
5	!	!	PUNCT	.	_	4	punct	_	_
"""

def test_ssurgeon_become_mwt():
    """
    Test that converting a document, adding a new MWT, works as expected
    """
    semgrex_pattern = "{word:It}=it . {word:/'s/}=s"
    ssurgeon_edits = ["EditNode -node it -is_mwt true  -is_first_mwt true  -mwt_text It's",
                      "EditNode -node s  -is_mwt true  -is_first_mwt false -mwt_text It's"]

    doc = CoNLL.conll2doc(input_str=BECOME_MWT_DOC_INPUT)

    ssurgeon_response = ssurgeon.process_doc_one_operation(doc, semgrex_pattern, ssurgeon_edits)
    updated_doc = ssurgeon.convert_response_to_doc(doc, ssurgeon_response)

    result = "{:C}".format(updated_doc)
    compare_ignoring_whitespace(result, BECOME_MWT_DOC_EXPECTED)

EXISTING_MWT_DOC_INPUT = """
# sent_id = newsgroup-groups.google.com_GayMarriage_0ccbb50b41a5830b_ENG_20050321_181500-0005
# text = One of “NCRC4ME’s”
1	One	one	NUM	CD	NumType=Card	0	root	0:root	_
2	of	of	ADP	IN	_	4	case	8:case	_
3	“	"	PUNCT	``	_	4	punct	4:punct	SpaceAfter=No
4-5	NCRC4ME’s	_	_	_	_	_	_	_	SpaceAfter=No
4	NCRC4ME	NCRC4ME	PROPN	NNP	Number=Sing	1	compound	8:compound	_
5	’s	's	PART	POS	_	4	case	4:case	_
6	”	"	PUNCT	''	_	4	punct	4:punct	_
"""

# TODO: word 4 should not have SpaceAfter=No, but that needs to be fixed in ssurgeon.py first
# TODO: also, we shouldn't lose the enhanced dependencies...
EXISTING_MWT_DOC_EXPECTED = """
# sent_id = newsgroup-groups.google.com_GayMarriage_0ccbb50b41a5830b_ENG_20050321_181500-0005
# text = One of “NCRC4ME’s”
1	One	one	NUM	CD	NumType=Card	0	root	_	_
2	of	of	ADP	IN	_	4	case	_	_
3	“	"	PUNCT	``	_	4	punct	_	SpaceAfter=No
4-5	NCRC4ME’s	_	_	_	_	_	_	_	SpaceAfter=No
4	NCRC4ME	NCRC4ME	PROPN	NNP	Number=Sing	1	compound	_	_
5	’s	's	PART	POS	_	4	case	_	_
6	”	"	PUNCT	''	_	4	punct	_	_
"""

def test_ssurgeon_existing_mwt_no_change():
    """
    Test that converting a document with an MWT works as expected
    """
    semgrex_pattern = "{word:It}=it . {word:/'s/}=s"
    ssurgeon_edits = ["EditNode -node it -is_mwt true  -is_first_mwt true  -mwt_text It's",
                      "EditNode -node s  -is_mwt true  -is_first_mwt false -mwt_text It's"]

    doc = CoNLL.conll2doc(input_str=EXISTING_MWT_DOC_INPUT)

    ssurgeon_response = ssurgeon.process_doc_one_operation(doc, semgrex_pattern, ssurgeon_edits)
    updated_doc = ssurgeon.convert_response_to_doc(doc, ssurgeon_response)

    result = "{:C}".format(updated_doc)
    compare_ignoring_whitespace(result, EXISTING_MWT_DOC_EXPECTED)
