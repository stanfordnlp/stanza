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
2	of	of	ADP	IN	_	4	case	4:case	_
3	“	"	PUNCT	``	_	4	punct	4:punct	SpaceAfter=No
4-5	NCRC4ME’s	_	_	_	_	_	_	_	SpaceAfter=No
4	NCRC4ME	NCRC4ME	PROPN	NNP	Number=Sing	1	compound	1:compound	_
5	’s	's	PART	POS	_	4	case	4:case	_
6	”	"	PUNCT	''	_	4	punct	4:punct	_
"""

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

    Note regarding this test:
    Currently it works because ssurgeon.py doesn't look at the
      "changed" flag because of a bug in EditNode in CoreNLP 4.5.3
    If that is fixed, but the enhanced dependencies aren't fixed,
      this test will fail because the enhanced dependencies *aren't*
      removed.  Fixing the enhanced dependencies as well will fix
      that, though.
    """
    semgrex_pattern = "{word:It}=it . {word:/'s/}=s"
    ssurgeon_edits = ["EditNode -node it -is_mwt true  -is_first_mwt true  -mwt_text It's",
                      "EditNode -node s  -is_mwt true  -is_first_mwt false -mwt_text It's"]

    doc = CoNLL.conll2doc(input_str=EXISTING_MWT_DOC_INPUT)

    ssurgeon_response = ssurgeon.process_doc_one_operation(doc, semgrex_pattern, ssurgeon_edits)
    updated_doc = ssurgeon.convert_response_to_doc(doc, ssurgeon_response)

    result = "{:C}".format(updated_doc)
    compare_ignoring_whitespace(result, EXISTING_MWT_DOC_EXPECTED)

def check_empty_test(input_text, expected=None, echo=False):
    if expected is None:
        expected = input_text

    doc = CoNLL.conll2doc(input_str=input_text)

    # we don't want to edit this, just test the to/from conversion
    ssurgeon_response = ssurgeon.process_doc(doc, [])
    updated_doc = ssurgeon.convert_response_to_doc(doc, ssurgeon_response)

    result = "{:C}".format(updated_doc)
    if echo:
        print("INPUT")
        print(input_text)
        print("EXPECTED")
        print(expected)
        print("RESULT")
        print(result)
    compare_ignoring_whitespace(result, expected)

ITALIAN_MWT_INPUT = """
# sent_id = train_78
# text = @user dovrebbe fare pace col cervello
# twittiro = IMPLICIT	ANALOGY
1	@user	@user	SYM	SYM	_	3	nsubj	_	_
2	dovrebbe	dovere	AUX	VM	Mood=Cnd|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	3	aux	_	_
3	fare	fare	VERB	V	VerbForm=Inf	0	root	_	_
4	pace	pace	NOUN	S	Gender=Fem|Number=Sing	3	obj	_	_
5-6	col	_	_	_	_	_	_	_	_
5	con	con	ADP	E	_	7	case	_	_
6	il	il	DET	RD	Definite=Def|Gender=Masc|Number=Sing|PronType=Art	7	det	_	_
7	cervello	cervello	NOUN	S	Gender=Masc|Number=Sing	3	obl	_	_
"""

def test_ssurgeon_mwt_text():
    """
    Test that an MWT which is split into pieces which don't make up
    the original token results in a correct #text annotation

    For example, in Italian, "col" splits into "con il", and we want
    the #text to contain "col"
    """
    check_empty_test(ITALIAN_MWT_INPUT)

ITALIAN_SPACES_AFTER_INPUT="""
# sent_id = train_1114
# text = ““““ buona scuola ““““
# twittiro = EXPLICIT	OTHER
1	“	“	PUNCT	FB	_	6	punct	_	SpaceAfter=No
2	“	“	PUNCT	FB	_	6	punct	_	SpaceAfter=No
3	“	“	PUNCT	FB	_	6	punct	_	SpaceAfter=No
4	“	“	PUNCT	FB	_	6	punct	_	_
5	buona	buono	ADJ	A	Gender=Fem|Number=Sing	6	amod	_	_
6	scuola	scuola	NOUN	S	Gender=Fem|Number=Sing	0	root	_	_
7	“	“	PUNCT	FB	_	6	punct	_	SpaceAfter=No
8	“	“	PUNCT	FB	_	6	punct	_	SpaceAfter=No
9	“	“	PUNCT	FB	_	6	punct	_	SpaceAfter=No
10	“	“	PUNCT	FB	_	6	punct	_	SpacesAfter=\\n
"""

ITALIAN_SPACES_AFTER_YES_INPUT="""
# sent_id = train_1114
# text = ““““ buona scuola ““““
# twittiro = EXPLICIT	OTHER
1	“	“	PUNCT	FB	_	6	punct	_	SpaceAfter=No
2	“	“	PUNCT	FB	_	6	punct	_	SpaceAfter=No
3	“	“	PUNCT	FB	_	6	punct	_	SpaceAfter=No
4	“	“	PUNCT	FB	_	6	punct	_	SpaceAfter=Yes
5	buona	buono	ADJ	A	Gender=Fem|Number=Sing	6	amod	_	_
6	scuola	scuola	NOUN	S	Gender=Fem|Number=Sing	0	root	_	_
7	“	“	PUNCT	FB	_	6	punct	_	SpaceAfter=No
8	“	“	PUNCT	FB	_	6	punct	_	SpaceAfter=No
9	“	“	PUNCT	FB	_	6	punct	_	SpaceAfter=No
10	“	“	PUNCT	FB	_	6	punct	_	SpacesAfter=\\n
"""


def test_ssurgeon_spaces_after_text():
    """
    Test that SpacesAfter goes and comes back the same way

    Tested using some random example from the UD_Italian-TWITTIRO dataset
    """
    check_empty_test(ITALIAN_SPACES_AFTER_INPUT)

def test_ssurgeon_spaces_after_yes():
    """
    Test that an unnecessary SpaceAfter=Yes is eliminated
    """
    check_empty_test(ITALIAN_SPACES_AFTER_YES_INPUT, ITALIAN_SPACES_AFTER_INPUT)

EMPTY_VALUES_INPUT = """
# text = Jennifer has lovely antennae.
# sent_id = 12
# comment = if you're in to that kind of thing
1	Jennifer	_	_	_	Number=Sing	2	nsubj	_	ner=S-PERSON
2	has	_	_	_	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	_	ner=O
3	lovely	_	_	_	Degree=Pos	4	amod	_	ner=O
4	antennae	_	_	_	Number=Plur	2	obj	_	SpaceAfter=No|ner=O
5	.	_	_	_	_	2	punct	_	ner=O
"""

def test_ssurgeon_blank_values():
    """
    Check that various None fields such as lemma & xpos are not turned into blanks

    Tests, like regulations, are often written in blood
    """
    check_empty_test(EMPTY_VALUES_INPUT)

# first couple sentences of UD_Cantonese-HK
# we change the order of the misc column in word 3 to make sure the
# pieces don't get unnecessarily reordered by ssurgeon
CANTONESE_MISC_WORDS_INPUT = """
# sent_id = 1
# text = 你喺度搵乜嘢呀？
1	你	你	PRON	_	_	3	nsubj	_	SpaceAfter=No|Translit=nei5|Gloss=2SG
2	喺度	喺度	ADV	_	_	3	advmod	_	SpaceAfter=No|Translit=hai2dou6|Gloss=PROG
3	搵	搵	VERB	_	_	0	root	_	Translit=wan2|Gloss=find|SpaceAfter=No
4	乜嘢	乜嘢	PRON	_	_	3	obj	_	SpaceAfter=No|Translit=mat1je5|Gloss=what
5	呀	呀	PART	_	_	3	discourse:sp	_	SpaceAfter=No|Translit=aa3|Gloss=SFP
6	？	？	PUNCT	_	_	3	punct	_	SpaceAfter=No

# sent_id = 2
# text = 咪執返啲嘢去阿哥個新屋度囖。
1	咪	咪	ADV	_	_	2	advmod	_	SpaceAfter=No
2	執	執	VERB	_	_	0	root	_	SpaceAfter=No
3	返	返	VERB	_	_	2	compound:dir	_	SpaceAfter=No
4	啲	啲	NOUN	_	NounType=Clf	5	clf:det	_	SpaceAfter=No
5	嘢	嘢	NOUN	_	_	3	obj	_	SpaceAfter=No
6	去	去	VERB	_	_	2	conj	_	SpaceAfter=No
7	阿哥	阿哥	NOUN	_	_	10	nmod	_	SpaceAfter=No
8	個	個	NOUN	_	NounType=Clf	10	clf:det	_	SpaceAfter=No
9	新	新	ADJ	_	_	10	amod	_	SpaceAfter=No
10	屋	屋	NOUN	_	_	6	obj	_	SpaceAfter=No
11	度	度	ADP	_	_	10	case:loc	_	SpaceAfter=No
12	囖	囖	PART	_	_	2	discourse:sp	_	SpaceAfter=No
13	。	。	PUNCT	_	_	2	punct	_	SpaceAfter=No
"""

def test_ssurgeon_misc_words():
    """
    Check that various None fields such as lemma & xpos are not turned into blanks

    Tests, like regulations, are often written in blood
    """
    check_empty_test(CANTONESE_MISC_WORDS_INPUT)

ITALIAN_MWT_SPACE_AFTER_INPUT = """
# sent_id = train_78
# text = @user dovrebbe fare pace colcervello
# twittiro = IMPLICIT	ANALOGY
1	@user	@user	SYM	SYM	_	3	nsubj	_	_
2	dovrebbe	dovere	AUX	VM	Mood=Cnd|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	3	aux	_	_
3	fare	fare	VERB	V	VerbForm=Inf	0	root	_	_
4	pace	pace	NOUN	S	Gender=Fem|Number=Sing	3	obj	_	_
5-6	col	_	_	_	_	_	_	_	SpaceAfter=No
5	con	con	ADP	E	_	7	case	_	_
6	il	il	DET	RD	Definite=Def|Gender=Masc|Number=Sing|PronType=Art	7	det	_	_
7	cervello	cervello	NOUN	S	Gender=Masc|Number=Sing	3	obl	_	RandomFeature=foo
"""

def test_ssurgeon_mwt_space_after():
    """
    Check the SpaceAfter=No on an MWT (rather than a word)

    the RandomFeature=foo is on account of a silly bug in the initial
    version of passing in MWT misc features
    """
    check_empty_test(ITALIAN_MWT_SPACE_AFTER_INPUT)

ITALIAN_MWT_MISC_INPUT = """
# sent_id = train_78
# text = @user dovrebbe farepacecolcervello
# twittiro = IMPLICIT	ANALOGY
1	@user	@user	SYM	SYM	_	3	nsubj	_	_
2	dovrebbe	dovere	AUX	VM	Mood=Cnd|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	3	aux	_	_
3-4	farepace	_	_	_	_	_	_	_	Players=GonnaPlay|SpaceAfter=No
3	fare	fare	VERB	V	VerbForm=Inf	0	root	_	_
4	pace	pace	NOUN	S	Gender=Fem|Number=Sing	3	obj	_	_
5-6	col	_	_	_	_	_	_	_	SpaceAfter=No|Haters=GonnaHate
5	con	con	ADP	E	_	7	case	_	_
6	il	il	DET	RD	Definite=Def|Gender=Masc|Number=Sing|PronType=Art	7	det	_	_
7	cervello	cervello	NOUN	S	Gender=Masc|Number=Sing	3	obl	_	RandomFeature=foo
"""

def test_ssurgeon_mwt_misc():
    """
    Check the SpaceAfter=No on an MWT (rather than a word)

    the RandomFeature=foo is on account of a silly bug in the initial
    version of passing in MWT misc features
    """
    # currently commented out because the public version of CoreNLP doesn't support it
    #check_empty_test(ITALIAN_MWT_MISC_INPUT)

