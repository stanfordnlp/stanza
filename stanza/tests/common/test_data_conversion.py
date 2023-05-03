"""
Basic tests of the data conversion
"""

import io
import pytest
import tempfile
from zipfile import ZipFile

import stanza
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document
from stanza.tests import *

pytestmark = pytest.mark.pipeline

# data for testing
CONLL = [[['1', 'Nous', 'il', 'PRON', '_', 'Number=Plur|Person=1|PronType=Prs', '3', 'nsubj', '_', 'start_char=0|end_char=4'],
          ['2', 'avons', 'avoir', 'AUX', '_', 'Mood=Ind|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin', '3', 'aux:tense', '_', 'start_char=5|end_char=10'],
          ['3', 'atteint', 'atteindre', 'VERB', '_', 'Gender=Masc|Number=Sing|Tense=Past|VerbForm=Part', '0', 'root', '_', 'start_char=11|end_char=18'],
          ['4', 'la', 'le', 'DET', '_', 'Definite=Def|Gender=Fem|Number=Sing|PronType=Art', '5', 'det', '_', 'start_char=19|end_char=21'],
          ['5', 'fin', 'fin', 'NOUN', '_', 'Gender=Fem|Number=Sing', '3', 'obj', '_', 'start_char=22|end_char=25'],
          ['6-7', 'du', '_', '_', '_', '_', '_', '_', '_', 'start_char=26|end_char=28'],
          ['6', 'de', 'de', 'ADP', '_', '_', '8', 'case', '_', '_'],
          ['7', 'le', 'le', 'DET', '_', 'Definite=Def|Gender=Masc|Number=Sing|PronType=Art', '8', 'det', '_', '_'],
          ['8', 'sentier', 'sentier', 'NOUN', '_', 'Gender=Masc|Number=Sing', '5', 'nmod', '_', 'start_char=29|end_char=36'],
          ['9', '.', '.', 'PUNCT', '_', '_', '3', 'punct', '_', 'start_char=36|end_char=37']]]


DICT = [[{'id': (1,), 'text': 'Nous', 'lemma': 'il', 'upos': 'PRON', 'feats': 'Number=Plur|Person=1|PronType=Prs', 'head': 3, 'deprel': 'nsubj', 'misc': 'start_char=0|end_char=4'},
         {'id': (2,), 'text': 'avons', 'lemma': 'avoir', 'upos': 'AUX', 'feats': 'Mood=Ind|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin', 'head': 3, 'deprel': 'aux:tense', 'misc': 'start_char=5|end_char=10'},
         {'id': (3,), 'text': 'atteint', 'lemma': 'atteindre', 'upos': 'VERB', 'feats': 'Gender=Masc|Number=Sing|Tense=Past|VerbForm=Part', 'head': 0, 'deprel': 'root', 'misc': 'start_char=11|end_char=18'},
         {'id': (4,), 'text': 'la', 'lemma': 'le', 'upos': 'DET', 'feats': 'Definite=Def|Gender=Fem|Number=Sing|PronType=Art', 'head': 5, 'deprel': 'det', 'misc': 'start_char=19|end_char=21'},
         {'id': (5,), 'text': 'fin', 'lemma': 'fin', 'upos': 'NOUN', 'feats': 'Gender=Fem|Number=Sing', 'head': 3, 'deprel': 'obj', 'misc': 'start_char=22|end_char=25'},
         {'id': (6, 7), 'text': 'du', 'misc': 'start_char=26|end_char=28'},
         {'id': (6,), 'text': 'de', 'lemma': 'de', 'upos': 'ADP', 'head': 8, 'deprel': 'case'},
         {'id': (7,), 'text': 'le', 'lemma': 'le', 'upos': 'DET', 'feats': 'Definite=Def|Gender=Masc|Number=Sing|PronType=Art', 'head': 8, 'deprel': 'det'},
         {'id': (8,), 'text': 'sentier', 'lemma': 'sentier', 'upos': 'NOUN', 'feats': 'Gender=Masc|Number=Sing', 'head': 5, 'deprel': 'nmod', 'misc': 'start_char=29|end_char=36'},
         {'id': (9,), 'text': '.', 'lemma': '.', 'upos': 'PUNCT', 'head': 3, 'deprel': 'punct', 'misc': 'start_char=36|end_char=37'}]]

def test_conll_to_dict():
    dicts = CoNLL.convert_conll(CONLL)
    assert dicts == DICT

def test_dict_to_conll():
    document = Document(DICT)
    # :c = no comments
    conll = [[sentence.split("\t") for sentence in doc.split("\n")] for doc in "{:c}".format(document).split("\n\n")]
    assert conll == CONLL

def test_dict_to_doc_and_doc_to_dict():
    """
    Test the conversion from raw dict to Document and back

    This code path will first turn start_char|end_char into start_char & end_char fields in the Document
    That version to a dict will have separate fields for each of those
    Finally, the conversion from that dict to a list of conll entries should convert that back to misc
    """
    document = Document(DICT)
    dicts = document.to_dict()
    document = Document(dicts)
    conll = [[sentence.split("\t") for sentence in doc.split("\n")] for doc in "{:c}".format(document).split("\n\n")]
    assert conll == CONLL

# sample is two sentences long so that the tests check multiple sentences
RUSSIAN_SAMPLE="""
# sent_id = yandex.reviews-f-8xh5zqnmwak3t6p68y4rhwd4e0-1969-9253
# genre = review
# text = Как- то слишком мало цветов получают актёры после спектакля.
1	Как	как-то	ADV	_	Degree=Pos|PronType=Ind	7	advmod	_	SpaceAfter=No
2	-	-	PUNCT	_	_	3	punct	_	_
3	то	то	PART	_	_	1	list	_	deprel=list:goeswith
4	слишком	слишком	ADV	_	Degree=Pos	5	advmod	_	_
5	мало	мало	ADV	_	Degree=Pos	6	advmod	_	_
6	цветов	цветок	NOUN	_	Animacy=Inan|Case=Gen|Gender=Masc|Number=Plur	7	obj	_	_
7	получают	получать	VERB	_	Aspect=Imp|Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act	0	root	_	_
8	актёры	актер	NOUN	_	Animacy=Anim|Case=Nom|Gender=Masc|Number=Plur	7	nsubj	_	_
9	после	после	ADP	_	_	10	case	_	_
10	спектакля	спектакль	NOUN	_	Animacy=Inan|Case=Gen|Gender=Masc|Number=Sing	7	obl	_	SpaceAfter=No
11	.	.	PUNCT	_	_	7	punct	_	_

# sent_id = 4
# genre = social
# text = В женщине важна верность, а не красота.
1	В	в	ADP	_	_	2	case	_	_
2	женщине	женщина	NOUN	_	Animacy=Anim|Case=Loc|Gender=Fem|Number=Sing	3	obl	_	_
3	важна	важный	ADJ	_	Degree=Pos|Gender=Fem|Number=Sing|Variant=Short	0	root	_	_
4	верность	верность	NOUN	_	Animacy=Inan|Case=Nom|Gender=Fem|Number=Sing	3	nsubj	_	SpaceAfter=No
5	,	,	PUNCT	_	_	8	punct	_	_
6	а	а	CCONJ	_	_	8	cc	_	_
7	не	не	PART	_	Polarity=Neg	8	advmod	_	_
8	красота	красота	NOUN	_	Animacy=Inan|Case=Nom|Gender=Fem|Number=Sing	4	conj	_	SpaceAfter=No
9	.	.	PUNCT	_	_	3	punct	_	_
""".strip()

RUSSIAN_TEXT = ["Как- то слишком мало цветов получают актёры после спектакля.", "В женщине важна верность, а не красота."]
RUSSIAN_IDS = ["yandex.reviews-f-8xh5zqnmwak3t6p68y4rhwd4e0-1969-9253", "4"]

def check_russian_doc(doc):
    """
    Refactored the test for the Russian doc so we can use it to test various file methods
    """
    lines = RUSSIAN_SAMPLE.split("\n")
    assert len(doc.sentences) == 2
    assert lines[0] == doc.sentences[0].comments[0]
    assert lines[1] == doc.sentences[0].comments[1]
    assert lines[2] == doc.sentences[0].comments[2]
    for sent_idx, (expected_text, expected_id, sentence) in enumerate(zip(RUSSIAN_TEXT, RUSSIAN_IDS, doc.sentences)):
        assert expected_text == sentence.text
        assert expected_id == sentence.sent_id
        assert sent_idx == sentence.index
        assert len(sentence.comments) == 3

    sentences = "{:C}".format(doc)
    sentences = sentences.split("\n\n")
    assert len(sentences) == 2

    sentence = sentences[0].split("\n")
    assert len(sentence) == 14
    assert lines[0] == sentence[0]
    assert lines[1] == sentence[1]
    assert lines[2] == sentence[2]

    # assert that the weird deprel=list:goeswith was properly handled
    assert doc.sentences[0].words[2].head == 1
    assert doc.sentences[0].words[2].deprel == "list:goeswith"

def test_write_russian_doc(tmp_path):
    """
    Specifically test the write_doc2conll method
    """
    filename = tmp_path / "russian.conll"
    doc = CoNLL.conll2doc(input_str=RUSSIAN_SAMPLE)
    check_russian_doc(doc)
    CoNLL.write_doc2conll(doc, filename)

    with open(filename) as fin:
        text = fin.read()

    # the conll docs have to end with \n\n
    assert text.endswith("\n\n")

    # but to compare against the original, strip off the whitespace
    text = text.strip()

    # we skip the first sentence because the "deprel=list:goeswith" is weird
    # note that the deprel itself is checked in check_russian_doc
    text = text[text.find("# sent_id = 4"):]
    sample = RUSSIAN_SAMPLE[RUSSIAN_SAMPLE.find("# sent_id = 4"):]
    assert text == sample

    doc2 = CoNLL.conll2doc(filename)
    check_russian_doc(doc2)

# random sentence from EN_Pronouns
ENGLISH_SAMPLE = """
# newdoc
# sent_id = 1
# text = It is hers.
# previous = Which person owns this?
# comment = copular subject
1	It	it	PRON	PRP	Number=Sing|Person=3|PronType=Prs	3	nsubj	_	_
2	is	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	3	cop	_	_
3	hers	hers	PRON	PRP	Gender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs	0	root	_	SpaceAfter=No
4	.	.	PUNCT	.	_	3	punct	_	_
""".strip()

def test_write_to_io():
    doc = CoNLL.conll2doc(input_str=ENGLISH_SAMPLE)
    output = io.StringIO()
    CoNLL.write_doc2conll(doc, output)
    output_value = output.getvalue()
    assert output_value.endswith("\n\n")
    assert output_value.strip() == ENGLISH_SAMPLE

def test_write_doc2conll_append(tmp_path):
    doc = CoNLL.conll2doc(input_str=ENGLISH_SAMPLE)
    filename = tmp_path / "english.conll"
    CoNLL.write_doc2conll(doc, filename)
    CoNLL.write_doc2conll(doc, filename, mode="a")

    with open(filename) as fin:
        text = fin.read()
    expected = ENGLISH_SAMPLE + "\n\n" + ENGLISH_SAMPLE + "\n\n"
    assert text == expected

def test_doc_with_comments():
    """
    Test that a doc with comments gets converted back with comments
    """
    doc = CoNLL.conll2doc(input_str=RUSSIAN_SAMPLE)
    check_russian_doc(doc)

def test_unusual_misc():
    """
    The above RUSSIAN_SAMPLE resulted in a blank misc field in one particular implementation of the conll code
    (the below test would fail)
    """
    doc = CoNLL.conll2doc(input_str=RUSSIAN_SAMPLE)
    sentences = "{:C}".format(doc).split("\n\n")
    assert len(sentences) == 2
    sentence = sentences[0].split("\n")
    assert len(sentence) == 14

    for word in sentence:
        pieces = word.split("\t")
        assert len(pieces) == 1 or len(pieces) == 10
        if len(pieces) == 10:
            assert all(piece for piece in pieces)

def test_file():
    """
    Test loading a doc from a file
    """
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "russian.conll")
        with open(filename, "w", encoding="utf-8") as fout:
            fout.write(RUSSIAN_SAMPLE)
        doc = CoNLL.conll2doc(input_file=filename)
        check_russian_doc(doc)

def test_zip_file():
    """
    Test loading a doc from a zip file
    """
    with tempfile.TemporaryDirectory() as tempdir:
        zip_file = os.path.join(tempdir, "russian.zip")
        filename = "russian.conll"
        with ZipFile(zip_file, "w") as zout:
            with zout.open(filename, "w") as fout:
                fout.write(RUSSIAN_SAMPLE.encode())

        doc = CoNLL.conll2doc(input_file=filename, zip_file=zip_file)
        check_russian_doc(doc)

SIMPLE_NER = """
# text = Teferi's best friend is Karn
# sent_id = 0
1	Teferi	_	_	_	_	0	_	_	start_char=0|end_char=6|ner=S-PERSON
2	's	_	_	_	_	1	_	_	start_char=6|end_char=8|ner=O
3	best	_	_	_	_	2	_	_	start_char=9|end_char=13|ner=O
4	friend	_	_	_	_	3	_	_	start_char=14|end_char=20|ner=O
5	is	_	_	_	_	4	_	_	start_char=21|end_char=23|ner=O
6	Karn	_	_	_	_	5	_	_	start_char=24|end_char=28|ner=S-PERSON
""".strip()

def test_simple_ner_conversion():
    """
    Test that tokens get properly created with NER tags
    """
    doc = CoNLL.conll2doc(input_str=SIMPLE_NER)
    assert len(doc.sentences) == 1
    sentence = doc.sentences[0]
    assert len(sentence.tokens) == 6
    EXPECTED_NER = ["S-PERSON", "O", "O", "O", "O", "S-PERSON"]
    for token, ner in zip(sentence.tokens, EXPECTED_NER):
        assert token.ner == ner
        # check that the ner, start_char, end_char fields were not put on the token's misc
        # those should all be set as specific fields on the token
        assert not token.misc
        assert len(token.words) == 1
        # they should also not reach the word's misc field
        assert not token.words[0].misc

    conll = "{:C}".format(doc)
    assert conll == SIMPLE_NER

MWT_NER = """
# text = This makes John's headache worse
# sent_id = 0
1	This	_	_	_	_	0	_	_	start_char=0|end_char=4|ner=O
2	makes	_	_	_	_	1	_	_	start_char=5|end_char=10|ner=O
3-4	John's	_	_	_	_	_	_	_	start_char=11|end_char=17|ner=S-PERSON
3	John	_	_	_	_	2	_	_	_
4	's	_	_	_	_	3	_	_	_
5	headache	_	_	_	_	4	_	_	start_char=18|end_char=26|ner=O
6	worse	_	_	_	_	5	_	_	start_char=27|end_char=32|ner=O
""".strip()

def test_mwt_ner_conversion():
    """
    Test that tokens including MWT get properly created with NER tags

    Note that this kind of thing happens with the EWT tokenizer for English, for example
    """
    doc = CoNLL.conll2doc(input_str=MWT_NER)
    assert len(doc.sentences) == 1
    sentence = doc.sentences[0]
    assert len(sentence.tokens) == 5
    EXPECTED_NER = ["O", "O", "S-PERSON", "O", "O"]
    EXPECTED_WORDS = [1, 1, 2, 1, 1]
    for token, ner, expected_words in zip(sentence.tokens, EXPECTED_NER, EXPECTED_WORDS):
        assert token.ner == ner
        # check that the ner, start_char, end_char fields were not put on the token's misc
        # those should all be set as specific fields on the token
        assert not token.misc
        assert len(token.words) == expected_words
        # they should also not reach the word's misc field
        assert not token.words[0].misc

    conll = "{:C}".format(doc)
    assert conll == MWT_NER


# A random sentence from et_ewt-ud-train.conllu
# which we use to test the deps conversion for multiple deps
ESTONIAN_DEPS = """
# newpar
# sent_id = aia_foorum_37
# text = Sestpeale ei mõistagi neid, kes koduaias sortidega tegelevad.
1	Sestpeale	sest_peale	ADV	D	_	3	advmod	3:advmod	_
2	ei	ei	AUX	V	Polarity=Neg	3	aux	3:aux	_
3	mõistagi	mõistma	VERB	V	Connegative=Yes|Mood=Ind|Tense=Pres|VerbForm=Fin|Voice=Act	0	root	0:root	_
4	neid	tema	PRON	P	Case=Par|Number=Plur|Person=3|PronType=Prs	3	obj	3:obj|9:nsubj	SpaceAfter=No
5	,	,	PUNCT	Z	_	9	punct	9:punct	_
6	kes	kes	PRON	P	Case=Nom|Number=Plur|PronType=Int,Rel	9	nsubj	4:ref	_
7	koduaias	kodu_aed	NOUN	S	Case=Ine|Number=Sing	9	obl	9:obl	_
8	sortidega	sort	NOUN	S	Case=Com|Number=Plur	9	obl	9:obl	_
9	tegelevad	tegelema	VERB	V	Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act	4	acl:relcl	4:acl	SpaceAfter=No
10	.	.	PUNCT	Z	_	3	punct	3:punct	_
""".strip()

def test_deps_conversion():
    doc = CoNLL.conll2doc(input_str=ESTONIAN_DEPS)
    assert len(doc.sentences) == 1
    sentence = doc.sentences[0]
    assert len(sentence.tokens) == 10

    word = doc.sentences[0].words[3]
    assert word.deps == "3:obj|9:nsubj"

    conll = "{:C}".format(doc)
    assert conll == ESTONIAN_DEPS

ESTONIAN_EMPTY_DEPS = """
# sent_id = ewtb2_000035_15
# text = Ja paari aasta pärast rôômalt maasikatele ...
1	Ja	ja	CCONJ	J	_	3	cc	5.1:cc	_
2	paari	paar	NUM	N	Case=Gen|Number=Sing|NumForm=Word|NumType=Card	3	nummod	3:nummod	_
3	aasta	aasta	NOUN	S	Case=Gen|Number=Sing	0	root	5.1:obl	_
4	pärast	pärast	ADP	K	AdpType=Post	3	case	3:case	_
5	rôômalt	rõõmsalt	ADV	D	Typo=Yes	3	advmod	5.1:advmod	Orphan=Yes|CorrectForm=rõõmsalt
5.1	panna	panema	VERB	V	VerbForm=Inf	_	_	0:root	Empty=5.1
6	maasikatele	maasikas	NOUN	S	Case=All|Number=Plur	3	obl	5.1:obl	Orphan=Yes
7	...	...	PUNCT	Z	_	3	punct	5.1:punct	_
""".strip()

def test_empty_deps_conversion():
    """
    Ideally we would be able to read & recreate the dependencies

    Currently that is not possible.  Perhaps it should be fixed.
    At the very least, we shouldn't fail horribly when reading this
    """
    doc = CoNLL.conll2doc(input_str=ESTONIAN_EMPTY_DEPS)
    assert len(doc.sentences) == 1
    sentence = doc.sentences[0]
    conll = "{:C}".format(doc)
    #assert len(sentence.tokens) == 10

    #word = doc.sentences[0].words[3]
    #assert word.deps == "3:obj|9:nsubj"

    #conll = "{:C}".format(doc)
    #assert conll == ESTONIAN_EMPTY_DOCS

