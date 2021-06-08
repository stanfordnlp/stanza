"""
Basic tests of the data conversion
"""
import pytest

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
    conll = CoNLL.convert_dict(DICT)
    assert conll == CONLL

def test_dict_to_doc_and_doc_to_dict():
    """
    Test the conversion from raw dict to Document and back
    This code path will first turn start_char|end_char into start_char & end_char fields in the Document
    That version to a dict will have separate fields for each of those
    Finally, the conversion from that dict to a list of conll entries should convert that back to misc
    """
    doc = Document(DICT)
    dicts = doc.to_dict()
    dicts_tupleid = []
    for sentence in dicts:
        items = []
        for item in sentence:
            item['id'] = item['id'] if isinstance(item['id'], tuple) else (item['id'], )
            items.append(item)
        dicts_tupleid.append(items)
    conll = CoNLL.convert_dict(DICT)
    assert conll == CONLL

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
""".strip()


def test_doc_with_comments():
    """
    Test that a doc with comments gets converted back with comments
    """
    lines = RUSSIAN_SAMPLE.split("\n")

    doc = CoNLL.conll2doc(input_str=RUSSIAN_SAMPLE)
    assert len(doc.sentences) == 1
    assert len(doc.sentences[0].comments) == 3
    assert lines[0] == doc.sentences[0].comments[0]
    assert lines[1] == doc.sentences[0].comments[1]
    assert lines[2] == doc.sentences[0].comments[2]

    sentences = CoNLL.doc2conll(doc)
    assert len(sentences) == 1

    sentence = sentences[0]
    assert len(sentence) == 14
    assert lines[0] == sentence[0]
    assert lines[1] == sentence[1]
    assert lines[2] == sentence[2]

def test_unusual_misc():
    """
    The above RUSSIAN_SAMPLE resulted in a blank misc field in one particular implementation of the conll code
    (the below test would fail)
    """
    doc = CoNLL.conll2doc(input_str=RUSSIAN_SAMPLE)
    sentences = CoNLL.doc2conll(doc)
    assert len(sentences) == 1
    assert len(sentences[0]) == 14

    for word in sentences[0]:
        pieces = word.split("\t")
        assert len(pieces) == 1 or len(pieces) == 10
        if len(pieces) == 10:
            assert all(piece for piece in pieces)
