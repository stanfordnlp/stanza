"""
Test the semgrex interface
"""

import pytest
import stanza
import stanza.server.semgrex as semgrex
from stanza.models.common.doc import Document
from stanza.protobuf import SemgrexRequest
from stanza.utils.conll import CoNLL

from stanza.tests import *

pytestmark = [pytest.mark.travis, pytest.mark.client]

TEST_ONE_SENTENCE = [[
    {
        "id": 1,
        "text": "Unban",
        "lemma": "unban",
        "upos": "VERB",
        "xpos": "VB",
        "feats": "Mood=Imp|VerbForm=Fin",
        "head": 0,
        "deprel": "root",
        "misc": "start_char=0|end_char=5"
    },
    {
        "id": 2,
        "text": "Mox",
        "lemma": "Mox",
        "upos": "PROPN",
        "xpos": "NNP",
        "feats": "Number=Sing",
        "head": 3,
        "deprel": "compound",
        "misc": "start_char=6|end_char=9"
    },
    {
        "id": 3,
        "text": "Opal",
        "lemma": "Opal",
        "upos": "PROPN",
        "xpos": "NNP",
        "feats": "Number=Sing",
        "head": 1,
        "deprel": "obj",
        "misc": "start_char=10|end_char=14",
        "ner": "GEM"
    },
    {
        "id": 4,
        "text": "!",
        "lemma": "!",
        "upos": "PUNCT",
        "xpos": ".",
        "head": 1,
        "deprel": "punct",
        "misc": "start_char=14|end_char=15"
    }]]

TEST_TWO_SENTENCES = [[
    {
      "id": 1,
      "text": "Unban",
      "lemma": "unban",
      "upos": "VERB",
      "xpos": "VB",
      "feats": "Mood=Imp|VerbForm=Fin",
      "head": 0,
      "deprel": "root",
      "misc": "start_char=0|end_char=5"
    },
    {
      "id": 2,
      "text": "Mox",
      "lemma": "Mox",
      "upos": "PROPN",
      "xpos": "NNP",
      "feats": "Number=Sing",
      "head": 3,
      "deprel": "compound",
      "misc": "start_char=6|end_char=9"
    },
    {
      "id": 3,
      "text": "Opal",
      "lemma": "Opal",
      "upos": "PROPN",
      "xpos": "NNP",
      "feats": "Number=Sing",
      "head": 1,
      "deprel": "obj",
      "misc": "start_char=10|end_char=14"
    },
    {
      "id": 4,
      "text": "!",
      "lemma": "!",
      "upos": "PUNCT",
      "xpos": ".",
      "head": 1,
      "deprel": "punct",
      "misc": "start_char=14|end_char=15"
    }],
    [{
      "id": 1,
      "text": "Unban",
      "lemma": "unban",
      "upos": "VERB",
      "xpos": "VB",
      "feats": "Mood=Imp|VerbForm=Fin",
      "head": 0,
      "deprel": "root",
      "misc": "start_char=16|end_char=21"
    },
    {
      "id": 2,
      "text": "Mox",
      "lemma": "Mox",
      "upos": "PROPN",
      "xpos": "NNP",
      "feats": "Number=Sing",
      "head": 3,
      "deprel": "compound",
      "misc": "start_char=22|end_char=25"
    },
    {
      "id": 3,
      "text": "Opal",
      "lemma": "Opal",
      "upos": "PROPN",
      "xpos": "NNP",
      "feats": "Number=Sing",
      "head": 1,
      "deprel": "obj",
      "misc": "start_char=26|end_char=30"
    },
    {
      "id": 4,
      "text": "!",
      "lemma": "!",
      "upos": "PUNCT",
      "xpos": ".",
      "head": 1,
      "deprel": "punct",
      "misc": "start_char=30|end_char=31"
    }]]

ONE_SENTENCE_DOC = Document(TEST_ONE_SENTENCE, "Unban Mox Opal!")
TWO_SENTENCE_DOC = Document(TEST_TWO_SENTENCES, "Unban Mox Opal! Unban Mox Opal!")


def check_response(response, response_len=1, semgrex_len=1, source_index=1, target_index=3, reln='obj'):
    assert len(response.result) == response_len
    assert len(response.result[0].result) == semgrex_len
    for semgrex_result in response.result[0].result:
        assert len(semgrex_result.match) == 1
        assert semgrex_result.match[0].matchIndex == source_index
        for match in semgrex_result.match:
            assert len(match.node) == 2
            assert match.node[0].name == 'source'
            assert match.node[0].matchIndex == source_index
            assert match.node[1].name == 'target'
            assert match.node[1].matchIndex == target_index
            assert len(match.reln) == 1
            assert match.reln[0].name == 'zzz'
            assert match.reln[0].reln == reln

def test_multi():
    with semgrex.Semgrex() as sem:
        response = sem.process(ONE_SENTENCE_DOC, "{}=source >obj=zzz {}=target")
        check_response(response)
        response = sem.process(ONE_SENTENCE_DOC, "{}=source >obj=zzz {}=target")
        check_response(response)
        response = sem.process(TWO_SENTENCE_DOC, "{}=source >obj=zzz {}=target")
        check_response(response, response_len=2)

def test_single_sentence():
    response = semgrex.process_doc(ONE_SENTENCE_DOC, "{}=source >obj=zzz {}=target")
    check_response(response)

def test_two_semgrex():
    response = semgrex.process_doc(ONE_SENTENCE_DOC, "{}=source >obj=zzz {}=target", "{}=source >obj=zzz {}=target")
    check_response(response, semgrex_len=2)

def test_two_sentences():
    response = semgrex.process_doc(TWO_SENTENCE_DOC, "{}=source >obj=zzz {}=target")
    check_response(response, response_len=2)

def test_word_attribute():
    response = semgrex.process_doc(ONE_SENTENCE_DOC, "{word:Mox}=source <=zzz {word:Opal}=target")
    check_response(response, response_len=1, source_index=2, reln='compound')

def test_lemma_attribute():
    response = semgrex.process_doc(ONE_SENTENCE_DOC, "{lemma:Mox}=source <=zzz {lemma:Opal}=target")
    check_response(response, response_len=1, source_index=2, reln='compound')

def test_xpos_attribute():
    response = semgrex.process_doc(ONE_SENTENCE_DOC, "{tag:NNP}=source <=zzz {word:Opal}=target")
    check_response(response, response_len=1, source_index=2, reln='compound')
    response = semgrex.process_doc(ONE_SENTENCE_DOC, "{pos:NNP}=source <=zzz {word:Opal}=target")
    check_response(response, response_len=1, source_index=2, reln='compound')

def test_upos_attribute():
    response = semgrex.process_doc(ONE_SENTENCE_DOC, "{cpos:PROPN}=source <=zzz {word:Opal}=target")
    check_response(response, response_len=1, source_index=2, reln='compound')

def test_ner_attribute():
    response = semgrex.process_doc(ONE_SENTENCE_DOC, "{cpos:PROPN}=source <=zzz {ner:GEM}=target")
    check_response(response, response_len=1, source_index=2, reln='compound')

def test_hand_built_request():
    """
    Essentially a test program: the result should be a response with
    one match, two named nodes, one named relation
    """
    request = SemgrexRequest()
    request.semgrex.append("{}=source >obj=zzz {}=target")
    query = request.query.add()

    for idx, word in enumerate(['Unban', 'Mox', 'Opal']):
        token = query.token.add()
        token.word = word
        token.value = word

        node = query.graph.node.add()
        node.sentenceIndex = 1
        node.index = idx+1

    edge = query.graph.edge.add()
    edge.source = 1
    edge.target = 3
    edge.dep = 'obj'

    edge = query.graph.edge.add()
    edge.source = 3
    edge.target = 2
    edge.dep = 'compound'

    response = semgrex.send_semgrex_request(request)
    check_response(response)

BLANK_DEPENDENCY_SENTENCE = """
# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0007
# text = You wonder if he was manipulating the market with his bombing targets.
1	You	you	PRON	PRP	Case=Nom|Person=2|PronType=Prs	2	nsubj	_	_
2	wonder	wonder	VERB	VBP	Mood=Ind|Number=Sing|Person=2|Tense=Pres|VerbForm=Fin	1	_	_	_
3	if	if	SCONJ	IN	_	6	mark	_	_
4	he	he	PRON	PRP	Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs	6	nsubj	_	_
5	was	be	AUX	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	6	aux	_	_
6	manipulating	manipulate	VERB	VBG	Tense=Pres|VerbForm=Part	2	ccomp	_	_
7	the	the	DET	DT	Definite=Def|PronType=Art	8	det	_	_
8	market	market	NOUN	NN	Number=Sing	6	obj	_	_
9	with	with	ADP	IN	_	12	case	_	_
10	his	his	PRON	PRP$	Case=Gen|Gender=Masc|Number=Sing|Person=3|Poss=Yes|PronType=Prs	12	nmod:poss	_	_
11	bombing	bombing	NOUN	NN	Number=Sing	12	compound	_	_
12	targets	target	NOUN	NNS	Number=Plur	6	obl	_	SpaceAfter=No
13	.	.	PUNCT	.	_	2	punct	_	_
""".lstrip()


def test_blank_dependency():
    """
    A user / contributor sent a dependency file with blank dependency labels and twisted up roots
    """
    blank_dep_doc = CoNLL.conll2doc(input_str=BLANK_DEPENDENCY_SENTENCE)
    blank_dep_request = semgrex.build_request(blank_dep_doc, "{}=root <_=edge {}")
    response = semgrex.send_semgrex_request(blank_dep_request)
    assert len(response.result) == 1
    assert len(response.result[0].result) == 1
    assert len(response.result[0].result[0].match) == 1
    # there should be a named node...
    assert len(response.result[0].result[0].match[0].node) == 1
    assert response.result[0].result[0].match[0].node[0].name == 'root'
    assert response.result[0].result[0].match[0].node[0].matchIndex == 2

    # ... and a named edge
    assert len(response.result[0].result[0].match[0].edge) == 1
    assert response.result[0].result[0].match[0].edge[0].source == 1
    assert response.result[0].result[0].match[0].edge[0].target == 2
    assert response.result[0].result[0].match[0].edge[0].reln == "_"
