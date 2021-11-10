"""
Test the semgrex interface
"""

import pytest
import stanza
import stanza.server.semgrex as semgrex
from stanza.protobuf import SemgrexRequest
from stanza.models.common.doc import Document

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
    with semgrex.Semgrex(classpath="$CLASSPATH") as sem:
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
