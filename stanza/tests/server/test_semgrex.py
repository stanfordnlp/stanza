"""
Test the semgrex interface
"""

import subprocess

import pytest
import stanza
import stanza.server.semgrex as semgrex
from stanza.models.common.doc import Document
from stanza.protobuf import SemgrexRequest, SemgrexResponse
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


def check_indices(response):
    """
    Check that the sentence and semgrex indices agree with each other

    This is for unsorted results, where every sentence has a result
    for every semgrex, so the positions are the indices.

    The indices on SentenceResult and PatternResult are only filled in
    by versions of CoreNLP which have them, so they are only checked
    when present.  The indices on each Match are always filled in
    """
    for sentence_idx, sentence_result in enumerate(response.sentence):
        if sentence_result.HasField("sentenceIndex"):
            assert sentence_result.sentenceIndex == sentence_idx
        for semgrex_idx, pattern_result in enumerate(sentence_result.pattern):
            if pattern_result.HasField("semgrexIndex"):
                assert pattern_result.semgrexIndex == semgrex_idx
            for match in pattern_result.match:
                assert match.sentenceIndex == sentence_idx
                assert match.semgrexIndex == semgrex_idx

def check_response(response, response_len=1, semgrex_len=1, source_index=1, target_index=3, reln='obj'):
    assert len(response.sentence) == response_len
    check_indices(response)
    assert len(response.sentence[0].pattern) == semgrex_len
    for pattern_result in response.sentence[0].pattern:
        assert len(pattern_result.match) == 1
        assert pattern_result.match[0].matchIndex == source_index
        for match in pattern_result.match:
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
    assert len(response.sentence) == 1
    assert len(response.sentence[0].pattern) == 1
    assert len(response.sentence[0].pattern[0].match) == 1
    match = response.sentence[0].pattern[0].match[0]
    # there should be a named node...
    assert len(match.node) == 1
    assert match.node[0].name == 'root'
    assert match.node[0].matchIndex == 2

    # ... and a named edge
    assert len(match.edge) == 1
    assert match.edge[0].source == 1
    assert match.edge[0].target == 2
    assert match.edge[0].reln == "_"

EXPECTED_ONE_SENTENCE_MATCH = """
# text = Unban Mox Opal!
# sent_id = 0
# semgrex pattern = |{cpos:PROPN}=source <=zzz {ner:GEM}=target| matched at 2:Mox  source=2:Mox target=3:Opal
# highlight tokens = 2 3
# highlight deprels = 2
1	Unban	unban	VERB	VB	Mood=Imp|VerbForm=Fin	0	root	_	start_char=0|end_char=5
2	Mox	Mox	PROPN	NNP	Number=Sing	3	compound	_	start_char=6|end_char=9
3	Opal	Opal	PROPN	NNP	Number=Sing	1	obj	_	SpaceAfter=No|start_char=10|end_char=14|ner=GEM
4	!	!	PUNCT	.	_	1	punct	_	SpaceAfter=No|start_char=14|end_char=15
""".strip()

def test_ner_annotated():
    semgrex_pattern = "{cpos:PROPN}=source <=zzz {ner:GEM}=target"
    # not using the existing ONE_SENTENCE_DOC as the Document may be mutated
    doc = Document(TEST_ONE_SENTENCE, "Unban Mox Opal!")
    response = semgrex.process_doc(doc, semgrex_pattern)
    doc = semgrex.annotate_doc(doc, response, semgrex_pattern, True, False)
    formatted = "{:C}".format(doc).strip()
    assert formatted == EXPECTED_ONE_SENTENCE_MATCH

EXPECTED_ONE_SENTENCE_NO_MATCH = """
# text = Unban Mox Opal!
# sent_id = 0
# semgrex pattern = |{cpos:ZZZZ}| did not match!
1	Unban	unban	VERB	VB	Mood=Imp|VerbForm=Fin	0	root	_	start_char=0|end_char=5
2	Mox	Mox	PROPN	NNP	Number=Sing	3	compound	_	start_char=6|end_char=9
3	Opal	Opal	PROPN	NNP	Number=Sing	1	obj	_	SpaceAfter=No|start_char=10|end_char=14|ner=GEM
4	!	!	PUNCT	.	_	1	punct	_	SpaceAfter=No|start_char=14|end_char=15
""".strip()

def test_not_annotated():
    semgrex_pattern = "{cpos:ZZZZ}"
    # not using the existing ONE_SENTENCE_DOC as the Document may be mutated
    doc = Document(TEST_ONE_SENTENCE, "Unban Mox Opal!")
    response = semgrex.process_doc(doc, semgrex_pattern)
    doc = semgrex.annotate_doc(doc, response, semgrex_pattern, False, False)
    formatted = "{:C}".format(doc).strip()
    assert formatted == EXPECTED_ONE_SENTENCE_NO_MATCH


def test_empty_not_annotated():
    """
    If there are no responses and match_only is set, the returned doc should be empty
    """
    semgrex_pattern = "{cpos:ZZZZ}"
    # not using the existing ONE_SENTENCE_DOC as the Document may be mutated
    doc = Document(TEST_ONE_SENTENCE, "Unban Mox Opal!")
    response = semgrex.process_doc(doc, semgrex_pattern)
    doc = semgrex.annotate_doc(doc, response, semgrex_pattern, True, False)
    formatted = "{:C}".format(doc).strip()
    assert formatted == ""

def test_only_not_annotated():
    semgrex_pattern = "{cpos:ZZZZ}"
    # not using the existing ONE_SENTENCE_DOC as the Document may be mutated
    doc = Document(TEST_ONE_SENTENCE, "Unban Mox Opal!")
    response = semgrex.process_doc(doc, semgrex_pattern)
    doc = semgrex.annotate_doc(doc, response, semgrex_pattern, False, True)
    formatted = "{:C}".format(doc).strip()
    assert formatted == EXPECTED_ONE_SENTENCE_NO_MATCH


# An empty word 5.1 for the elided "likes", which is only in the enhanced graph
ENHANCED_SENTENCE = """
# text = Sue likes coffee and Bill tea.
1	Sue	Sue	PROPN	NNP	Number=Sing	2	nsubj	2:nsubj	_
2	likes	like	VERB	VBZ	Number=Sing|Person=3|Tense=Pres	0	root	0:root	_
3	coffee	coffee	NOUN	NN	Number=Sing	2	obj	2:obj	_
4	and	and	CCONJ	CC	_	5	cc	5.1:cc	_
5	Bill	Bill	PROPN	NNP	Number=Sing	2	conj	5.1:nsubj	_
5.1	likes	like	VERB	VBZ	Number=Sing|Person=3|Tense=Pres	_	_	2:conj:and	CopyOf=2
6	tea	tea	NOUN	NN	Number=Sing	5	orphan	5.1:obj	SpaceAfter=No
7	.	.	PUNCT	.	_	2	punct	2:punct	_
""".lstrip()

# A sentence with no enhanced dependencies at all
BASIC_ONLY_SENTENCE = """
# text = Unban Mox Opal!
1	Unban	unban	VERB	VB	Mood=Imp|VerbForm=Fin	0	root	_	_
2	Mox	Mox	PROPN	NNP	Number=Sing	3	compound	_	_
3	Opal	Opal	PROPN	NNP	Number=Sing	1	obj	_	SpaceAfter=No
4	!	!	PUNCT	.	_	1	punct	_	_
""".lstrip()

def enhanced_doc():
    return CoNLL.conll2doc(input_str=ENHANCED_SENTENCE, ignore_gapping=False)

def mixed_doc():
    """
    One sentence with enhanced dependencies, one without
    """
    return CoNLL.conll2doc(input_str=ENHANCED_SENTENCE + "\n" + BASIC_ONLY_SENTENCE, ignore_gapping=False)

def test_enhanced_request():
    """
    The basic graph goes in graph and the enhanced graph in enhancedGraph,
    over one token list which has the empty words as well
    """
    request = semgrex.build_request(mixed_doc(), "{}", enhanced=True)
    assert len(request.query) == 2

    query = request.query[0]
    assert len(query.token) == 8
    assert [(token.index, token.emptyIndex) for token in query.token if token.emptyIndex] == [(5, 1)]
    assert len(query.graph.node) == 7
    assert len(query.enhancedGraph.node) == 8
    assert len(query.enhancedGraph.token) == 0
    assert any(edge.targetEmpty == 1 and edge.dep == "conj:and" for edge in query.enhancedGraph.edge)

    # no enhanced dependencies means no enhanced graph
    query = request.query[1]
    assert len(query.token) == 4
    assert not query.HasField("enhancedGraph")

def test_default_request_has_enhanced():
    """
    The enhanced graph is sent unless asked not to
    """
    request = semgrex.build_request(enhanced_doc(), "{}")
    query = request.query[0]
    assert len(query.token) == 8
    assert len(query.enhancedGraph.node) == 8

def test_no_enhanced_request():
    request = semgrex.build_request(enhanced_doc(), "{}", enhanced=False)
    query = request.query[0]
    assert len(query.token) == 7
    assert not query.HasField("enhancedGraph")

def test_enhanced_empty_root():
    """
    A match can start at an empty word, which only the enhanced graph has
    """
    response = semgrex.process_doc(enhanced_doc(), "{} !< {} <@enhanced {}", enhanced=True)
    assert len(response.sentence) == 1
    matches = response.sentence[0].pattern[0].match
    assert len(matches) == 1
    assert matches[0].matchIndex == 5
    assert matches[0].matchEmptyIndex == 1

def test_enhanced_edges():
    """
    Named edges say which graph they came from, and named nodes can be empty words
    """
    response = semgrex.process_doc(enhanced_doc(), "{word:likes}=l >nsubj=e1 {}=s >/conj.*/@enhanced=e2 {}=c", enhanced=True)
    matches = response.sentence[0].pattern[0].match
    assert len(matches) == 1
    match = matches[0]
    nodes = {node.name: (node.matchIndex, node.emptyIndex) for node in match.node}
    assert nodes == {"l": (2, 0), "s": (1, 0), "c": (5, 1)}
    edges = {edge.name: edge for edge in match.edge}
    assert edges["e1"].graph == SemgrexResponse.GraphName.BASIC
    assert edges["e2"].graph == SemgrexResponse.GraphName.ENHANCED
    assert edges["e2"].reln == "conj:and"
    assert (edges["e2"].target, edges["e2"].targetEmpty) == (5, 1)

def test_enhanced_context():
    with semgrex.Semgrex() as sem:
        response = sem.process(enhanced_doc(), "{word:Bill} <nsubj@enhanced {}=h")
    match = response.sentence[0].pattern[0].match[0]
    assert (match.node[0].matchIndex, match.node[0].emptyIndex) == (5, 1)

def test_mixed_doc_basic_pattern():
    """
    Sentences without enhanced dependencies are fine for patterns which
    only search the basic graph
    """
    response = semgrex.process_doc(mixed_doc(), "{}=source >obj {}=target", enhanced=True)
    assert len(response.sentence) == 2
    assert [match.matchIndex for match in response.sentence[0].pattern[0].match] == [2]
    assert [match.matchIndex for match in response.sentence[1].pattern[0].match] == [1]

def test_mixed_doc_enhanced_pattern():
    """
    A sentence without enhanced dependencies fails a pattern which
    searches the enhanced graph, rather than quietly giving wrong answers.
    With an empty enhanced graph, this pattern would match every word
    of the second sentence
    """
    with pytest.raises(subprocess.CalledProcessError):
        semgrex.process_doc(mixed_doc(), "{}=w < {} !<@enhanced {}", enhanced=True)

def test_enhanced_pattern_needs_enhanced():
    """
    Searching the enhanced graph without sending it is an error from CoreNLP
    """
    with pytest.raises(subprocess.CalledProcessError):
        semgrex.process_doc(enhanced_doc(), "{} <@enhanced {}", enhanced=False)

EXPECTED_ENHANCED_MATCH = """
# text = Sue likes coffee and Bill tea.
# sent_id = 0
# semgrex pattern = |{word:likes}=l >/conj.*/@enhanced=e {}=c| matched at 2:likes  l=2:likes c=5.1:likes
# highlight tokens = 2 5.1
# highlight deprels = 5.1
1	Sue	Sue	PROPN	NNP	Number=Sing	2	nsubj	2:nsubj	_
2	likes	like	VERB	VBZ	Number=Sing|Person=3|Tense=Pres	0	root	0:root	_
3	coffee	coffee	NOUN	NN	Number=Sing	2	obj	2:obj	_
4	and	and	CCONJ	CC	_	5	cc	5.1:cc	_
5	Bill	Bill	PROPN	NNP	Number=Sing	2	conj	5.1:nsubj	_
5.1	likes	like	VERB	VBZ	Number=Sing|Person=3|Tense=Pres	_	_	2:conj:and	CopyOf=2
6	tea	tea	NOUN	NN	Number=Sing	5	orphan	5.1:obj	SpaceAfter=No
7	.	.	PUNCT	.	_	2	punct	2:punct	_
""".strip()

def test_enhanced_annotated():
    """
    The annotated CoNLL-U keeps the empty word and the enhanced
    dependencies, and refers to the empty word as 5.1
    """
    doc = enhanced_doc()
    pattern = "{word:likes}=l >/conj.*/@enhanced=e {}=c"
    response = semgrex.process_doc(doc, pattern, enhanced=True)
    annotated = semgrex.annotate_doc(doc, response, pattern, matches_only=True, exclude_matches=False)
    assert "{:C}".format(annotated) == EXPECTED_ENHANCED_MATCH
