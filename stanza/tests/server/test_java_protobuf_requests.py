import tempfile

import pytest

from stanza.models.common.utils import misc_to_space_after, space_after_to_misc
from stanza.models.constituency import tree_reader
from stanza.server import java_protobuf_requests
from stanza.tests import *
from stanza.utils.conll import CoNLL
from stanza.protobuf import DependencyGraph

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def check_tree(proto_tree, py_tree, py_score):
    tree, tree_score = java_protobuf_requests.from_tree(proto_tree)
    assert tree_score == py_score
    assert tree == py_tree

def test_build_tree():
    text="((S (VP (VB Unban)) (NP (NNP Mox) (NNP Opal))))\n( (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"
    trees = tree_reader.read_trees(text)
    assert len(trees) == 2

    for tree in trees:
        proto_tree = java_protobuf_requests.build_tree(trees[0], 1.0)
        check_tree(proto_tree, trees[0], 1.0)


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


def test_convert_networkx_graph():
    doc = CoNLL.conll2doc(input_str=ESTONIAN_EMPTY_DEPS, ignore_gapping=False)
    deps = doc.sentences[0]._enhanced_dependencies

    graph = DependencyGraph()
    java_protobuf_requests.convert_networkx_graph(graph, doc.sentences[0], 0)
    assert len(graph.rootNode) == 1
    assert graph.rootNode[0] == 0
    nodes = sorted([(x.index, x.emptyIndex) for x in graph.node])
    expected_nodes = [(1,0), (2,0), (3,0), (4,0), (5,0), (5,1), (6,0), (7,0)]
    assert nodes == expected_nodes

    edges = [(x.target, x.dep) for x in graph.edge if x.source == 5 and x.sourceEmpty == 1]
    edges = sorted(edges)
    expected_edges = [(1, 'cc'), (3, 'obl'), (5, 'advmod'), (6, 'obl'), (7, 'punct')]
    assert edges == expected_edges

ENGLISH_NBSP_SAMPLE="""
# sent_id = newsgroup-groups.google.com_n3td3v_e874a1e5eb995654_ENG_20060120_052200-0011
# text = Please note that neither the e-mail address nor name of the sender have been verified.
1	Please	please	INTJ	UH	_	2	discourse	_	_
2	note	note	VERB	VB	Mood=Imp|VerbForm=Fin	0	root	_	_
3	that	that	SCONJ	IN	_	15	mark	_	_
4	neither	neither	CCONJ	CC	_	7	cc:preconj	_	_
5	the	the	DET	DT	Definite=Def|PronType=Art	7	det	_	_
6	e-mail	e-mail	NOUN	NN	Number=Sing	7	compound	_	_
7	address	address	NOUN	NN	Number=Sing	15	nsubj:pass	_	_
8	nor	nor	CCONJ	CC	_	9	cc	_	_
9	name	name	NOUN	NN	Number=Sing	7	conj	_	_
10	of	of	ADP	IN	_	12	case	_	_
11	the	the	DET	DT	Definite=Def|PronType=Art	12	det	_	_
12	sender	sender	NOUN	NN	Number=Sing	7	nmod	_	_
13	have	have	AUX	VBP	Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin	15	aux	_	SpacesAfter=\\u00A0
14	been	be	AUX	VBN	Tense=Past|VerbForm=Part	15	aux:pass	_	_
15	verified	verify	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	2	ccomp	_	SpaceAfter=No
16	.	.	PUNCT	.	_	2	punct	_	_
""".strip()

def test_nbsp_doc():
    """
    Test that the space conversion methods will convert to and from NBSP
    """
    doc = CoNLL.conll2doc(input_str=ENGLISH_NBSP_SAMPLE)

    assert doc.sentences[0].text == "Please note that neither the e-mail address nor name of the sender have been verified."
    assert doc.sentences[0].tokens[12].spaces_after == " "
    assert misc_to_space_after("SpacesAfter=\\u00A0") == ' '
    assert space_after_to_misc(' ') == "SpacesAfter=\\u00A0"

    conllu = "{:C}".format(doc)
    assert conllu == ENGLISH_NBSP_SAMPLE
