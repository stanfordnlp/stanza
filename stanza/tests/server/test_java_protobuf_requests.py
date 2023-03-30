import tempfile

import pytest

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
