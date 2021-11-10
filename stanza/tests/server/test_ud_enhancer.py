import pytest
import stanza
from stanza.tests import *

from stanza.models.common.doc import Document
import stanza.server.ud_enhancer as ud_enhancer

pytestmark = [pytest.mark.pipeline]

def check_edges(graph, source, target, num, isExtra=None):
    edges = [edge for edge in graph.edge if edge.source == source and edge.target == target]
    assert len(edges) == num
    if num == 1:
        assert edges[0].isExtra == isExtra

def test_one_sentence():
    nlp = stanza.Pipeline(dir=TEST_MODELS_DIR, processors="tokenize,pos,lemma,depparse")
    doc = nlp("This is the car that I bought")
    result = ud_enhancer.process_doc(doc, language="en", pronouns_pattern=None)

    assert len(result.sentence) == 1
    sentence = result.sentence[0]

    basic = sentence.basicDependencies
    assert len(basic.node) == 7
    assert len(basic.edge) == 6
    check_edges(basic, 4, 7, 1, False)
    check_edges(basic, 7, 4, 0)

    enhanced = sentence.enhancedDependencies
    assert len(enhanced.node) == 7
    assert len(enhanced.edge) == 7
    check_edges(enhanced, 4, 7, 1, False)
    # this is the new edge
    check_edges(enhanced, 7, 4, 1, True)
