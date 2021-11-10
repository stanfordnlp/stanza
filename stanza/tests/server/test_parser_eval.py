"""
Test the parser eval interface
"""

import pytest
import stanza
from stanza.models.constituency import tree_reader
from stanza.protobuf import EvaluateParserRequest, EvaluateParserResponse
from stanza.server.parser_eval import build_request, EvaluateParser
from stanza.tests.test_java_protobuf_requests import check_tree

from stanza.tests import *

pytestmark = [pytest.mark.travis, pytest.mark.client]

def build_one_tree_treebank():
    text = "((S (VP (VB Unban)) (NP (NNP Mox) (NNP Opal))))"
    trees = tree_reader.read_trees(text)
    assert len(trees) == 1
    gold = trees[0]
    prediction = (gold, 1.0)
    treebank = [(gold, [prediction])]
    return treebank

def test_build_request_one_tree():
    treebank = build_one_tree_treebank()
    request = build_request(treebank)

    assert len(request.treebank) == 1
    check_tree(request.treebank[0].gold, treebank[0][0], None)
    assert len(request.treebank[0].predicted) == 1
    check_tree(request.treebank[0].predicted[0], treebank[0][1][0][0], treebank[0][1][0][1])


def test_score_one_tree():
    treebank = build_one_tree_treebank()

    with EvaluateParser(classpath="$CLASSPATH") as ep:
        response = ep.process(treebank)
        assert response.f1 == pytest.approx(1.0)
