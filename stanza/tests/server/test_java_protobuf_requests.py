import tempfile

import pytest

from stanza.models.constituency import tree_reader
from stanza.server import java_protobuf_requests
from stanza.tests import *

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
