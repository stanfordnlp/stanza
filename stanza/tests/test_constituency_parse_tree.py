import pytest

from stanza.models.constituency.parse_tree import Tree

from stanza.tests import *

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def test_leaf_preterminal():
    foo = Tree(label="foo")
    assert foo.is_leaf()
    assert not foo.is_preterminal()
    assert len(foo.children) == 0
    assert str(foo) == 'foo'

    bar = Tree(label="bar", children=foo)
    assert not bar.is_leaf()
    assert bar.is_preterminal()
    assert len(bar.children) == 1
    assert str(bar) == "(bar foo)"

    baz = Tree(label="baz", children=[bar])
    assert not baz.is_leaf()
    assert not baz.is_preterminal()
    assert len(baz.children) == 1
    assert str(baz) == "(baz (bar foo))"


def test_depth():
    text = "(foo) ((S (VP (VB Unban)) (NP (NNP Mox) (NNP Opal))))"
    trees = tree_reader.read_trees(text)
    assert trees[0].depth() == 0
    assert trees[1].depth() == 4
