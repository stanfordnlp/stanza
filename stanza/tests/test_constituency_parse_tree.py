import pytest

from stanza.models.constituency.parse_tree import Tree
from stanza.models.constituency import tree_reader

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

def test_unique_labels():
    """
    Test getting the unique labels from a tree

    Assumes tree_reader works, which should be fine since it is tested elsewhere
    """
    text="((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?))) ((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"

    trees = tree_reader.read_trees(text)

    labels = Tree.get_unique_constituent_labels(trees)
    expected = ['NP', 'PP', 'ROOT', 'SBARQ', 'SQ', 'VP', 'WHNP']
    assert labels == expected

def test_unique_tags():
    """
    Test getting the unique tags from a tree
    """
    text="((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"

    trees = tree_reader.read_trees(text)

    tags = Tree.get_unique_tags(trees)
    expected = ['.', 'DT', 'IN', 'NN', 'VBZ', 'WP']
    assert tags == expected


def test_unique_words():
    """
    Test getting the unique words from a tree
    """
    text="((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"

    trees = tree_reader.read_trees(text)

    words = Tree.get_unique_words(trees)
    expected = ['?', 'Who', 'in', 'seat', 'sits', 'this']
    assert words == expected

def test_root_labels():
    text="( (SBARQ-FOO (WHNP-BAR (WP Who)) (SQ#ASDF (VP=1 (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"
    trees = tree_reader.read_trees(text)
    assert ["ROOT"] == Tree.get_root_labels(trees)

    text=("( (SBARQ-FOO (WHNP-BAR (WP Who)) (SQ#ASDF (VP=1 (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))" +
          "( (SBARQ-FOO (WHNP-BAR (WP Who)) (SQ#ASDF (VP=1 (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))" +
          "( (SBARQ-FOO (WHNP-BAR (WP Who)) (SQ#ASDF (VP=1 (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))")
    trees = tree_reader.read_trees(text)
    assert ["ROOT"] == Tree.get_root_labels(trees)

    text="(FOO) (BAR)"
    trees = tree_reader.read_trees(text)
    assert ["BAR", "FOO"] == Tree.get_root_labels(trees)
