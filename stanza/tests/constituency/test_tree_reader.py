import pytest
from stanza.models.constituency import tree_reader

from stanza.tests import *

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def test_simple():
    """
    Tests reading two simple trees from the same text
    """
    text = "(VB Unban) (NNP Opal)"
    trees = tree_reader.read_trees(text)
    assert len(trees) == 2
    assert trees[0].is_preterminal()
    assert trees[0].label == 'VB'
    assert trees[0].children[0].label == 'Unban'
    assert trees[1].is_preterminal()
    assert trees[1].label == 'NNP'
    assert trees[1].children[0].label == 'Opal'

def test_newlines():
    """
    The same test should work if there are newlines
    """
    text = "(VB Unban)\n\n(NNP Opal)"
    trees = tree_reader.read_trees(text)
    assert len(trees) == 2

def test_complicated():
    """
    A more complicated tree that should successfully read
    """
    text="( (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"
    trees = tree_reader.read_trees(text)
    assert len(trees) == 1
    tree = trees[0]
    assert not tree.is_leaf()
    assert not tree.is_preterminal()
    assert tree.label == 'ROOT'
    assert len(tree.children) == 1
    assert tree.children[0].label == 'SBARQ'
    assert len(tree.children[0].children) == 3
    assert [x.label for x in tree.children[0].children] == ['WHNP', 'SQ', '.']
    # etc etc

def test_one_word():
    """
    Check that one node trees are correctly read

    probably not super relevant for the parsing use case
    """
    text="(FOO) (BAR)"
    trees = tree_reader.read_trees(text)
    assert len(trees) == 2

    assert trees[0].is_leaf()
    assert trees[0].label == 'FOO'

    assert trees[1].is_leaf()
    assert trees[1].label == 'BAR'
