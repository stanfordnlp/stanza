import pytest

from stanza import Pipeline
from stanza.models.constituency import tree_reader
from stanza.models.constituency import utils

from stanza.tests import *

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]


@pytest.fixture(scope="module")
def pipeline():
    return Pipeline(dir=TEST_MODELS_DIR, lang="en", processors="tokenize, pos", tokenize_pretokenized=True)



def test_xpos_retag(pipeline):
    """
    Test using the English tagger that trees will be correctly retagged by read_trees using xpos
    """
    text = "((S (VP (X Find)) (NP (X Mox) (X Opal))))   ((S (NP (X Ragavan)) (VP (X steals) (NP (X important) (X cards)))))"
    expected = "((S (VP (VB Find)) (NP (NNP Mox) (NN Opal)))) ((S (NP (NNP Ragavan)) (VP (VBZ steals) (NP (JJ important) (NNS cards)))))"

    trees = tree_reader.read_trees(text)

    new_trees = utils.retag_trees(trees, pipeline, xpos=True)
    assert new_trees == tree_reader.read_trees(expected)



def test_upos_retag(pipeline):
    """
    Test using the English tagger that trees will be correctly retagged by read_trees using upos
    """
    text = "((S (VP (X Find)) (NP (X Mox) (X Opal))))   ((S (NP (X Ragavan)) (VP (X steals) (NP (X important) (X cards)))))"
    expected = "((S (VP (VERB Find)) (NP (PROPN Mox) (NOUN Opal)))) ((S (NP (PROPN Ragavan)) (VP (VERB steals) (NP (ADJ important) (NOUN cards)))))"

    trees = tree_reader.read_trees(text)

    new_trees = utils.retag_trees(trees, pipeline, xpos=False)
    assert new_trees == tree_reader.read_trees(expected)


def test_replace_tags():
    """
    Test the underlying replace_tags method

    Also tests that the method throws exceptions when it is supposed to
    """
    text = "((S (VP (X Find)) (NP (X Mox) (X Opal))))"
    expected = "((S (VP (A Find)) (NP (B Mox) (C Opal))))"

    trees = tree_reader.read_trees(text)

    new_tags = ["A", "B", "C"]
    new_tree = utils.replace_tags(trees[0], new_tags)

    assert new_tree == tree_reader.read_trees(expected)[0]

    with pytest.raises(ValueError):
        new_tags = ["A", "B"]
        new_tree = utils.replace_tags(trees[0], new_tags)

    with pytest.raises(ValueError):
        new_tags = ["A", "B", "C", "D"]
        new_tree = utils.replace_tags(trees[0], new_tags)

