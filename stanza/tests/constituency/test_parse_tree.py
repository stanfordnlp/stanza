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


def test_yield_preterminals():
    text = "((S (VP (VB Unban)) (NP (NNP Mox) (NNP Opal))))"
    trees = tree_reader.read_trees(text)

    preterminals = list(trees[0].yield_preterminals())
    assert len(preterminals) == 3
    assert str(preterminals) == "[(VB Unban), (NNP Mox), (NNP Opal)]"

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

def test_rare_words():
    """
    Test getting the unique words from a tree
    """
    text="((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))  ((SBARQ (NP (DT this) (NN seat)) (. ?)))"

    trees = tree_reader.read_trees(text)

    words = Tree.get_rare_words(trees, 0.5)
    expected = ['Who', 'in', 'sits']
    assert words == expected

def test_common_words():
    """
    Test getting the unique words from a tree
    """
    text="((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))  ((SBARQ (NP (DT this) (NN seat)) (. ?)))"

    trees = tree_reader.read_trees(text)

    words = Tree.get_common_words(trees, 3)
    expected = ['?', 'seat', 'this']
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

def test_prune_none():
    text=["((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (-NONE- in) (NP (DT this) (NN seat))))) (. ?)))", # test one dead node
          "((SBARQ (WHNP (-NONE- Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))", # test recursive dead nodes
          "((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (-NONE- this) (-NONE- seat))))) (. ?)))"] # test all children dead
    expected=["(ROOT (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (NP (DT this) (NN seat))))) (. ?)))",
              "(ROOT (SBARQ (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))",
              "(ROOT (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in)))) (. ?)))"]

    for t, e in zip(text, expected):
        trees = tree_reader.read_trees(t)
        assert len(trees) == 1
        tree = trees[0].prune_none()
        assert e == str(tree)

def test_simplify_labels():
    text="( (SBARQ-FOO (WHNP-BAR (WP Who)) (SQ#ASDF (VP=1 (VBZ sits) (PP (IN in) (NP (DT this) (- -))))) (. ?)))"
    expected = "(ROOT (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (- -))))) (. ?)))"
    trees = tree_reader.read_trees(text)
    trees = [t.simplify_labels() for t in trees]
    assert len(trees) == 1
    assert expected == str(trees[0])

def test_remap_constituent_labels():
    text="(ROOT (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in)))) (. ?)))"
    expected="(ROOT (FOO (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in)))) (. ?)))"

    label_map = { "SBARQ": "FOO" }
    trees = tree_reader.read_trees(text)
    trees = [t.remap_constituent_labels(label_map) for t in trees]
    assert len(trees) == 1
    assert expected == str(trees[0])

def test_remap_constituent_words():
    text="(ROOT (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in)))) (. ?)))"
    expected="(ROOT (SBARQ (WHNP (WP unban)) (SQ (VP (VBZ mox) (PP (IN opal)))) (. ?)))"

    word_map = { "Who": "unban", "sits": "mox", "in": "opal" }
    trees = tree_reader.read_trees(text)
    trees = [t.remap_words(word_map) for t in trees]
    assert len(trees) == 1
    assert expected == str(trees[0])

def test_replace_words():
    text="(ROOT (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in)))) (. ?)))"
    expected="(ROOT (SBARQ (WHNP (WP unban)) (SQ (VP (VBZ mox) (PP (IN opal)))) (. ?)))"
    new_words = ["unban", "mox", "opal", "?"]

    trees = tree_reader.read_trees(text)
    assert len(trees) == 1
    tree = trees[0]
    new_tree = tree.replace_words(new_words)
    assert expected == str(new_tree)


def test_compound_constituents():
    # TODO: add skinny trees like this to the various transition tests
    text="((VP (VB Unban)))"
    trees = tree_reader.read_trees(text)
    assert Tree.get_compound_constituents(trees) == [('ROOT', 'VP')]

    text="(ROOT (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in)))) (. ?)))"
    trees = tree_reader.read_trees(text)
    assert Tree.get_compound_constituents(trees) == [('PP',), ('ROOT', 'SBARQ'), ('SQ', 'VP'), ('WHNP',)]

    text="((VP (VB Unban)))   (ROOT (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in)))) (. ?)))"
    trees = tree_reader.read_trees(text)
    assert Tree.get_compound_constituents(trees) == [('PP',), ('ROOT', 'SBARQ'), ('ROOT', 'VP'), ('SQ', 'VP'), ('WHNP',)]

def test_equals():
    """
    Check one tree from the actual dataset for ==

    when built with compound Open, this didn't work because of a silly bug
    """
    text = "(ROOT (S (NP (DT The) (NNP Arizona) (NNPS Corporations) (NNP Commission)) (VP (VBD authorized) (NP (NP (DT an) (ADJP (CD 11.5)) (NN %) (NN rate) (NN increase)) (PP (IN at) (NP (NNP Tucson) (NNP Electric) (NNP Power) (NNP Co.))) (, ,) (UCP (ADJP (ADJP (RB substantially) (JJR lower)) (SBAR (IN than) (S (VP (VBN recommended) (NP (JJ last) (NN month)) (PP (IN by) (NP (DT a) (NN commission) (NN hearing) (NN officer))))))) (CC and) (NP (NP (QP (RB barely) (PDT half)) (DT the) (NN rise)) (VP (VBN sought) (PP (IN by) (NP (DT the) (NN utility)))))))) (. .)))"

    trees = tree_reader.read_trees(text)
    assert len(trees) == 1
    tree = trees[0]

    assert tree == tree

    trees2 = tree_reader.read_trees(text)
    tree2 = trees2[0]

    assert tree is not tree2
    assert tree == tree2


# This tree was causing the model to barf on CTB7,
# although it turns out the problem was just the
# depth of the unary, not the list
CHINESE_LONG_LIST_TREE = """
(ROOT
 (IP
  (NP (NNP 证券法))
  (VP
   (PP
    (IN 对)
    (NP
     (DNP
      (NP
       (NP (NNP 中国))
       (NP
        (NN 证券)
        (NN 市场)))
      (DEC 的))
     (NP (NN 运作))))
   (, ，)
   (PP
    (PP
     (IN 从)
     (NP
      (NP (NN 股票))
      (NP (VV 发行) (EC 、) (VV 交易))))
    (, ，)
    (PP
     (VV 到)
     (NP
      (NP (NN 上市) (NN 公司) (NN 收购))
      (EC 、)
      (NP (NN 证券) (NN 交易所))
      (EC 、)
      (NP (NN 证券) (NN 公司))
      (EC 、)
      (NP (NN 登记) (NN 结算) (NN 机构))
      (EC 、)
      (NP (NN 交易) (NN 服务) (NN 机构))
      (EC 、)
      (NP (NN 证券业) (NN 协会))
      (EC 、)
      (NP (NN 证券) (NN 监督) (NN 管理) (NN 机构))
      (CC 和)
      (NP
       (DNP
        (NP (CP (CP (IP (VP (VV 违法))))))
        (DEC 的))
       (NP (NN 法律) (NN 责任))))))
   (ADVP (RB 都))
   (VP
    (VV 作)
    (AS 了)
    (NP
     (ADJP (JJ 详细))
     (NP (NN 规定)))))
  (. 。)))
"""

WEIRD_UNARY = """
  (DNP
    (NP (CP (CP (IP (VP (ASDF
      (NP (NN 上市) (NN 公司) (NN 收购))
      (EC 、)
      (NP (NN 证券) (NN 交易所))
      (EC 、)
      (NP (NN 证券) (NN 公司))
      (EC 、)
      (NP (NN 登记) (NN 结算) (NN 机构))
      (EC 、)
      (NP (NN 交易) (NN 服务) (NN 机构))
      (EC 、)
      (NP (NN 证券业) (NN 协会))
      (EC 、)
      (NP (NN 证券) (NN 监督) (NN 管理) (NN 机构))))))))
    (DEC 的))
"""


def test_count_unaries():
    trees = tree_reader.read_trees(CHINESE_LONG_LIST_TREE)
    assert len(trees) == 1
    assert trees[0].count_unary_depth() == 5

    trees = tree_reader.read_trees(WEIRD_UNARY)
    assert len(trees) == 1
    assert trees[0].count_unary_depth() == 5

def test_str_bracket_labels():
    text = "((S (VP (VB Unban)) (NP (NNP Mox) (NNP Opal))))"
    expected = "(_ROOT (_S (_VP (_VB Unban )_VB )_VP (_NP (_NNP Mox )_NNP (_NNP Opal )_NNP )_NP )_S )_ROOT"

    trees = tree_reader.read_trees(text)
    assert len(trees) == 1
    assert "{:L}".format(trees[0]) == expected

def test_all_leaves_are_preterminals():
    text = "((S (VP (VB Unban)) (NP (NNP Mox) (NNP Opal))))"
    trees = tree_reader.read_trees(text)
    assert len(trees) == 1
    assert trees[0].all_leaves_are_preterminals()

    text = "((S (VP (VB Unban)) (NP (Mox) (NNP Opal))))"
    trees = tree_reader.read_trees(text)
    assert len(trees) == 1
    assert not trees[0].all_leaves_are_preterminals()

def test_pretty_print():
    """
    Pretty print a couple trees - newlines & indentation
    """
    text = "(ROOT (S (VP (VB Unban)) (NP (NNP Mox) (NNP Opal)))) (ROOT (S (NP (DT The) (NNP Arizona) (NNPS Corporations) (NNP Commission)) (VP (VBD authorized) (NP (NP (DT an) (ADJP (CD 11.5)) (NN %) (NN rate) (NN increase)) (PP (IN at) (NP (NNP Tucson) (NNP Electric)))))))"
    trees = tree_reader.read_trees(text)
    assert len(trees) == 2

    expected = """(ROOT
  (S
    (VP (VB Unban))
    (NP (NNP Mox) (NNP Opal))))
"""

    assert "{:P}".format(trees[0]) == expected

    expected = """(ROOT
  (S
    (NP (DT The) (NNP Arizona) (NNPS Corporations) (NNP Commission))
    (VP
      (VBD authorized)
      (NP
        (NP
          (DT an)
          (ADJP (CD 11.5))
          (NN %)
          (NN rate)
          (NN increase))
        (PP
          (IN at)
          (NP (NNP Tucson) (NNP Electric)))))))
"""
    assert "{:P}".format(trees[1]) == expected

    assert text == "{:O} {:O}".format(*trees)
