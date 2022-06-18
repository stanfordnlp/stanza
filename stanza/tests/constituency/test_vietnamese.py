"""
A few tests for Vietnamese parsing, which has some difficulties related to spaces in words

Technically some other languages can have this, too, like that one French token
"""

import tempfile

import pytest

from stanza.models.common import pretrain
from stanza.models.constituency import tree_reader

from stanza.tests import *
from stanza.tests.constituency.test_trainer import build_trainer

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

# just one tree so far, but maybe we can add more
VI_TREEBANK               = '(ROOT (S-TTL (NP (" ") (N-H Đảo) (Np Đài Loan) (" ") (PP (E-H ở) (NP (N-H đồng bằng) (NP (N-H sông) (Np Cửu Long))))) (. .)))'

VI_TREEBANK_UNDERSCORE    = '(ROOT (S-TTL (NP (" ") (N-H Đảo) (Np Đài_Loan) (" ") (PP (E-H ở) (NP (N-H đồng_bằng) (NP (N-H sông) (Np Cửu_Long))))) (. .)))'

VI_TREEBANK_SIMPLE        = '(ROOT (S (NP (" ") (N Đảo) (Np Đài Loan) (" ") (PP (E ở) (NP (N đồng bằng) (NP (N sông) (Np Cửu Long))))) (. .)))'

EXPECTED_LABELED_BRACKETS = '(_ROOT (_S (_NP (_" " )_" (_N Đảo )_N (_Np Đài_Loan )_Np (_" " )_" (_PP (_E ở )_E (_NP (_N đồng_bằng )_N (_NP (_N sông )_N (_Np Cửu_Long )_Np )_NP )_NP )_PP )_NP (_. . )_. )_S )_ROOT'


def test_read_vi_tree():
    """
    Test that an individual tree with spaces in the leaves is being processed as we expect
    """
    text = VI_TREEBANK.split("\n")[0]
    trees = tree_reader.read_trees(text)
    assert len(trees) == 1
    assert str(trees[0]) == text
    # this is the first NP
    #   the third node of that NP, eg (Np Đài Loan)
    node = trees[0].children[0].children[0].children[2]
    assert node.is_preterminal()
    assert node.children[0].label == "Đài Loan"

VI_EMBEDDING = """
4 4
Đảo          0.11 0.21 0.31 0.41
Đài Loan     0.12 0.22 0.32 0.42
đồng bằng    0.13 0.23 0.33 0.43
sông         0.14 0.24 0.34 0.44
""".strip()

def test_vi_embedding():
    """
    Test that a VI embedding's words are correctly found when processing trees
    """
    trees = tree_reader.read_trees(VI_TREEBANK)
    words = set(trees[0].leaf_labels())

    with tempfile.TemporaryDirectory() as tempdir:
        emb_filename = os.path.join(tempdir, "emb.txt")
        pt_filename = os.path.join(tempdir, "emb.pt")
        with open(emb_filename, "w", encoding="utf-8") as fout:
            fout.write(VI_EMBEDDING)
        pt = pretrain.Pretrain(filename=pt_filename, vec_filename=emb_filename, save_to_file=True)
        pt.load()

        trainer = build_trainer(pt_filename)
        model = trainer.model

    assert model.num_words_known(words) == 4


def test_space_formatting():
    """
    By default, spaces are left as spaces, but there is a format option to change spaces
    """
    text = VI_TREEBANK.split("\n")[0]
    trees = tree_reader.read_trees(text)
    assert len(trees) == 1
    assert str(trees[0]) == text

    assert "{:_}".format(trees[0]) == VI_TREEBANK_UNDERSCORE


def test_language_formatting():
    """
    Test turning the parse tree into a 'language' for GPT
    """
    text = VI_TREEBANK.split("\n")[0]
    trees = tree_reader.read_trees(text)
    trees = [t.prune_none().simplify_labels() for t in trees]
    assert len(trees) == 1
    assert str(trees[0]) == VI_TREEBANK_SIMPLE

    text = "{:L}".format(trees[0])
    assert text == EXPECTED_LABELED_BRACKETS

