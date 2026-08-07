import pytest
import stanza
from stanza.tests import *

from stanza.utils.datasets import prepare_tokenizer_treebank

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def test_has_space_after_no():
    assert prepare_tokenizer_treebank.has_space_after_no("SpaceAfter=No")
    assert prepare_tokenizer_treebank.has_space_after_no("UnbanMoxOpal=Yes|SpaceAfter=No")
    assert prepare_tokenizer_treebank.has_space_after_no("SpaceAfter=No|UnbanMoxOpal=Yes")
    assert not prepare_tokenizer_treebank.has_space_after_no("SpaceAfter=Yes")
    assert not prepare_tokenizer_treebank.has_space_after_no("CorrectSpaceAfter=No")
    assert not prepare_tokenizer_treebank.has_space_after_no("_")


def test_add_space_after_no():
    assert prepare_tokenizer_treebank.add_space_after_no("_") == "SpaceAfter=No"
    assert prepare_tokenizer_treebank.add_space_after_no("MoxOpal=Unban") == "MoxOpal=Unban|SpaceAfter=No"
    with pytest.raises(ValueError):
        prepare_tokenizer_treebank.add_space_after_no("SpaceAfter=No")

def test_remove_space_after_no():
    assert prepare_tokenizer_treebank.remove_space_after_no("SpaceAfter=No") == "_"
    assert prepare_tokenizer_treebank.remove_space_after_no("SpaceAfter=No|MoxOpal=Unban") == "MoxOpal=Unban"
    assert prepare_tokenizer_treebank.remove_space_after_no("MoxOpal=Unban|SpaceAfter=No") == "MoxOpal=Unban"
    with pytest.raises(ValueError):
        prepare_tokenizer_treebank.remove_space_after_no("_")

def read_test_doc(doc):
    sentences = [x.strip().split("\n") for x in doc.split("\n\n")]
    return sentences
