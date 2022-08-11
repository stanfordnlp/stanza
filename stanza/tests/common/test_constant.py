"""
Test the conversion to lcodes and splitting of dataset names
"""

import tempfile

import pytest

import stanza
from stanza.models.common.constant import treebank_to_short_name, lang_to_langcode, is_right_to_left, two_to_three_letters, langlower2lcode
from stanza.tests import *

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def test_treebank():
    """
    Test the entire treebank name conversion
    """
    # conversion of a UD_ name
    assert "hi_hdtb" == treebank_to_short_name("UD_Hindi-HDTB")
    # conversion of names without UD
    assert "hi_fire2013" == treebank_to_short_name("Hindi-fire2013")
    assert "hi_fire2013" == treebank_to_short_name("Hindi-Fire2013")
    assert "hi_fire2013" == treebank_to_short_name("Hindi-FIRE2013")
    # already short names are generally preserved
    assert "hi_fire2013" == treebank_to_short_name("hi-fire2013")
    assert "hi_fire2013" == treebank_to_short_name("hi_fire2013")
    # a special case
    assert "zh-hant_pud" == treebank_to_short_name("UD_Chinese-PUD")
    # a special case already converted once
    assert "zh-hant_pud" == treebank_to_short_name("zh-hant_pud")
    assert "zh-hant_pud" == treebank_to_short_name("zh-hant-pud")
    assert "zh-hans_gsdsimp" == treebank_to_short_name("zh-hans_gsdsimp")

    assert "wo_masakhane" == treebank_to_short_name("wo_masakhane")
    assert "wo_masakhane" == treebank_to_short_name("wol_masakhane")
    assert "wo_masakhane" == treebank_to_short_name("Wol_masakhane")
    assert "wo_masakhane" == treebank_to_short_name("wolof_masakhane")
    assert "wo_masakhane" == treebank_to_short_name("Wolof_masakhane")

def test_lang_to_langcode():
    assert "hi" == lang_to_langcode("Hindi")
    assert "hi" == lang_to_langcode("HINDI")
    assert "hi" == lang_to_langcode("hindi")
    assert "hi" == lang_to_langcode("HI")
    assert "hi" == lang_to_langcode("hi")

def test_right_to_left():
    assert is_right_to_left("ar")
    assert is_right_to_left("Arabic")

    assert not is_right_to_left("en")
    assert not is_right_to_left("English")

def test_two_to_three():
    assert lang_to_langcode("Wolof") == "wo"
    assert lang_to_langcode("wol") == "wo"

    assert "wo" in two_to_three_letters
    assert two_to_three_letters["wo"] == "wol"

def test_langlower():
    assert lang_to_langcode("WOLOF") == "wo"
    assert lang_to_langcode("nOrWeGiAn") == "nb"

    assert "soj" == langlower2lcode["soi"]
    assert "soj" == langlower2lcode["sohi"]
