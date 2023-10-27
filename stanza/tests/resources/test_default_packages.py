import pytest

import stanza

from stanza.resources import default_packages

def test_default_pretrains():
    """
    Test that all languages with a default treebank have a default pretrain or are specifically marked as not having a pretrain
    """
    for lang in default_packages.default_treebanks.keys():
        assert lang in default_packages.no_pretrain_languages or lang in default_packages.default_pretrains, "Lang %s does not have a default pretrain marked!" % lang

def test_no_pretrain_languages():
    """
    Test that no languages have no_default_pretrain marked despite having a pretrain
    """
    for lang in default_packages.no_pretrain_languages:
        assert lang not in default_packages.default_pretrains, "Lang %s is marked as no_pretrain but has a default pretrain!" % lang




    
