"""
Test a couple different classes of trees to check the output of the Starlang conversion
"""

import os
import tempfile

import pytest

from stanza.utils.datasets.constituency import convert_starlang

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

TREE="( (S (NP (NP {morphologicalAnalysis=bayan+NOUN+A3SG+PNON+NOM}{metaMorphemes=bayan}{turkish=Bayan}{english=Ms.}{semantics=TUR10-0396530}{namedEntity=PERSON}{propBank=ARG0$TUR10-0148580}{englishSemantics=ENG31-06352895-n}) (NP {morphologicalAnalysis=haag+NOUN+PROP+A3SG+PNON+NOM}{metaMorphemes=haag}{turkish=Haag}{english=Haag}{semantics=TUR10-0000000}{namedEntity=PERSON}{propBank=ARG0$TUR10-0148580}))  (VP (NP {morphologicalAnalysis=elianti+NOUN+PROP+A3SG+PNON+NOM}{metaMorphemes=elianti}{turkish=Elianti}{english=Elianti}{semantics=TUR10-0000000}{namedEntity=NONE}{propBank=ARG1$TUR10-0148580}) (VP {morphologicalAnalysis=çal+VERB+POS+AOR+A3SG}{metaMorphemes=çal+Ar}{turkish=çalar}{english=plays}{semantics=TUR10-0148580}{namedEntity=NONE}{propBank=PREDICATE$TUR10-0148580}{englishSemantics=ENG31-01730049-v}))  (. {morphologicalAnalysis=.+PUNC}{metaMorphemes=.}{metaMorphemesMoved=.}{turkish=.}{english=.}{semantics=TUR10-1081860}{namedEntity=NONE}{propBank=NONE}))  )"

def test_read_tree():
    """
    Test a basic tree read
    """
    tree = convert_starlang.read_tree(TREE)
    assert "(ROOT (S (NP (NP Bayan) (NP Haag)) (VP (NP Elianti) (VP çalar)) (. .)))" == str(tree)

def test_missing_word():
    """
    Test that an error is thrown if the word is missing
    """
    tree_text = TREE.replace("turkish=", "foo=")
    with pytest.raises(ValueError):
        tree = convert_starlang.read_tree(tree_text)

def test_bad_label():
    """
    Test that an unexpected label results in an error
    """
    tree_text = TREE.replace("(S", "(s")
    with pytest.raises(ValueError):
        tree = convert_starlang.read_tree(tree_text)
