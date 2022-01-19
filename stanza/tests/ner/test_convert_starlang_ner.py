"""
Test a couple different classes of trees to check the output of the Starlang conversion for NER
"""

import os
import tempfile

import pytest

from stanza.utils.datasets.ner import convert_starlang_ner

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

TREE="( (S (NP (NP {morphologicalAnalysis=bayan+NOUN+A3SG+PNON+NOM}{metaMorphemes=bayan}{turkish=Bayan}{english=Ms.}{semantics=TUR10-0396530}{namedEntity=PERSON}{propBank=ARG0$TUR10-0148580}{englishSemantics=ENG31-06352895-n}) (NP {morphologicalAnalysis=haag+NOUN+PROP+A3SG+PNON+NOM}{metaMorphemes=haag}{turkish=Haag}{english=Haag}{semantics=TUR10-0000000}{namedEntity=PERSON}{propBank=ARG0$TUR10-0148580}))  (VP (NP {morphologicalAnalysis=elianti+NOUN+PROP+A3SG+PNON+NOM}{metaMorphemes=elianti}{turkish=Elianti}{english=Elianti}{semantics=TUR10-0000000}{namedEntity=NONE}{propBank=ARG1$TUR10-0148580}) (VP {morphologicalAnalysis=çal+VERB+POS+AOR+A3SG}{metaMorphemes=çal+Ar}{turkish=çalar}{english=plays}{semantics=TUR10-0148580}{namedEntity=NONE}{propBank=PREDICATE$TUR10-0148580}{englishSemantics=ENG31-01730049-v}))  (. {morphologicalAnalysis=.+PUNC}{metaMorphemes=.}{metaMorphemesMoved=.}{turkish=.}{english=.}{semantics=TUR10-1081860}{namedEntity=NONE}{propBank=NONE}))  )"

def test_read_tree():
    """
    Test a basic tree read
    """
    sentence = convert_starlang_ner.read_tree(TREE)
    expected = [('Bayan', 'PERSON'), ('Haag', 'PERSON'), ('Elianti', 'O'), ('çalar', 'O'), ('.', 'O')]
    assert sentence == expected

