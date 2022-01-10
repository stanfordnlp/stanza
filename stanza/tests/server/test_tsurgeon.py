"""
Test the semgrex interface
"""

import pytest
import stanza
from stanza.models.constituency import tree_reader
from stanza.server.tsurgeon import process_trees, Tsurgeon

from stanza.tests import *

pytestmark = [pytest.mark.travis, pytest.mark.client]



def test_simple():
    text="( (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"
    trees = tree_reader.read_trees(text)

    tregex = "WP=wp"
    tsurgeon = "relabel wp WWWPPP"
    result = process_trees(trees, (tregex, tsurgeon))
    assert len(result) == 1
    assert str(result[0]) == "(ROOT (SBARQ (WHNP (WWWPPP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"

def test_context():
    """
    Processing the same thing twice should work twice...
    """
    with Tsurgeon() as processor:
        text="( (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"
        trees = tree_reader.read_trees(text)

        tregex = "WP=wp"
        tsurgeon = "relabel wp WWWPPP"
        result = processor.process(trees, (tregex, tsurgeon))
        assert len(result) == 1
        assert str(result[0]) == "(ROOT (SBARQ (WHNP (WWWPPP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"

        result = processor.process(trees, (tregex, tsurgeon))
        assert len(result) == 1
        assert str(result[0]) == "(ROOT (SBARQ (WHNP (WWWPPP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"


def test_arboretum():
    """
    Test a couple expressions used when processing the Arboretum treebank

    That particular treebank was the original inspiration for adding the Tsurgeon interface
    """
    with Tsurgeon() as processor:
        text = "(s (par (fcl (n s1_1) (vp (v-fin s1_2) (v-pcp2 s1_4)) (adv s1_3) (np (pron-poss s1_5) (n s1_6) (pp (prp s1_7) (n s1_8)))) (pu s1_9) (conj-c s1_10) (fcl (adv s1_11) (v-fin s1_12) (np (prop s1_13) (pp (prp s1_14) (prop s1_15))) (np (art s1_16) (adjp (adv s1_17) (adj s1_18)) (n s1_19) (pp (prp s1_20) (np (pron-poss s1_21) (adj s1_22) (n s1_23) (prop s1_24))))) (pu s1_25)))"
        expected = "(s (par (fcl (n s1_1) (vp (v-fin s1_2) (adv s1_3) (v-pcp2 s1_4)) (np (pron-poss s1_5) (n s1_6) (pp (prp s1_7) (n s1_8)))) (pu s1_9) (conj-c s1_10) (fcl (adv s1_11) (v-fin s1_12) (np (prop s1_13) (pp (prp s1_14) (prop s1_15))) (np (art s1_16) (adjp (adv s1_17) (adj s1_18)) (n s1_19) (pp (prp s1_20) (np (pron-poss s1_21) (adj s1_22) (n s1_23) (prop s1_24))))) (pu s1_25)))"
        trees = tree_reader.read_trees(text)

        tregex = "s1_4 > (__=home > (__=parent > __=grandparent)) . (s1_3 > (__=move > =grandparent))"
        tsurgeon = "move move $+ home"
        result = processor.process(trees, (tregex, tsurgeon))
        assert len(result) == 1
        assert str(result[0]) == expected


        text = "(s (par (fcl (n s1_1) (vp (v-fin s1_2) (v-pcp2 s1_4)) (adv s1_3) (np (pron-poss s1_5) (n s1_6) (pp (prp s1_7) (n s1_8)))) (pu s1_9) (conj-c s1_10) (fcl (adv s1_11) (v-fin s1_12) (np (prop s1_13) (pp (prp s1_14) (prop s1_15))) (np (art s1_16) (adjp (adv s1_17) (adj s1_18)) (n s1_19) (pp (prp s1_20) (np (pron-poss s1_21) (adj s1_22) (n s1_23) (prop s1_24))))) (pu s1_25)))"
        expected = "(s (par (fcl (n s1_1) (vp (v-fin s1_2) (adv s1_3) (v-pcp2 s1_4)) (np (pron-poss s1_5) (n s1_6) (pp (prp s1_7) (n s1_8)))) (pu s1_9) (conj-c s1_10) (fcl (adv s1_11) (v-fin s1_12) (np (prop s1_13) (pp (prp s1_14) (prop s1_15))) (np (art s1_16) (adjp (adv s1_17) (adj s1_18)) (n s1_19) (pp (prp s1_20) (np (pron-poss s1_21) (adj s1_22) (n s1_23) (prop s1_24))))) (pu s1_25)))"
        trees = tree_reader.read_trees(text)

        tregex = "s1_4 > (__=home > (__=parent $+ (__=move <<, s1_3 <<- s1_3)))"
        tsurgeon = "move move $+ home"
        result = processor.process(trees, (tregex, tsurgeon))
        assert len(result) == 1
        assert str(result[0]) == expected
