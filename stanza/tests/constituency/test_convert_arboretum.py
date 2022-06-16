"""
Test a couple different classes of trees to check the output of the Arboretum conversion

Note that the text has been removed
"""

import os
import tempfile

import pytest

from stanza.server import tsurgeon
from stanza.tests import TEST_WORKING_DIR
from stanza.utils.datasets.constituency import convert_arboretum

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]


PROJ_EXAMPLE="""
<s id="s2" ref="AACBPIGY" source="id=AACBPIGY" forest="1/1" text="A B C D E F G H.">
	<graph root="s2_500">
		<terminals>
			<t id="s2_1" word="A" lemma="A" pos="prop" morph="NOM" extra="PROP:A compound brand"/>
			<t id="s2_2" word="B" lemma="B" pos="v-fin" morph="PR AKT" extra="mv"/>
			<t id="s2_3" word="C" lemma="C" pos="pron-pers" morph="2S ACC" extra="--"/>
			<t id="s2_4" word="D" lemma="D" pos="adj" morph="UTR S IDF NOM" extra="F:u+afhængig"/>
			<t id="s2_5" word="E" lemma="E" pos="prp" morph="--" extra="--"/>
			<t id="s2_6" word="F" lemma="F" pos="art" morph="NEU S DEF" extra="--"/>
			<t id="s2_7" word="G" lemma="G" pos="adj" morph="nG S DEF NOM" extra="--"/>
			<t id="s2_8" word="H" lemma="H" pos="n" morph="NEU S IDF NOM" extra="N:lys+net"/>
			<t id="s2_9" word="." lemma="--" pos="pu" morph="--" extra="--"/>
		</terminals>

		<nonterminals>
			<nt id="s2_500" cat="s">
				<edge label="STA" idref="s2_501"/>
			</nt>
			<nt id="s2_501" cat="fcl">
				<edge label="S" idref="s2_1"/>
				<edge label="P" idref="s2_2"/>
				<edge label="Od" idref="s2_3"/>
				<edge label="Co" idref="s2_502"/>
				<edge label="PU" idref="s2_9"/>
			</nt>
			<nt id="s2_502" cat="adjp">
				<edge label="H" idref="s2_4"/>
				<edge label="DA" idref="s2_503"/>
			</nt>
			<nt id="s2_503" cat="pp">
				<edge label="H" idref="s2_5"/>
				<edge label="DP" idref="s2_504"/>
			</nt>
			<nt id="s2_504" cat="np">
				<edge label="DN" idref="s2_6"/>
				<edge label="DN" idref="s2_7"/>
				<edge label="H" idref="s2_8"/>
			</nt>
		</nonterminals>
	</graph>
</s>
"""

NOT_FIX_NONPROJ_EXAMPLE="""
<s id="s322" ref="EDGBITSZ" source="id=EDGBITSZ" forest="1/2" text="A B C D E, F G H I J.">
        <graph root="s322_500">
                <terminals>
                        <t id="s322_1" word="A" lemma="A" pos="prop" morph="NOM" extra="hum fem"/>
                        <t id="s322_2" word="B" lemma="B" pos="v-fin" morph="PR AKT" extra="mv"/>
                        <t id="s322_3" word="C" lemma="C" pos="pron-dem" morph="UTR S" extra="dem"/>
                        <t id="s322_4" word="D" lemma="D" pos="n" morph="UTR S IDF NOM" extra="--"/>
                        <t id="s322_5" word="E" lemma="E" pos="adv" morph="--" extra="--"/>
                        <t id="s322_6" word="," lemma="--" pos="pu" morph="--" extra="--"/>
                        <t id="s322_7" word="F" lemma="F" pos="pron-rel" morph="--" extra="rel"/>
                        <t id="s322_8" word="G" lemma="G" pos="prop" morph="NOM" extra="hum"/>
                        <t id="s322_9" word="H" lemma="H" pos="v-fin" morph="IMPF AKT" extra="mv"/>
                        <t id="s322_10" word="I" lemma="I" pos="prp" morph="--" extra="--"/>
                        <t id="s322_11" word="J" lemma="J" pos="n" morph="UTR S DEF NOM" extra="F:ur+premiere"/>
                        <t id="s322_12" word="." lemma="--" pos="pu" morph="--" extra="--"/>
                </terminals>

                <nonterminals>
                        <nt id="s322_500" cat="s">
                                <edge label="STA" idref="s322_501"/>
                        </nt>
                        <nt id="s322_501" cat="fcl">
                                <edge label="S" idref="s322_1"/>
                                <edge label="P" idref="s322_2"/>
                                <edge label="Od" idref="s322_502"/>
                                <edge label="Vpart" idref="s322_5"/>
                                <edge label="PU" idref="s322_6"/>
                                <edge label="PU" idref="s322_12"/>
                        </nt>
                        <nt id="s322_502" cat="np">
                                <edge label="DN" idref="s322_3"/>
                                <edge label="H" idref="s322_4"/>
                                <edge label="DN" idref="s322_503"/>
                        </nt>
                        <nt id="s322_503" cat="fcl">
                                <edge label="Od" idref="s322_7"/>
                                <edge label="S" idref="s322_8"/>
                                <edge label="P" idref="s322_9"/>
                                <edge label="Ao" idref="s322_504"/>
                        </nt>
                        <nt id="s322_504" cat="pp">
                                <edge label="H" idref="s322_10"/>
                                <edge label="DP" idref="s322_11"/>
                        </nt>
                </nonterminals>
        </graph>
</s>
"""


NONPROJ_EXAMPLE="""
<s id="s9" ref="AATCNKQZ" source="id=AATCNKQZ" forest="1/1" text="A B C D E F G H I.">
        <graph root="s9_500">
                <terminals>
                        <t id="s9_1" word="A" lemma="A" pos="adv" morph="--" extra="--"/>
                        <t id="s9_2" word="B" lemma="B" pos="adv" morph="--" extra="--"/>
                        <t id="s9_3" word="C" lemma="C" pos="v-fin" morph="IMPF AKT" extra="aux"/>
                        <t id="s9_4" word="D" lemma="D" pos="prop" morph="NOM" extra="hum"/>
                        <t id="s9_5" word="E" lemma="E" pos="adv" morph="--" extra="--"/>
                        <t id="s9_6" word="F" lemma="F" pos="v-pcp2" morph="PAS" extra="mv"/>
                        <t id="s9_7" word="G" lemma="G" pos="prp" morph="--" extra="--"/>
                        <t id="s9_8" word="H" lemma="H" pos="num" morph="--" extra="card"/>
                        <t id="s9_9" word="I" lemma="I" pos="n" morph="UTR P IDF NOM" extra="N:patrulje+vogn"/>
                        <t id="s9_10" word="." lemma="--" pos="pu" morph="--" extra="--"/>
                </terminals>

                <nonterminals>
                        <nt id="s9_500" cat="s">
                                <edge label="STA" idref="s9_501"/>
                        </nt>
                        <nt id="s9_501" cat="fcl">
                                <edge label="fA" idref="s9_502"/>
                                <edge label="P" idref="s9_503"/>
                                <edge label="S" idref="s9_4"/>
                                <edge label="fA" idref="s9_5"/>
                                <edge label="fA" idref="s9_504"/>
                                <edge label="PU" idref="s9_10"/>
                        </nt>
                        <nt id="s9_502" cat="advp">
                                <edge label="DA" idref="s9_1"/>
                                <edge label="H" idref="s9_2"/>
                        </nt>
                        <nt id="s9_503" cat="vp">
                                <edge label="Vaux" idref="s9_3"/>
                                <edge label="Vm" idref="s9_6"/>
                        </nt>
                        <nt id="s9_504" cat="pp">
                                <edge label="H" idref="s9_7"/>
                                <edge label="DP" idref="s9_505"/>
                        </nt>
                        <nt id="s9_505" cat="np">
                                <edge label="DN" idref="s9_8"/>
                                <edge label="H" idref="s9_9"/>
                        </nt>
                </nonterminals>
        </graph>
</s>
"""

def test_projective_example():
    """
    Test reading a basic tree, along with some further manipulations from the conversion program
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tempdir:
        test_name = os.path.join(tempdir, "proj.xml")
        with open(test_name, "w", encoding="utf-8") as fout:
            fout.write(PROJ_EXAMPLE)
        sentences = convert_arboretum.read_xml_file(test_name)
        assert len(sentences) == 1

    tree, words = convert_arboretum.process_tree(sentences[0])
    expected_tree = "(s (fcl (prop s2_1) (v-fin s2_2) (pron-pers s2_3) (adjp (adj s2_4) (pp (prp s2_5) (np (art s2_6) (adj s2_7) (n s2_8)))) (pu s2_9)))"
    assert str(tree) == expected_tree
    assert [w.word for w in words.values()] == ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '.']
    assert not convert_arboretum.word_sequence_missing_words(tree)
    with tsurgeon.Tsurgeon() as tsurgeon_processor:
        assert tree == convert_arboretum.check_words(tree, tsurgeon_processor)

    # check that the words can be replaced as expected
    replaced_tree = convert_arboretum.replace_words(tree, words)
    expected_tree = "(s (fcl (prop A) (v-fin B) (pron-pers C) (adjp (adj D) (pp (prp E) (np (art F) (adj G) (n H)))) (pu .)))"
    assert str(replaced_tree) == expected_tree
    assert convert_arboretum.split_underscores(replaced_tree) == replaced_tree

    # fake a word which should be split
    words['s2_1'] = words['s2_1']._replace(word='foo_bar')
    replaced_tree = convert_arboretum.replace_words(tree, words)
    expected_tree = "(s (fcl (prop foo_bar) (v-fin B) (pron-pers C) (adjp (adj D) (pp (prp E) (np (art F) (adj G) (n H)))) (pu .)))"
    assert str(replaced_tree) == expected_tree
    expected_tree = "(s (fcl (np (prop foo) (prop bar)) (v-fin B) (pron-pers C) (adjp (adj D) (pp (prp E) (np (art F) (adj G) (n H)))) (pu .)))"
    assert str(convert_arboretum.split_underscores(replaced_tree)) == expected_tree


def test_not_fix_example():
    """
    Test that a non-projective tree which we don't have a heuristic for quietly fails
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tempdir:
        test_name = os.path.join(tempdir, "nofix.xml")
        with open(test_name, "w", encoding="utf-8") as fout:
            fout.write(NOT_FIX_NONPROJ_EXAMPLE)
        sentences = convert_arboretum.read_xml_file(test_name)
        assert len(sentences) == 1

    tree, words = convert_arboretum.process_tree(sentences[0])
    assert not convert_arboretum.word_sequence_missing_words(tree)
    with tsurgeon.Tsurgeon() as tsurgeon_processor:
        assert convert_arboretum.check_words(tree, tsurgeon_processor) is None


def test_fix_proj_example():
    """
    Test that a non-projective tree can be rearranged as expected

    Note that there are several other classes of non-proj tree we could test as well...
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tempdir:
        test_name = os.path.join(tempdir, "fix.xml")
        with open(test_name, "w", encoding="utf-8") as fout:
            fout.write(NONPROJ_EXAMPLE)
        sentences = convert_arboretum.read_xml_file(test_name)
        assert len(sentences) == 1

    tree, words = convert_arboretum.process_tree(sentences[0])
    assert not convert_arboretum.word_sequence_missing_words(tree)
    # the 4 and 5 are moved inside the 3-6 node
    expected_orig = "(s (fcl (advp (adv s9_1) (adv s9_2)) (vp (v-fin s9_3) (v-pcp2 s9_6)) (prop s9_4) (adv s9_5) (pp (prp s9_7) (np (num s9_8) (n s9_9))) (pu s9_10)))"
    expected_proj = "(s (fcl (advp (adv s9_1) (adv s9_2)) (vp (v-fin s9_3) (prop s9_4) (adv s9_5) (v-pcp2 s9_6)) (pp (prp s9_7) (np (num s9_8) (n s9_9))) (pu s9_10)))"
    assert str(tree) == expected_orig
    with tsurgeon.Tsurgeon() as tsurgeon_processor:
        assert str(convert_arboretum.check_words(tree, tsurgeon_processor)) == expected_proj

