"""
Basic testing of the English pipeline
"""

import pytest
import stanfordnlp

from tests import *


def setup_module(module):
    """Set up resources for all tests in this module"""
    safe_rm(EN_MODELS_DIR)
    stanfordnlp.download('en', resource_dir=TEST_WORKING_DIR, force=True)


def teardown_module(module):
    """Clean up resources after tests complete"""
    safe_rm(EN_MODELS_DIR)


# data for testing
EN_DOC = "Barack Obama was born in Hawaii.  He was elected president in 2008.  Obama attended Harvard."

EN_DOC_TOKENS_GOLD = """
<Token index=1;words=[<Word index=1;text=Barack;lemma=Barack;upos=PROPN;xpos=NNP;feats=Number=Sing;governor=4;dependency_relation=nsubj:pass>]>
<Token index=2;words=[<Word index=2;text=Obama;lemma=Obama;upos=PROPN;xpos=NNP;feats=Number=Sing;governor=1;dependency_relation=flat>]>
<Token index=3;words=[<Word index=3;text=was;lemma=be;upos=AUX;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin;governor=4;dependency_relation=aux:pass>]>
<Token index=4;words=[<Word index=4;text=born;lemma=bear;upos=VERB;xpos=VBN;feats=Tense=Past|VerbForm=Part|Voice=Pass;governor=0;dependency_relation=root>]>
<Token index=5;words=[<Word index=5;text=in;lemma=in;upos=ADP;xpos=IN;feats=_;governor=6;dependency_relation=case>]>
<Token index=6;words=[<Word index=6;text=Hawaii;lemma=Hawaii;upos=PROPN;xpos=NNP;feats=Number=Sing;governor=4;dependency_relation=obl>]>
<Token index=7;words=[<Word index=7;text=.;lemma=.;upos=PUNCT;xpos=.;feats=_;governor=4;dependency_relation=punct>]>

<Token index=1;words=[<Word index=1;text=He;lemma=he;upos=PRON;xpos=PRP;feats=Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs;governor=3;dependency_relation=nsubj:pass>]>
<Token index=2;words=[<Word index=2;text=was;lemma=be;upos=AUX;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin;governor=3;dependency_relation=aux:pass>]>
<Token index=3;words=[<Word index=3;text=elected;lemma=elect;upos=VERB;xpos=VBN;feats=Tense=Past|VerbForm=Part|Voice=Pass;governor=0;dependency_relation=root>]>
<Token index=4;words=[<Word index=4;text=president;lemma=president;upos=NOUN;xpos=NN;feats=Number=Sing;governor=3;dependency_relation=obj>]>
<Token index=5;words=[<Word index=5;text=in;lemma=in;upos=ADP;xpos=IN;feats=_;governor=6;dependency_relation=case>]>
<Token index=6;words=[<Word index=6;text=2008;lemma=2008;upos=NUM;xpos=CD;feats=NumType=Card;governor=3;dependency_relation=obl>]>
<Token index=7;words=[<Word index=7;text=.;lemma=.;upos=PUNCT;xpos=.;feats=_;governor=3;dependency_relation=punct>]>

<Token index=1;words=[<Word index=1;text=Obama;lemma=Obama;upos=PROPN;xpos=NNP;feats=Number=Sing;governor=2;dependency_relation=nsubj>]>
<Token index=2;words=[<Word index=2;text=attended;lemma=attend;upos=VERB;xpos=VBD;feats=Mood=Ind|Tense=Past|VerbForm=Fin;governor=0;dependency_relation=root>]>
<Token index=3;words=[<Word index=3;text=Harvard;lemma=Harvard;upos=PROPN;xpos=NNP;feats=Number=Sing;governor=2;dependency_relation=obj>]>
<Token index=4;words=[<Word index=4;text=.;lemma=.;upos=PUNCT;xpos=.;feats=_;governor=2;dependency_relation=punct>]>
""".strip()

EN_DOC_DEPENDENCY_PARSES_GOLD = """
('Barack', '4', 'nsubj:pass')
('Obama', '1', 'flat')
('was', '4', 'aux:pass')
('born', '0', 'root')
('in', '6', 'case')
('Hawaii', '4', 'obl')
('.', '4', 'punct')

('He', '3', 'nsubj:pass')
('was', '3', 'aux:pass')
('elected', '0', 'root')
('president', '3', 'obj')
('in', '6', 'case')
('2008', '3', 'obl')
('.', '3', 'punct')

('Obama', '2', 'nsubj')
('attended', '0', 'root')
('Harvard', '2', 'obj')
('.', '2', 'punct')
""".strip()


@pytest.fixture(scope="module")
def processed_doc():
    """ Document created by running full English pipeline on a few sentences """
    nlp = stanfordnlp.Pipeline(models_dir=TEST_WORKING_DIR)
    return nlp(EN_DOC)


def test_tokens(processed_doc):
    assert "\n\n".join([sent.tokens_string() for sent in processed_doc.sentences]) == EN_DOC_TOKENS_GOLD


def test_dependency_parse(processed_doc):
    assert "\n\n".join([sent.dependencies_string() for sent in processed_doc.sentences]) == \
           EN_DOC_DEPENDENCY_PARSES_GOLD
