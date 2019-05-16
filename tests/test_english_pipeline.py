"""
Basic testing of the English pipeline
"""

import pytest
import stanfordnlp

from tests import *


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

EN_DOC_WORDS_GOLD = """
<Word index=1;text=Barack;lemma=Barack;upos=PROPN;xpos=NNP;feats=Number=Sing;governor=4;dependency_relation=nsubj:pass>
<Word index=2;text=Obama;lemma=Obama;upos=PROPN;xpos=NNP;feats=Number=Sing;governor=1;dependency_relation=flat>
<Word index=3;text=was;lemma=be;upos=AUX;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin;governor=4;dependency_relation=aux:pass>
<Word index=4;text=born;lemma=bear;upos=VERB;xpos=VBN;feats=Tense=Past|VerbForm=Part|Voice=Pass;governor=0;dependency_relation=root>
<Word index=5;text=in;lemma=in;upos=ADP;xpos=IN;feats=_;governor=6;dependency_relation=case>
<Word index=6;text=Hawaii;lemma=Hawaii;upos=PROPN;xpos=NNP;feats=Number=Sing;governor=4;dependency_relation=obl>
<Word index=7;text=.;lemma=.;upos=PUNCT;xpos=.;feats=_;governor=4;dependency_relation=punct>

<Word index=1;text=He;lemma=he;upos=PRON;xpos=PRP;feats=Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs;governor=3;dependency_relation=nsubj:pass>
<Word index=2;text=was;lemma=be;upos=AUX;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin;governor=3;dependency_relation=aux:pass>
<Word index=3;text=elected;lemma=elect;upos=VERB;xpos=VBN;feats=Tense=Past|VerbForm=Part|Voice=Pass;governor=0;dependency_relation=root>
<Word index=4;text=president;lemma=president;upos=NOUN;xpos=NN;feats=Number=Sing;governor=3;dependency_relation=obj>
<Word index=5;text=in;lemma=in;upos=ADP;xpos=IN;feats=_;governor=6;dependency_relation=case>
<Word index=6;text=2008;lemma=2008;upos=NUM;xpos=CD;feats=NumType=Card;governor=3;dependency_relation=obl>
<Word index=7;text=.;lemma=.;upos=PUNCT;xpos=.;feats=_;governor=3;dependency_relation=punct>

<Word index=1;text=Obama;lemma=Obama;upos=PROPN;xpos=NNP;feats=Number=Sing;governor=2;dependency_relation=nsubj>
<Word index=2;text=attended;lemma=attend;upos=VERB;xpos=VBD;feats=Mood=Ind|Tense=Past|VerbForm=Fin;governor=0;dependency_relation=root>
<Word index=3;text=Harvard;lemma=Harvard;upos=PROPN;xpos=NNP;feats=Number=Sing;governor=2;dependency_relation=obj>
<Word index=4;text=.;lemma=.;upos=PUNCT;xpos=.;feats=_;governor=2;dependency_relation=punct>
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

EN_DOC_CONLLU_GOLD = """
1	Barack	Barack	PROPN	NNP	Number=Sing	4	nsubj:pass	_	_
2	Obama	Obama	PROPN	NNP	Number=Sing	1	flat	_	_
3	was	be	AUX	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	4	aux:pass	_	_
4	born	bear	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	0	root	_	_
5	in	in	ADP	IN	_	6	case	_	_
6	Hawaii	Hawaii	PROPN	NNP	Number=Sing	4	obl	_	_
7	.	.	PUNCT	.	_	4	punct	_	_

1	He	he	PRON	PRP	Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs	3	nsubj:pass	_	_
2	was	be	AUX	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	3	aux:pass	_	_
3	elected	elect	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	0	root	_	_
4	president	president	NOUN	NN	Number=Sing	3	obj	_	_
5	in	in	ADP	IN	_	6	case	_	_
6	2008	2008	NUM	CD	NumType=Card	3	obl	_	_
7	.	.	PUNCT	.	_	3	punct	_	_

1	Obama	Obama	PROPN	NNP	Number=Sing	2	nsubj	_	_
2	attended	attend	VERB	VBD	Mood=Ind|Tense=Past|VerbForm=Fin	0	root	_	_
3	Harvard	Harvard	PROPN	NNP	Number=Sing	2	obj	_	_
4	.	.	PUNCT	.	_	2	punct	_	_

""".lstrip()


@pytest.fixture(scope="module")
def processed_doc():
    """ Document created by running full English pipeline on a few sentences """
    nlp = stanfordnlp.Pipeline(models_dir=TEST_MODELS_DIR)
    return nlp(EN_DOC)


def test_text(processed_doc):
    assert processed_doc.text == EN_DOC

    
def test_conllu(processed_doc):
    assert processed_doc.conll_file.conll_as_string() == EN_DOC_CONLLU_GOLD


def test_tokens(processed_doc):
    assert "\n\n".join([sent.tokens_string() for sent in processed_doc.sentences]) == EN_DOC_TOKENS_GOLD


def test_words(processed_doc):
    assert "\n\n".join([sent.words_string() for sent in processed_doc.sentences]) == EN_DOC_WORDS_GOLD


def test_dependency_parse(processed_doc):
    assert "\n\n".join([sent.dependencies_string() for sent in processed_doc.sentences]) == \
           EN_DOC_DEPENDENCY_PARSES_GOLD
