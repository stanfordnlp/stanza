"""
Basic testing of the English pipeline
"""

import pytest
import stanfordnlp
from stanfordnlp.utils.conll import CoNLL

from tests import *


# data for testing
EN_DOC = "Barack Obama was born in Hawaii.  He was elected president in 2008.  Obama attended Harvard."

EN_DOC_TOKENS_GOLD = """
<Token id=1;words=[<Word id=1;text=Barack;lemma=Barack;upos=PROPN;xpos=NNP;feats=Number=Sing;head=4;deprel=nsubj:pass>]>
<Token id=2;words=[<Word id=2;text=Obama;lemma=Obama;upos=PROPN;xpos=NNP;feats=Number=Sing;head=1;deprel=flat>]>
<Token id=3;words=[<Word id=3;text=was;lemma=be;upos=AUX;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin;head=4;deprel=aux:pass>]>
<Token id=4;words=[<Word id=4;text=born;lemma=bear;upos=VERB;xpos=VBN;feats=Tense=Past|VerbForm=Part|Voice=Pass;head=0;deprel=root>]>
<Token id=5;words=[<Word id=5;text=in;lemma=in;upos=ADP;xpos=IN;head=6;deprel=case>]>
<Token id=6;words=[<Word id=6;text=Hawaii;lemma=Hawaii;upos=PROPN;xpos=NNP;feats=Number=Sing;head=4;deprel=obl>]>
<Token id=7;words=[<Word id=7;text=.;lemma=.;upos=PUNCT;xpos=.;head=4;deprel=punct>]>

<Token id=1;words=[<Word id=1;text=He;lemma=he;upos=PRON;xpos=PRP;feats=Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs;head=3;deprel=nsubj:pass>]>
<Token id=2;words=[<Word id=2;text=was;lemma=be;upos=AUX;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin;head=3;deprel=aux:pass>]>
<Token id=3;words=[<Word id=3;text=elected;lemma=elect;upos=VERB;xpos=VBN;feats=Tense=Past|VerbForm=Part|Voice=Pass;head=0;deprel=root>]>
<Token id=4;words=[<Word id=4;text=president;lemma=president;upos=PROPN;xpos=NNP;feats=Number=Sing;head=3;deprel=xcomp>]>
<Token id=5;words=[<Word id=5;text=in;lemma=in;upos=ADP;xpos=IN;head=6;deprel=case>]>
<Token id=6;words=[<Word id=6;text=2008;lemma=2008;upos=NUM;xpos=CD;feats=NumType=Card;head=3;deprel=obl>]>
<Token id=7;words=[<Word id=7;text=.;lemma=.;upos=PUNCT;xpos=.;head=3;deprel=punct>]>

<Token id=1;words=[<Word id=1;text=Obama;lemma=Obama;upos=PROPN;xpos=NNP;feats=Number=Sing;head=2;deprel=nsubj>]>
<Token id=2;words=[<Word id=2;text=attended;lemma=attend;upos=VERB;xpos=VBD;feats=Mood=Ind|Tense=Past|VerbForm=Fin;head=0;deprel=root>]>
<Token id=3;words=[<Word id=3;text=Harvard;lemma=Harvard;upos=PROPN;xpos=NNP;feats=Number=Sing;head=2;deprel=obj>]>
<Token id=4;words=[<Word id=4;text=.;lemma=.;upos=PUNCT;xpos=.;head=2;deprel=punct>]>
""".strip()

EN_DOC_WORDS_GOLD = """
<Word id=1;text=Barack;lemma=Barack;upos=PROPN;xpos=NNP;feats=Number=Sing;head=4;deprel=nsubj:pass>
<Word id=2;text=Obama;lemma=Obama;upos=PROPN;xpos=NNP;feats=Number=Sing;head=1;deprel=flat>
<Word id=3;text=was;lemma=be;upos=AUX;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin;head=4;deprel=aux:pass>
<Word id=4;text=born;lemma=bear;upos=VERB;xpos=VBN;feats=Tense=Past|VerbForm=Part|Voice=Pass;head=0;deprel=root>
<Word id=5;text=in;lemma=in;upos=ADP;xpos=IN;head=6;deprel=case>
<Word id=6;text=Hawaii;lemma=Hawaii;upos=PROPN;xpos=NNP;feats=Number=Sing;head=4;deprel=obl>
<Word id=7;text=.;lemma=.;upos=PUNCT;xpos=.;head=4;deprel=punct>

<Word id=1;text=He;lemma=he;upos=PRON;xpos=PRP;feats=Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs;head=3;deprel=nsubj:pass>
<Word id=2;text=was;lemma=be;upos=AUX;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin;head=3;deprel=aux:pass>
<Word id=3;text=elected;lemma=elect;upos=VERB;xpos=VBN;feats=Tense=Past|VerbForm=Part|Voice=Pass;head=0;deprel=root>
<Word id=4;text=president;lemma=president;upos=PROPN;xpos=NNP;feats=Number=Sing;head=3;deprel=xcomp>
<Word id=5;text=in;lemma=in;upos=ADP;xpos=IN;head=6;deprel=case>
<Word id=6;text=2008;lemma=2008;upos=NUM;xpos=CD;feats=NumType=Card;head=3;deprel=obl>
<Word id=7;text=.;lemma=.;upos=PUNCT;xpos=.;head=3;deprel=punct>

<Word id=1;text=Obama;lemma=Obama;upos=PROPN;xpos=NNP;feats=Number=Sing;head=2;deprel=nsubj>
<Word id=2;text=attended;lemma=attend;upos=VERB;xpos=VBD;feats=Mood=Ind|Tense=Past|VerbForm=Fin;head=0;deprel=root>
<Word id=3;text=Harvard;lemma=Harvard;upos=PROPN;xpos=NNP;feats=Number=Sing;head=2;deprel=obj>
<Word id=4;text=.;lemma=.;upos=PUNCT;xpos=.;head=2;deprel=punct>
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
('president', '3', 'xcomp')
('in', '6', 'case')
('2008', '3', 'obl')
('.', '3', 'punct')

('Obama', '2', 'nsubj')
('attended', '0', 'root')
('Harvard', '2', 'obj')
('.', '2', 'punct')
""".strip()

EN_DOC_CONLLU_GOLD = """
1	Barack	Barack	PROPN	NNP	Number=Sing	4	nsubj:pass	_	beginCharOffset=0|endCharOffset=6
2	Obama	Obama	PROPN	NNP	Number=Sing	1	flat	_	beginCharOffset=7|endCharOffset=12
3	was	be	AUX	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	4	aux:pass	_	beginCharOffset=13|endCharOffset=16
4	born	bear	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	0	root	_	beginCharOffset=17|endCharOffset=21
5	in	in	ADP	IN	_	6	case	_	beginCharOffset=22|endCharOffset=24
6	Hawaii	Hawaii	PROPN	NNP	Number=Sing	4	obl	_	beginCharOffset=25|endCharOffset=31
7	.	.	PUNCT	.	_	4	punct	_	beginCharOffset=31|endCharOffset=32

1	He	he	PRON	PRP	Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs	3	nsubj:pass	_	beginCharOffset=34|endCharOffset=36
2	was	be	AUX	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	3	aux:pass	_	beginCharOffset=37|endCharOffset=40
3	elected	elect	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	0	root	_	beginCharOffset=41|endCharOffset=48
4	president	president	PROPN	NNP	Number=Sing	3	xcomp	_	beginCharOffset=49|endCharOffset=58
5	in	in	ADP	IN	_	6	case	_	beginCharOffset=59|endCharOffset=61
6	2008	2008	NUM	CD	NumType=Card	3	obl	_	beginCharOffset=62|endCharOffset=66
7	.	.	PUNCT	.	_	3	punct	_	beginCharOffset=66|endCharOffset=67

1	Obama	Obama	PROPN	NNP	Number=Sing	2	nsubj	_	beginCharOffset=69|endCharOffset=74
2	attended	attend	VERB	VBD	Mood=Ind|Tense=Past|VerbForm=Fin	0	root	_	beginCharOffset=75|endCharOffset=83
3	Harvard	Harvard	PROPN	NNP	Number=Sing	2	obj	_	beginCharOffset=84|endCharOffset=91
4	.	.	PUNCT	.	_	2	punct	_	beginCharOffset=91|endCharOffset=92

""".lstrip()


@pytest.fixture(scope="module")
def processed_doc():
    """ Document created by running full English pipeline on a few sentences """
    nlp = stanfordnlp.Pipeline(models_dir=TEST_MODELS_DIR)
    return nlp(EN_DOC)


def test_text(processed_doc):
    assert processed_doc.text == EN_DOC

    
def test_conllu(processed_doc):
    assert CoNLL.conll_as_string(CoNLL.convert_dict(processed_doc.to_dict())) == EN_DOC_CONLLU_GOLD


def test_tokens(processed_doc):
    assert "\n\n".join([sent.tokens_string() for sent in processed_doc.sentences]) == EN_DOC_TOKENS_GOLD


def test_words(processed_doc):
    assert "\n\n".join([sent.words_string() for sent in processed_doc.sentences]) == EN_DOC_WORDS_GOLD


def test_dependency_parse(processed_doc):
    assert "\n\n".join([sent.dependencies_string() for sent in processed_doc.sentences]) == \
           EN_DOC_DEPENDENCY_PARSES_GOLD
