"""
Basic testing of the English pipeline
"""

import pytest
import stanza
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document

from stanza.tests import *

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

# data for testing
EN_DOC = "Barack Obama was born in Hawaii.  He was elected president in 2008.  Obama attended Harvard."

EN_DOCS = ["Barack Obama was born in Hawaii.", "He was elected president in 2008.", "Obama attended Harvard."]

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
<Token id=4;words=[<Word id=4;text=president;lemma=president;upos=NOUN;xpos=NN;feats=Number=Sing;head=3;deprel=xcomp>]>
<Token id=5;words=[<Word id=5;text=in;lemma=in;upos=ADP;xpos=IN;head=6;deprel=case>]>
<Token id=6;words=[<Word id=6;text=2008;lemma=2008;upos=NUM;xpos=CD;feats=NumForm=Digit|NumType=Card;head=3;deprel=obl>]>
<Token id=7;words=[<Word id=7;text=.;lemma=.;upos=PUNCT;xpos=.;head=3;deprel=punct>]>

<Token id=1;words=[<Word id=1;text=Obama;lemma=Obama;upos=PROPN;xpos=NNP;feats=Number=Sing;head=2;deprel=nsubj>]>
<Token id=2;words=[<Word id=2;text=attended;lemma=attend;upos=VERB;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin;head=0;deprel=root>]>
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
<Word id=4;text=president;lemma=president;upos=NOUN;xpos=NN;feats=Number=Sing;head=3;deprel=xcomp>
<Word id=5;text=in;lemma=in;upos=ADP;xpos=IN;head=6;deprel=case>
<Word id=6;text=2008;lemma=2008;upos=NUM;xpos=CD;feats=NumForm=Digit|NumType=Card;head=3;deprel=obl>
<Word id=7;text=.;lemma=.;upos=PUNCT;xpos=.;head=3;deprel=punct>

<Word id=1;text=Obama;lemma=Obama;upos=PROPN;xpos=NNP;feats=Number=Sing;head=2;deprel=nsubj>
<Word id=2;text=attended;lemma=attend;upos=VERB;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin;head=0;deprel=root>
<Word id=3;text=Harvard;lemma=Harvard;upos=PROPN;xpos=NNP;feats=Number=Sing;head=2;deprel=obj>
<Word id=4;text=.;lemma=.;upos=PUNCT;xpos=.;head=2;deprel=punct>
""".strip()

EN_DOC_DEPENDENCY_PARSES_GOLD = """
('Barack', 4, 'nsubj:pass')
('Obama', 1, 'flat')
('was', 4, 'aux:pass')
('born', 0, 'root')
('in', 6, 'case')
('Hawaii', 4, 'obl')
('.', 4, 'punct')

('He', 3, 'nsubj:pass')
('was', 3, 'aux:pass')
('elected', 0, 'root')
('president', 3, 'xcomp')
('in', 6, 'case')
('2008', 3, 'obl')
('.', 3, 'punct')

('Obama', 2, 'nsubj')
('attended', 0, 'root')
('Harvard', 2, 'obj')
('.', 2, 'punct')
""".strip()

EN_DOC_CONLLU_GOLD = """
# text = Barack Obama was born in Hawaii.
# sent_id = 0
1	Barack	Barack	PROPN	NNP	Number=Sing	4	nsubj:pass	_	start_char=0|end_char=6|ner=B-PERSON
2	Obama	Obama	PROPN	NNP	Number=Sing	1	flat	_	start_char=7|end_char=12|ner=E-PERSON
3	was	be	AUX	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	4	aux:pass	_	start_char=13|end_char=16|ner=O
4	born	bear	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	0	root	_	start_char=17|end_char=21|ner=O
5	in	in	ADP	IN	_	6	case	_	start_char=22|end_char=24|ner=O
6	Hawaii	Hawaii	PROPN	NNP	Number=Sing	4	obl	_	start_char=25|end_char=31|ner=S-GPE
7	.	.	PUNCT	.	_	4	punct	_	start_char=31|end_char=32|ner=O

# text = He was elected president in 2008.
# sent_id = 1
1	He	he	PRON	PRP	Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs	3	nsubj:pass	_	start_char=34|end_char=36|ner=O
2	was	be	AUX	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	3	aux:pass	_	start_char=37|end_char=40|ner=O
3	elected	elect	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	0	root	_	start_char=41|end_char=48|ner=O
4	president	president	NOUN	NN	Number=Sing	3	xcomp	_	start_char=49|end_char=58|ner=O
5	in	in	ADP	IN	_	6	case	_	start_char=59|end_char=61|ner=O
6	2008	2008	NUM	CD	NumForm=Digit|NumType=Card	3	obl	_	start_char=62|end_char=66|ner=S-DATE
7	.	.	PUNCT	.	_	3	punct	_	start_char=66|end_char=67|ner=O

# text = Obama attended Harvard.
# sent_id = 2
1	Obama	Obama	PROPN	NNP	Number=Sing	2	nsubj	_	start_char=69|end_char=74|ner=S-PERSON
2	attended	attend	VERB	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	0	root	_	start_char=75|end_char=83|ner=O
3	Harvard	Harvard	PROPN	NNP	Number=Sing	2	obj	_	start_char=84|end_char=91|ner=S-ORG
4	.	.	PUNCT	.	_	2	punct	_	start_char=91|end_char=92|ner=O

""".lstrip()

EN_DOC_CONLLU_GOLD_MULTIDOC = """
# text = Barack Obama was born in Hawaii.
# sent_id = 0
1	Barack	Barack	PROPN	NNP	Number=Sing	4	nsubj:pass	_	start_char=0|end_char=6|ner=B-PERSON
2	Obama	Obama	PROPN	NNP	Number=Sing	1	flat	_	start_char=7|end_char=12|ner=E-PERSON
3	was	be	AUX	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	4	aux:pass	_	start_char=13|end_char=16|ner=O
4	born	bear	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	0	root	_	start_char=17|end_char=21|ner=O
5	in	in	ADP	IN	_	6	case	_	start_char=22|end_char=24|ner=O
6	Hawaii	Hawaii	PROPN	NNP	Number=Sing	4	obl	_	start_char=25|end_char=31|ner=S-GPE
7	.	.	PUNCT	.	_	4	punct	_	start_char=31|end_char=32|ner=O

# text = He was elected president in 2008.
# sent_id = 1
1	He	he	PRON	PRP	Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs	3	nsubj:pass	_	start_char=0|end_char=2|ner=O
2	was	be	AUX	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	3	aux:pass	_	start_char=3|end_char=6|ner=O
3	elected	elect	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	0	root	_	start_char=7|end_char=14|ner=O
4	president	president	NOUN	NN	Number=Sing	3	xcomp	_	start_char=15|end_char=24|ner=O
5	in	in	ADP	IN	_	6	case	_	start_char=25|end_char=27|ner=O
6	2008	2008	NUM	CD	NumForm=Digit|NumType=Card	3	obl	_	start_char=28|end_char=32|ner=S-DATE
7	.	.	PUNCT	.	_	3	punct	_	start_char=32|end_char=33|ner=O

# text = Obama attended Harvard.
# sent_id = 2
1	Obama	Obama	PROPN	NNP	Number=Sing	2	nsubj	_	start_char=0|end_char=5|ner=S-PERSON
2	attended	attend	VERB	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	0	root	_	start_char=6|end_char=14|ner=O
3	Harvard	Harvard	PROPN	NNP	Number=Sing	2	obj	_	start_char=15|end_char=22|ner=S-ORG
4	.	.	PUNCT	.	_	2	punct	_	start_char=22|end_char=23|ner=O

""".lstrip()

class TestEnglishPipeline:
    @pytest.fixture(scope="class")
    def pipeline(self):
        return stanza.Pipeline(dir=TEST_MODELS_DIR)

    @pytest.fixture(scope="class")
    def processed_doc(self, pipeline):
        """ Document created by running full English pipeline on a few sentences """
        return pipeline(EN_DOC)

    def test_text(self, processed_doc):
        assert processed_doc.text == EN_DOC


    def test_conllu(self, processed_doc):
        assert CoNLL.doc2conll_text(processed_doc) == EN_DOC_CONLLU_GOLD


    def test_tokens(self, processed_doc):
        assert "\n\n".join([sent.tokens_string() for sent in processed_doc.sentences]) == EN_DOC_TOKENS_GOLD


    def test_words(self, processed_doc):
        assert "\n\n".join([sent.words_string() for sent in processed_doc.sentences]) == EN_DOC_WORDS_GOLD


    def test_dependency_parse(self, processed_doc):
        assert "\n\n".join([sent.dependencies_string() for sent in processed_doc.sentences]) == \
               EN_DOC_DEPENDENCY_PARSES_GOLD

    def test_empty(self, pipeline):
        # make sure that various models handle the degenerate empty case
        pipeline("")
        pipeline("--")

    @pytest.fixture(scope="class")
    def processed_multidoc(self, pipeline):
        """ Document created by running full English pipeline on a few sentences """
        docs = [Document([], text=t) for t in EN_DOCS]
        return pipeline(docs)


    def test_conllu_multidoc(self, processed_multidoc):
        assert "".join([CoNLL.doc2conll_text(doc) for doc in processed_multidoc]) == EN_DOC_CONLLU_GOLD_MULTIDOC

    def test_tokens_multidoc(self, processed_multidoc):
        assert "\n\n".join([sent.tokens_string() for processed_doc in processed_multidoc for sent in processed_doc.sentences]) == EN_DOC_TOKENS_GOLD


    def test_words_multidoc(self, processed_multidoc):
        assert "\n\n".join([sent.words_string() for processed_doc in processed_multidoc for sent in processed_doc.sentences]) == EN_DOC_WORDS_GOLD

    def test_sentence_indices_multidoc(self, processed_multidoc):
        sentences = [sent for doc in processed_multidoc for sent in doc.sentences]
        for sent_idx, sentence in enumerate(sentences):
            assert sent_idx == sentence.index

    def test_dependency_parse_multidoc(self, processed_multidoc):
        assert "\n\n".join([sent.dependencies_string() for processed_doc in processed_multidoc for sent in processed_doc.sentences]) == \
               EN_DOC_DEPENDENCY_PARSES_GOLD


    @pytest.fixture(scope="class")
    def processed_multidoc_variant(self):
        """ Document created by running full English pipeline on a few sentences """
        docs = [Document([], text=t) for t in EN_DOCS]
        nlp = stanza.Pipeline(dir=TEST_MODELS_DIR, processors={'tokenize': 'spacy'})
        return nlp(docs)

    def test_dependency_parse_multidoc_variant(self, processed_multidoc_variant):
        assert "\n\n".join([sent.dependencies_string() for processed_doc in processed_multidoc_variant for sent in processed_doc.sentences]) == \
               EN_DOC_DEPENDENCY_PARSES_GOLD

    def test_constituency_parser(self):
        nlp = stanza.Pipeline(dir=TEST_MODELS_DIR, processors="tokenize,pos,constituency")
        doc = nlp("This is a test")
        assert str(doc.sentences[0].constituency) == '(ROOT (S (NP (DT This)) (VP (VBZ is) (NP (DT a) (NN test)))))'
