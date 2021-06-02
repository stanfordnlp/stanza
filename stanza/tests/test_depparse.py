"""
Basic tests of the depparse processor boolean flags
"""
import pytest

import stanza
from stanza.pipeline.core import PipelineRequirementsException
from stanza.utils.conll import CoNLL
from stanza.tests import *

pytestmark = pytest.mark.pipeline

# data for testing
EN_DOC = "Barack Obama was born in Hawaii.  He was elected president in 2008.  Obama attended Harvard."

EN_DOC_CONLLU_PRETAGGED = """
1	Barack	_	PROPN	NNP	Number=Sing	0	_	_	_
2	Obama	_	PROPN	NNP	Number=Sing	1	_	_	_
3	was	_	AUX	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	2	_	_	_
4	born	_	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	3	_	_	_
5	in	_	ADP	IN	_	4	_	_	_
6	Hawaii	_	PROPN	NNP	Number=Sing	5	_	_	_
7	.	_	PUNCT	.	_	6	_	_	_

1	He	_	PRON	PRP	Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs	0	_	_	_
2	was	_	AUX	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	1	_	_	_
3	elected	_	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	2	_	_	_
4	president	_	PROPN	NNP	Number=Sing	3	_	_	_
5	in	_	ADP	IN	_	4	_	_	_
6	2008	_	NUM	CD	NumType=Card	5	_	_	_
7	.	_	PUNCT	.	_	6	_	_	_

1	Obama	_	PROPN	NNP	Number=Sing	0	_	_	_
2	attended	_	VERB	VBD	Mood=Ind|Tense=Past|VerbForm=Fin	1	_	_	_
3	Harvard	_	PROPN	NNP	Number=Sing	2	_	_	_
4	.	_	PUNCT	.	_	3	_	_	_


""".lstrip()

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


def test_depparse():
    nlp = stanza.Pipeline(dir=TEST_MODELS_DIR, lang='en')
    doc = nlp(EN_DOC)
    assert EN_DOC_DEPENDENCY_PARSES_GOLD == '\n\n'.join([sent.dependencies_string() for sent in doc.sentences])


def test_depparse_with_pretagged_doc():
    nlp = stanza.Pipeline(**{'processors': 'depparse', 'dir': TEST_MODELS_DIR, 'lang': 'en',
                                  'depparse_pretagged': True})

    doc = CoNLL.conll2doc(input_str=EN_DOC_CONLLU_PRETAGGED)
    processed_doc = nlp(doc)

    assert EN_DOC_DEPENDENCY_PARSES_GOLD == '\n\n'.join(
        [sent.dependencies_string() for sent in processed_doc.sentences])


def test_raises_requirements_exception_if_pretagged_not_passed():
    with pytest.raises(PipelineRequirementsException):
        stanza.Pipeline(**{'processors': 'depparse', 'dir': TEST_MODELS_DIR, 'lang': 'en'})
