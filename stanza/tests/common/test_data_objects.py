"""
Basic tests of the depparse processor boolean flags
"""
import pytest

import stanza
from stanza.models.common.doc import Document, Sentence, Word
from stanza.tests import *

pytestmark = pytest.mark.pipeline

# data for testing
EN_DOC = "This is a test document. Pretty cool!"

EN_DOC_UPOS_XPOS = (('PRON_DT', 'AUX_VBZ', 'DET_DT', 'NOUN_NN', 'NOUN_NN', 'PUNCT_.'), ('ADV_RB', 'ADJ_JJ', 'PUNCT_.'))

EN_DOC2 = "Chris wrote a sentence. Then another."

def test_readonly():
    Document.add_property('some_property', 123)
    nlp = stanza.Pipeline(dir=TEST_MODELS_DIR, lang='en')
    doc = nlp(EN_DOC)
    assert doc.some_property == 123
    with pytest.raises(ValueError):
        doc.some_property = 456


def test_getter():
    Word.add_property('upos_xpos', getter=lambda self: f"{self.upos}_{self.xpos}")
    nlp = stanza.Pipeline(**{'dir': TEST_MODELS_DIR, 'lang': 'en'})

    doc = nlp(EN_DOC)

    assert EN_DOC_UPOS_XPOS == tuple(tuple(word.upos_xpos for word in sentence.words) for sentence in doc.sentences)

def test_setter_getter():
    int2str = {0: 'ok', 1: 'good', 2: 'bad'}
    str2int = {'ok': 0, 'good': 1, 'bad': 2}
    def setter(self, value):
        self._classname = str2int[value]
    Sentence.add_property('classname', getter=lambda self: int2str[self._classname] if self._classname is not None else None, setter=setter)
    nlp = stanza.Pipeline(**{'dir': TEST_MODELS_DIR, 'lang': 'en'})

    doc = nlp(EN_DOC)
    sentence = doc.sentences[0]
    sentence.classname = 'good'
    assert sentence._classname == 1

    # don't try this at home
    sentence._classname = 2
    assert sentence.classname == 'bad'

def test_backpointer():
    nlp = stanza.Pipeline(dir=TEST_MODELS_DIR, lang='en')
    doc = nlp(EN_DOC2)
    ent = doc.ents[0]
    assert ent.sent is doc.sentences[0]
    assert list(doc.iter_words())[0].sent is doc.sentences[0]
    assert list(doc.iter_tokens())[-1].sent is doc.sentences[-1]
