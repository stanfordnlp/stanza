"""
Basic tests of the depparse processor boolean flags
"""
import pytest

import stanza
from stanza.models.common.doc import Document
from stanza.pipeline.core import PipelineRequirementsException
from stanza.pipeline.processor import Processor, ProcessorVariant, register_processor, register_processor_variant
from stanza.utils.conll import CoNLL
from tests import *

pytestmark = pytest.mark.pipeline

# data for testing
EN_DOC = "This is a test sentence. This is another!"

EN_DOC_LOWERCASE_TOKENS = '''<Token id=1;words=[<Word id=1;text=this>]>
<Token id=2;words=[<Word id=2;text=is>]>
<Token id=3;words=[<Word id=3;text=a>]>
<Token id=4;words=[<Word id=4;text=test>]>
<Token id=5;words=[<Word id=5;text=sentence>]>
<Token id=6;words=[<Word id=6;text=.>]>

<Token id=1;words=[<Word id=1;text=this>]>
<Token id=2;words=[<Word id=2;text=is>]>
<Token id=3;words=[<Word id=3;text=another>]>
<Token id=4;words=[<Word id=4;text=!>]>'''

EN_DOC_LOL_TOKENS = '''<Token id=1;words=[<Word id=1;text=LOL>]>
<Token id=2;words=[<Word id=2;text=LOL>]>
<Token id=3;words=[<Word id=3;text=LOL>]>
<Token id=4;words=[<Word id=4;text=LOL>]>
<Token id=5;words=[<Word id=5;text=LOL>]>
<Token id=6;words=[<Word id=6;text=LOL>]>
<Token id=7;words=[<Word id=7;text=LOL>]>
<Token id=8;words=[<Word id=8;text=LOL>]>'''

@register_processor("lowercase")
class LowercaseProcessor(Processor):
    ''' Processor that lowercases all text '''
    _requires = set(['tokenize'])
    _provides = set(['lowercase'])

    def __init__(self, config, pipeline, use_gpu):
        pass

    def _set_up_model(self, *args):
        pass

    def process(self, doc):
        doc.text = doc.text.lower()
        for sent in doc.sentences:
            for tok in sent.tokens:
                tok.text = tok.text.lower()

            for word in sent.words:
                word.text = word.text.lower()

        return doc

def test_register_processor():
    nlp = stanza.Pipeline(dir=TEST_MODELS_DIR, lang='en', processors='tokenize,lowercase')
    doc = nlp(EN_DOC)
    assert EN_DOC_LOWERCASE_TOKENS == '\n\n'.join(sent.tokens_string() for sent in doc.sentences)

@register_processor_variant("tokenize", "lol")
class LOLTokenizer(ProcessorVariant):
    ''' An alternative tokenizer that splits text by space and replaces all tokens with LOL '''

    def __init__(self, lang):
        pass

    def process(self, text):
        sentence = [{'id': f'{i+1}', 'text': 'LOL'} for i, tok in enumerate(text.split())]
        return Document([sentence], text)

def test_register_processor_variant():
    nlp = stanza.Pipeline(dir=TEST_MODELS_DIR, lang='en', processors={"tokenize": "lol"}, package=None)
    doc = nlp(EN_DOC)
    assert EN_DOC_LOL_TOKENS == '\n\n'.join(sent.tokens_string() for sent in doc.sentences)
