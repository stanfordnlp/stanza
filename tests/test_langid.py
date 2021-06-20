"""
Basic tests of langid module
"""

from stanza.models.common.doc import Document
from stanza.pipeline.core import Pipeline
from stanza.pipeline.multilingual import MultilingualPipeline

def test_langid():
    """
    Basic test of language identification
    """
    english_text = "This is an English sentence."
    french_text = "C'est une phrase française."
    docs = [english_text, french_text]

    nlp = Pipeline(lang='multilingual', processors="langid")
    docs = [Document([], text=text) for text in docs]
    nlp(docs)
    predictions = [doc.lang for doc in docs]
    assert predictions == ["en", "fr"]

def test_multilingual_pipeline():
    """
    Basic test of multilingual pipeline
    """
    
    french_text = "C'est une phrase française."

