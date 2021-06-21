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

def test_text_cleaning():
    """
    Basic test of cleaning text
    """
    docs = ["Bonjour le monde! #thisisfrench #ilovefrance",
            "Bonjour le monde! https://t.co/U0Zjp3tusD"]
    docs = [Document([], text=text) for text in docs]
    
    nlp = Pipeline(lang="multilingual", processors="langid")
    nlp(docs)
    assert [doc.lang for doc in docs] == ["it", "it"]
    
    nlp = Pipeline(lang="multilingual", processors="langid", langid_clean_text=True)
    nlp(docs)
    assert [doc.lang for doc in docs] == ["fr", "fr"]

def test_lang_subset():
    """
    Basic test of restricting output to subset of languages
    """
    docs = ["Bonjour le monde! #thisisfrench #ilovefrance",
            "Bonjour le monde! https://t.co/U0Zjp3tusD"]
    docs = [Document([], text=text) for text in docs]
    
    nlp = Pipeline(lang="multilingual", processors="langid")
    nlp(docs)
    assert [doc.lang for doc in docs] == ["it", "it"]
    
    nlp = Pipeline(lang="multilingual", processors="langid", langid_lang_subset=["en","fr"])
    nlp(docs)
    assert [doc.lang for doc in docs] == ["fr", "fr"]
    
    nlp = Pipeline(lang="multilingual", processors="langid", langid_lang_subset=["en"])
    nlp(docs)
    assert [doc.lang for doc in docs] == ["en", "en"]

def test_multilingual_pipeline():
    """
    Basic test of multilingual pipeline
    """
    english_text = "This is an English sentence."
    english_deps_gold = "\n".join((
        "('This', 5, 'nsubj')",
        "('is', 5, 'cop')",
        "('an', 5, 'det')",
        "('English', 5, 'amod')",
        "('sentence', 0, 'root')",
        "('.', 5, 'punct')"
    ))

    french_text = "C'est une phrase française."
    french_deps_gold = "\n".join((
        "(\"C'\", 4, 'nsubj')",
        "('est', 4, 'cop')",
        "('une', 4, 'det')",
        "('phrase', 0, 'root')",
        "('française', 4, 'amod')",
        "('.', 4, 'punct')"
    ))

    nlp = MultilingualPipeline()
    docs = [english_text, french_text]
    docs = nlp(docs)

    assert docs[0].lang == "en"
    assert docs[0].sentences[0].dependencies_string() == english_deps_gold
    assert docs[1].lang == "fr"
    assert docs[1].sentences[0].dependencies_string() == french_deps_gold

