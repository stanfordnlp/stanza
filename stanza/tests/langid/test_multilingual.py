"""
Tests specifically for the MultilingualPipeline
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Union

import pytest

from stanza.models.common.doc import Document
from stanza.models.common.foundation_cache import FoundationCache
from stanza.pipeline.core import ProcessorName, ProcessorNames
from stanza.pipeline.multilingual import (
    LanguagePipelineConfigs,
    MultilingualPipeline,
    PipelineConfig,
    _copy_language_configs,
)

from stanza.tests import TEST_MODELS_DIR

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]


def run_multilingual_pipeline(
        en_has_dependencies: bool = True,
        fr_has_dependencies: bool = True,
        lang_configs: Optional[LanguagePipelineConfigs] = None,
        max_cache_size: int = 10,
        processors: Optional[Union[ProcessorName, ProcessorNames]] = None,
    ) -> None:
    english_text = "This is an English sentence."
    english_words = ["This", "is", "an", "English", "sentence", "."]
    english_deps_gold = "\n".join((
        "('This', 5, 'nsubj')",
        "('is', 5, 'cop')",
        "('an', 5, 'det')",
        "('English', 5, 'amod')",
        "('sentence', 0, 'root')",
        "('.', 5, 'punct')"
    ))
    if not en_has_dependencies:
        english_deps_gold = ""

    french_text = "C'est une phrase française."
    french_words = ["C'", "est", "une", "phrase", "française", "."]
    french_deps_gold = "\n".join((
        "(\"C'\", 4, 'nsubj')",
        "('est', 4, 'cop')",
        "('une', 4, 'det')",
        "('phrase', 0, 'root')",
        "('française', 4, 'amod')",
        "('.', 4, 'punct')"
    ))
    if not fr_has_dependencies:
        french_deps_gold = ""

    if lang_configs is None:
        lang_configs = {
            "en": {"processors": "tokenize,pos,lemma,depparse"},
            "fr": {"processors": "tokenize,pos,lemma,depparse"},
        }
    nlp = MultilingualPipeline(
        model_dir=TEST_MODELS_DIR,
        download_method=None,
        lang_configs=lang_configs,
        max_cache_size=max_cache_size,
        processors=processors,
    )
    texts = [english_text, french_text]
    processed = nlp(texts)
    if not isinstance(processed, list):
        raise AssertionError("Multilingual batch processing returned one Document")
    docs: list[Document] = processed

    assert docs[0].lang == "en"
    assert len(docs[0].sentences) == 1
    assert [x.text for x in docs[0].sentences[0].words] == english_words
    assert docs[0].sentences[0].dependencies_string() == english_deps_gold

    assert len(docs[1].sentences) == 1
    assert docs[1].lang == "fr"
    assert [x.text for x in docs[1].sentences[0].words] == french_words
    assert docs[1].sentences[0].dependencies_string() == french_deps_gold


def test_multilingual_pipeline() -> None:
    """
    Basic test of multilingual pipeline
    """
    run_multilingual_pipeline()

def test_multilingual_pipeline_small_cache() -> None:
    """
    Test with the cache size 1
    """
    run_multilingual_pipeline(max_cache_size=1)


def test_multilingual_input_shapes() -> None:
    nlp = MultilingualPipeline(
        model_dir=TEST_MODELS_DIR,
        download_method=None,
        lang_configs=defaultdict(lambda: {"processors": "tokenize"}),
    )

    scalar_text = nlp("This is an English sentence.")
    assert isinstance(scalar_text, Document)
    assert scalar_text.lang == "en"

    scalar_document_input = Document([], text="C'est une phrase française.")
    scalar_document = nlp(scalar_document_input)
    assert scalar_document is scalar_document_input
    assert scalar_document.lang == "fr"

    document_batch_input = [
        Document([], text="This is another English sentence."),
        Document([], text="Voici encore une phrase française."),
    ]
    document_batch = nlp(document_batch_input)
    assert isinstance(document_batch, list)
    assert document_batch == document_batch_input
    assert [document.lang for document in document_batch] == ["en", "fr"]


def test_multilingual_config() -> None:
    """
    Test with only tokenize for the EN pipeline
    """
    lang_configs: LanguagePipelineConfigs = {
        "en": {"processors": "tokenize"}
    }

    run_multilingual_pipeline(en_has_dependencies=False, lang_configs=lang_configs)

def test_multilingual_processors_limited() -> None:
    """
    Test loading an available subset of processors
    """
    run_multilingual_pipeline(en_has_dependencies=False, fr_has_dependencies=False, lang_configs={}, processors="tokenize")
    run_multilingual_pipeline(en_has_dependencies=True, fr_has_dependencies=False, lang_configs={"en": {"processors": "tokenize,pos,lemma,depparse"}}, processors="tokenize")
    # this should not fail, as it will drop the zzzzzzzzzz processor for the languages which don't have it
    run_multilingual_pipeline(en_has_dependencies=False, fr_has_dependencies=False, lang_configs={}, processors="tokenize,zzzzzzzzzz")


def test_defaultdict_config() -> None:
    """
    Test that you can pass in a defaultdict for the lang_configs argument
    """
    lang_configs: defaultdict[str, PipelineConfig] = defaultdict(
        lambda: {"processors": "tokenize"}
    )
    run_multilingual_pipeline(en_has_dependencies=False, fr_has_dependencies=False, lang_configs=lang_configs)

    lang_configs = defaultdict(lambda: {"processors": "tokenize"})
    lang_configs["en"] = {"processors": "tokenize,pos,lemma,depparse"}
    run_multilingual_pipeline(en_has_dependencies=True, fr_has_dependencies=False, lang_configs=lang_configs)


def test_defaultdict_config_does_not_copy_factory_values() -> None:
    foundation_cache = FoundationCache()
    lang_configs: defaultdict[str, PipelineConfig] = defaultdict(
        lambda: {"foundation_cache": foundation_cache}
    )

    copied_configs = _copy_language_configs(lang_configs)

    assert copied_configs["en"].get("foundation_cache") is foundation_cache
