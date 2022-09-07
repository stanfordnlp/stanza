"""
Class for running multilingual pipelines
"""

import torch

import copy
import logging

from stanza.models.common.doc import Document
from stanza.pipeline.core import Pipeline
from stanza.pipeline._constants import *
from stanza.resources.common import DEFAULT_MODEL_DIR

logger = logging.getLogger('stanza')

class MultilingualPipeline:
    """
    Pipeline for handling multilingual data. Takes in text, detects language, and routes request to pipeline for that
    language.
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        lang_id_config: dict = None,
        lang_configs: dict = None,
        ld_batch_size: int = 64,
        max_cache_size: int = 10,
        use_gpu: bool = None,
        restrict: bool = False,
    ):
        # set up configs and cache for various language pipelines
        self.model_dir = model_dir
        self.lang_id_config = {} if lang_id_config is None else copy.deepcopy(lang_id_config)
        self.lang_configs = {} if lang_configs is None else copy.deepcopy(lang_configs)
        self.max_cache_size = max_cache_size
        self.pipeline_cache = {}
        self.lang_request_history = []

        # if lang is not in any of the lang_configs, update them to
        # include the lang parameter.  otherwise, the default language
        # will always be used...
        for lang in self.lang_configs:
            if 'lang' not in self.lang_configs[lang]:
                self.lang_configs[lang]['lang'] = lang

        if restrict and 'langid_lang_subset' not in self.lang_id_config:
            known_langs = sorted(self.lang_configs.keys())
            if known_langs == 0:
                logger.warning("MultilingualPipeline asked to restrict to lang_configs, but lang_configs was empty.  Ignoring...")
            else:
                logger.debug("Restricting MultilingualPipeline to %s", known_langs)
                self.lang_id_config['langid_lang_subset'] = known_langs

        # set use_gpu
        if use_gpu is None:
            self.use_gpu = torch.cuda.is_available()
        else:
            self.use_gpu = use_gpu
        
        # build language id pipeline
        self.lang_id_pipeline = Pipeline(dir=self.model_dir, lang='multilingual', processors="langid", 
                                         use_gpu=self.use_gpu, **self.lang_id_config)

    def _update_pipeline_cache(self, lang):
        """
        Do any necessary updates to the pipeline cache for this language. This includes building a new
        pipeline for the lang, and possibly clearing out a language with the old last access date.
        """

        # update request history
        if lang in self.lang_request_history:
            self.lang_request_history.remove(lang)
        self.lang_request_history.append(lang)

        # update language configs
        if lang not in self.lang_configs:
            self.lang_configs[lang] = {'lang': lang}

        # update pipeline cache
        if lang not in self.pipeline_cache:
            logger.debug("Loading unknown language in MultilingualPipeline: %s", lang)
            # clear least recently used lang from pipeline cache
            if len(self.pipeline_cache) == self.max_cache_size:
                lru_lang = self.lang_request_history[0]
                self.pipeline_cache.pop(lru_lang)
                self.lang_request_history.remove(lru_lang)
            self.pipeline_cache[lang] = Pipeline(dir=self.model_dir, **self.lang_configs[lang])

    def process(self, doc):
        """
        Run language detection on a string, a Document, or a list of either, route to language specific pipeline
        """

        # only return a list if given a list
        singleton_input = not isinstance(doc, list)
        if singleton_input:
            docs = [doc]
        else:
            docs = doc

        if docs and isinstance(docs[0], str):
            docs = [Document([], text=text) for text in docs]

        # run language identification
        docs_w_langid = self.lang_id_pipeline.process(docs)

        # create language specific batches, store global idx with each doc
        lang_batches = {}
        for doc_idx, doc in enumerate(docs_w_langid):
            logger.debug("Language for document %d: %s", doc_idx, doc.lang)
            if doc.lang not in lang_batches:
                lang_batches[doc.lang] = []
            lang_batches[doc.lang].append(doc)

        # run through each language, submit a batch to the language specific pipeline
        for lang in lang_batches.keys():
            self._update_pipeline_cache(lang)
            self.pipeline_cache[lang](lang_batches[lang])

        # only return a list if given a list
        if singleton_input:
            return docs_w_langid[0]
        else:
            return docs_w_langid

    def __call__(self, doc):
        doc = self.process(doc)
        return doc

