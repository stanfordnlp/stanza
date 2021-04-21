"""
Class for running multilingual pipelines
"""

import stanza
import torch

from stanza.models.common.doc import Document
from stanza.models.langid.model import LangIDBiLSTM
from stanza.pipeline._constants import *

class MultilingualPipeline:
    """
    Pipeline for handling multilingual data. Takes in text, detects language, and routes request to pipeline for that
    language.
    """

    def __init__(
        self,
        lang_configs: dict = None,
        ld_model: str = None,
        ld_batch_size: int = 64,
        max_cache_size: int = 10,
        use_gpu: bool = None
    ):
        # set up configs and cache for various langauge pipelines
        self.lang_configs = {} if lang_configs is None else lang_configs
        self.max_cache_size = max_cache_size
        self.pipeline_cache = {}
        self.lang_request_history = []
        
        # set use_gpu
        if use_gpu is None:
            self.use_gpu = torch.cuda.is_available()
        else:
            self.use_gpu = use_gpu
        
        # build language detector
        self.language_id = LangIDBiLSTM(model=ld_model, use_cuda=use_gpu, batch_size=ld_batch_size)

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
            # clear least recently used lang from pipeline cache
            if len(self.pipeline_cache) == self.max_cache_size:
                lru_lang = self.lang_request_history[0]
                self.pipeline_cache.remove(lru_lang)
                self.lang_request_history.remove(lru_lang)
            self.pipeline_cache[lang] = stanza.Pipeline(**self.lang_configs[lang])

    def process(self, doc):
        """
        Run a Stanza pipeline on a string or list of strings. For each string identify language, and route text to a
        pipeline for that language.
        """

        if isinstance(doc, str):
            doc = [doc]
        
        # determine languages, create per-language batches
        doc_languages = self.language_id.process(doc)
        language_batches = {}
        for text, lang in zip(doc, doc_languages):
            if lang not in language_batches:
                language_batches[lang] = []
            language_batches[lang].append(text)

        # run through each language, submit a batch to the language specific pipeline
        results = []
        for lang in language_batches.keys():
            self._update_pipeline_cache(lang)
            lang_batch = [stanza.Document([], text=d) for d in language_batches[lang]]
            results += self.pipeline_cache[lang](lang_batch)

        return results

    def __call__(self, doc):
        doc = self.process(doc)
        return doc

