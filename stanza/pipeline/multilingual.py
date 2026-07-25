"""
Class for running multilingual pipelines
"""

from __future__ import annotations

from collections import defaultdict, OrderedDict
from collections.abc import Callable, Mapping, Sequence
import copy
import logging
import os
from typing import Optional, overload, Protocol, TYPE_CHECKING, Union, runtime_checkable

from stanza.models.common.doc import Document
from stanza.models.common.foundation_cache import FoundationCache
from stanza.models.common.utils import default_device
from stanza.models.tokenization.utils import TokenizerPostprocessor
from stanza.pipeline.core import (
    DownloadMethod,
    DownloadMethodInput,
    Pipeline,
    PipelineDevice,
    ProcessorName,
    ProcessorNames,
)
from stanza.resources.common import (
    DEFAULT_MODEL_DIR,
    Package,
    Processors,
    Proxies,
    Resources,
    get_language_resources,
    load_resources_json,
)

if TYPE_CHECKING:
    from typing_extensions import TypedDict

logger = logging.getLogger('stanza')


ConfigurationScalar = Union[
    None,
    bool,
    int,
    float,
    str,
    os.PathLike[str],
    DownloadMethod,
    PipelineDevice,
    FoundationCache,
]
ConfigurationValue = Union[
    ConfigurationScalar,
    Sequence["ConfigurationValue"],
    Mapping[str, "ConfigurationValue"],
]

if TYPE_CHECKING:
    class PipelineConfig(TypedDict, total=False, extra_items=ConfigurationValue):
        lang: str
        dir: str
        package: Optional[Package]
        processors: Optional[Union[ProcessorName, Processors]]
        logging_level: Optional[str]
        verbose: Optional[bool]
        use_gpu: Optional[bool]
        model_dir: Optional[str]
        download_method: Optional[DownloadMethodInput]
        resources_url: str
        resources_branch: Optional[str]
        resources_version: str
        resources_filepath: Optional[Union[str, os.PathLike[str]]]
        proxies: Optional[Proxies]
        foundation_cache: Optional[FoundationCache]
        device: Optional[PipelineDevice]
        allow_unknown_language: bool
        langid_lang_subset: Sequence[str]
        langid_batch_size: int
        langid_clean_text: bool
        tokenize_postprocessor: TokenizerPostprocessor
else:
    PipelineConfig = dict

LanguagePipelineConfigs = Mapping[str, PipelineConfig]
MutableLanguagePipelineConfigs = dict[str, PipelineConfig]

MultilingualDocumentInput = Union[str, Document]
MultilingualDocumentInputs = Union[list[str], list[Document]]


@runtime_checkable
class _LanguageConfigDefaults(Protocol):
    default_factory: Optional[Callable[[], PipelineConfig]]


def _build_pipeline(config: PipelineConfig) -> Pipeline:
    return Pipeline(**config)


def _copy_language_configs(
        lang_configs: Optional[LanguagePipelineConfigs],
    ) -> MutableLanguagePipelineConfigs:
    if lang_configs is None:
        return {}

    copied_configs: MutableLanguagePipelineConfigs = {
        lang: copy.deepcopy(config)
        for lang, config in lang_configs.items()
    }
    if not isinstance(lang_configs, _LanguageConfigDefaults):
        return copied_configs

    default_factory = lang_configs.default_factory
    if default_factory is None:
        return defaultdict(None, copied_configs)

    # deepcopy(defaultdict) preserves the factory itself and only copies
    # entries which already exist.  In particular, do not deepcopy a value
    # returned later by the factory: Pipeline configs may contain a
    # FoundationCache, whose thread lock is intentionally not copyable.
    return defaultdict(default_factory, copied_configs)


class MultilingualPipeline:
    """
    Pipeline for handling multilingual data. Takes in text, detects language, and routes request to pipeline for that
    language.

    You can specify options to individual language pipelines with the lang_configs field.
    For example, if you want English pipelines to have NER, but want to turn that off for French, you can do:
        lang_configs = {"en": {"processors": "tokenize,pos,lemma,depparse,ner"},
                        "fr": {"processors": "tokenize,pos,lemma,depparse"}}
        pipeline = MultilingualPipeline(lang_configs=lang_configs)

    You can also pass in a defaultdict created in such a way that it provides default parameters for each language.
    For example, in order to only get tokenization for each language:
    (remembering that the Pipeline will automagically add MWT to a language which uses MWT):
        from collections import defaultdict
        lang_configs = defaultdict(lambda: dict(processors="tokenize"))
        pipeline = MultilingualPipeline(lang_configs=lang_configs)

    download_method can be set as in Pipeline to turn off downloading
      of the .json config or turn off downloading of everything
    """

    def __init__(self,
                 model_dir: str = DEFAULT_MODEL_DIR,
                 lang_id_config: Optional[PipelineConfig] = None,
                 lang_configs: Optional[LanguagePipelineConfigs] = None,
                 ld_batch_size: int = 64,
                 max_cache_size: int = 10,
                 use_gpu: Optional[bool] = None,
                 restrict: bool = False,
                 device: Optional[PipelineDevice] = None,
                 download_method: Optional[DownloadMethodInput] = DownloadMethod.DOWNLOAD_RESOURCES,
                 processors: Optional[Union[ProcessorName, ProcessorNames]] = None,
    ) -> None:
        # set up configs and cache for various language pipelines
        self.model_dir = model_dir
        self.lang_id_config: PipelineConfig = (
            {}
            if lang_id_config is None
            else copy.deepcopy(lang_id_config)
        )
        self.lang_configs = _copy_language_configs(lang_configs)
        self.max_cache_size = max_cache_size
        # OrderedDict so we can use it as a LRU cache
        # most recent Pipeline goes to the end, pop the oldest one
        # when we run out of space
        self.pipeline_cache: OrderedDict[str, Pipeline] = OrderedDict()
        if processors is None:
            self.default_processors: Optional[list[ProcessorName]] = None
        elif isinstance(processors, str):
            self.default_processors = [x.strip() for x in processors.split(",")]
        else:
            self.default_processors = list(processors)

        self.download_method = download_method
        if 'download_method' not in self.lang_id_config:
            self.lang_id_config['download_method'] = self.download_method

        # if lang is not in any of the lang_configs, update them to
        # include the lang parameter.  otherwise, the default language
        # will always be used...
        for lang in self.lang_configs:
            if 'lang' not in self.lang_configs[lang]:
                self.lang_configs[lang]['lang'] = lang

        if restrict and 'langid_lang_subset' not in self.lang_id_config:
            known_langs = sorted(self.lang_configs.keys())
            logger.debug("Restricting MultilingualPipeline to %s", known_langs)
            self.lang_id_config['langid_lang_subset'] = known_langs

        # set use_gpu
        if device is None:
            if use_gpu is None or use_gpu == True:
                device = default_device()
            else:
                device = 'cpu'
        self.device: PipelineDevice = device
        
        # build language id pipeline
        lang_id_pipeline_config: PipelineConfig = self.lang_id_config.copy()
        for explicit_key in ("dir", "lang", "processors", "device"):
            if explicit_key in lang_id_pipeline_config:
                raise TypeError(
                    f"Pipeline() got multiple values for keyword argument '{explicit_key}'"
                )
        lang_id_pipeline_config.update(
            dir=self.model_dir,
            lang="multilingual",
            processors="langid",
            device=self.device,
        )
        self.lang_id_pipeline = _build_pipeline(lang_id_pipeline_config)
        # load the resources so that we can refer to it later when building a new pipeline
        # note that it was either downloaded or not based on download_method when building the lang_id_pipeline
        self.resources: Resources = load_resources_json(self.model_dir)

    def _update_pipeline_cache(self, lang: str) -> None:
        """
        Do any necessary updates to the pipeline cache for this language. This includes building a new
        pipeline for the lang, and possibly clearing out a language with the old last access date.
        """

        # update request history
        if lang in self.pipeline_cache:
            self.pipeline_cache.move_to_end(lang, last=True)

        # update language configs
        # try/except to allow for a defaultdict
        lang_config: PipelineConfig
        try:
            lang_config = self.lang_configs[lang]
        except KeyError:
            lang_config = {'lang': lang}
            self.lang_configs[lang] = lang_config

        # if a defaultdict is passed in, the defaultdict might not contain 'lang'
        # so even though we tried adding 'lang' in the constructor, we'll check again here
        if 'lang' not in lang_config:
            lang_config['lang'] = lang

        if 'download_method' not in lang_config:
            lang_config['download_method'] = self.download_method

        if 'processors' not in lang_config:
            if self.default_processors:
                lang_resources = get_language_resources(self.resources, lang)
                if not isinstance(lang_resources, Mapping):
                    raise ValueError(
                        f"Cannot load processors for unsupported language: {lang}"
                    )
                lang_processors = [x for x in self.default_processors if x in lang_resources]
                if lang_processors != self.default_processors:
                    logger.info("Not all requested processors %s available for %s.  Loading %s instead", self.default_processors, lang, lang_processors)
                lang_config['processors'] = ",".join(lang_processors)

        if 'device' not in lang_config:
            lang_config['device'] = self.device

        # update pipeline cache
        if lang not in self.pipeline_cache:
            logger.debug("Loading unknown language in MultilingualPipeline: %s", lang)
            # clear least recently used lang from pipeline cache
            if len(self.pipeline_cache) == self.max_cache_size:
                self.pipeline_cache.popitem(last=False)
            pipeline_config: PipelineConfig = self.lang_configs[lang].copy()
            if "dir" in pipeline_config:
                raise TypeError(
                    "Pipeline() got multiple values for keyword argument 'dir'"
                )
            pipeline_config["dir"] = self.model_dir
            self.pipeline_cache[lang] = _build_pipeline(pipeline_config)

    @overload
    def process(
            self,
            doc: MultilingualDocumentInput,
        ) -> Document:
        ...

    @overload
    def process(
            self,
            doc: MultilingualDocumentInputs,
        ) -> list[Document]:
        ...

    def process(
            self,
            doc: Union[MultilingualDocumentInput, MultilingualDocumentInputs],
        ) -> Union[Document, list[Document]]:
        """
        Run language detection on a string, a Document, or a list of either, route to language specific pipeline
        """

        # only return a list if given a list
        singleton_input = not isinstance(doc, list)
        if isinstance(doc, list):
            if doc and isinstance(doc[0], str):
                documents: list[Document] = []
                for text in doc:
                    if not isinstance(text, str):
                        raise TypeError(
                            "MultilingualPipeline batches cannot mix strings and Documents"
                        )
                    documents.append(Document([], text=text))
            else:
                documents = []
                for document in doc:
                    if not isinstance(document, Document):
                        raise TypeError(
                            "MultilingualPipeline batches cannot mix strings and Documents"
                        )
                    documents.append(document)
        elif isinstance(doc, str):
            documents = [Document([], text=doc)]
        else:
            documents = [doc]

        # run language identification
        docs_w_langid = self.lang_id_pipeline.process(documents)

        # create language specific batches, store global idx with each doc
        lang_batches: dict[str, list[Document]] = {}
        for doc_idx, document in enumerate(docs_w_langid):
            lang = document.lang
            if lang is None:
                raise ValueError(
                    f"Language identification returned no language for document {doc_idx}"
                )
            logger.debug("Language for document %d: %s", doc_idx, lang)
            if lang not in lang_batches:
                lang_batches[lang] = []
            lang_batches[lang].append(document)

        # run through each language, submit a batch to the language specific pipeline
        for lang in lang_batches.keys():
            self._update_pipeline_cache(lang)
            self.pipeline_cache[lang](lang_batches[lang])

        # only return a list if given a list
        if singleton_input:
            return docs_w_langid[0]
        else:
            return docs_w_langid

    @overload
    def __call__(
            self,
            doc: MultilingualDocumentInput,
        ) -> Document:
        ...

    @overload
    def __call__(
            self,
            doc: MultilingualDocumentInputs,
        ) -> list[Document]:
        ...

    def __call__(
            self,
            doc: Union[MultilingualDocumentInput, MultilingualDocumentInputs],
        ) -> Union[Document, list[Document]]:
        return self.process(doc)
