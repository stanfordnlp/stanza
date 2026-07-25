"""
Pipeline that runs tokenize,mwt,pos,lemma,depparse
"""

import argparse
from collections.abc import Iterable, Iterator, Sequence, Set
from enum import Enum
import io
import itertools
import sys
import logging
import json
import os
from typing import AbstractSet, Optional, Tuple, TypeVar, Union, overload

import torch

from stanza.pipeline._constants import *
from stanza.models.common.constant import langcode_to_lang
from stanza.models.common.doc import Document
from stanza.models.common.foundation_cache import FoundationCache
from stanza.models.common.utils import default_device
from stanza.pipeline.processor import Processor, ProcessorRequirementsException
from stanza.pipeline.registry import NAME_TO_PROCESSOR_CLASS, PIPELINE_NAMES, PROCESSOR_VARIANTS
from stanza.pipeline.langid_processor import LangIDProcessor
from stanza.pipeline.tokenize_processor import TokenizeProcessor
from stanza.pipeline.mwt_processor import MWTProcessor
from stanza.pipeline.pos_processor import POSProcessor
from stanza.pipeline.lemma_processor import LemmaProcessor
from stanza.pipeline.constituency_processor import ConstituencyProcessor
from stanza.pipeline.coref_processor import CorefProcessor
from stanza.pipeline.depparse_processor import DepparseProcessor
from stanza.pipeline.sentiment_processor import SentimentProcessor
from stanza.pipeline.ner_processor import NERProcessor
from stanza.resources.common import DEFAULT_MODEL_DIR, DEFAULT_RESOURCES_URL, DEFAULT_RESOURCES_VERSION, ImmutableProcessorEntry, ModelSpecification, Package, ProcessorEntries, Processors, Proxies, add_dependencies, add_mwt, download_models, download_resources_json, flatten_processor_list, load_resources_json, logging_level_context, maintain_processor_list, process_pipeline_parameters, resolve_language_resources, sort_processors
from stanza.resources.default_packages import PACKAGES
from stanza.utils.conll import CoNLL, CoNLLError
from stanza.utils.helper_func import make_table

logger = logging.getLogger('stanza')

ProcessorName = str
ProcessorNames = Union[Sequence[ProcessorName], AbstractSet[ProcessorName]]
PipelineDevice = Union[str, torch.device]
PretokenizedSentence = list[str]
PretokenizedDocument = list[PretokenizedSentence]
RawPipelineInput = Union[str, PretokenizedSentence, PretokenizedDocument]
PipelineSingleInput = Union[RawPipelineInput, Document]
PipelineInput = Union[PipelineSingleInput, list[Document]]
PipelineBatchItem = Union[str, Document]
_DocumentInputT = TypeVar("_DocumentInputT", Document, list[Document])
_RawPipelineInputT = TypeVar("_RawPipelineInputT", str, PretokenizedSentence, PretokenizedDocument)

class DownloadMethod(Enum):
    """
    Determines a couple options on how to download resources for the pipeline.

    NONE will not download anything, including HF transformers, probably resulting in failure if the resources aren't already in place.
    REUSE_RESOURCES will reuse the existing resources.json and models, but will download any missing models.
    DOWNLOAD_RESOURCES will download a new resources.json and will overwrite any out of date models.
    """
    NONE = 1
    REUSE_RESOURCES = 2
    DOWNLOAD_RESOURCES = 3

DownloadMethodInput = Union[DownloadMethod, str]

class LanguageNotDownloadedError(FileNotFoundError):
    def __init__(self, lang: str, lang_dir: str, model_path: str) -> None:
        super().__init__(f'Could not find the model file {model_path}.  The expected model directory {lang_dir} is missing.  Perhaps you need to run stanza.download("{lang}")')
        self.lang = lang
        self.lang_dir = lang_dir
        self.model_path = model_path

class UnsupportedProcessorError(FileNotFoundError):
    def __init__(self, processor: str, lang: str) -> None:
        super().__init__(f'Processor {processor} is not known for language {lang}.  If you have created your own model, please specify the {processor}_model_path parameter when creating the pipeline.')
        self.processor = processor
        self.lang = lang

class IllegalPackageError(ValueError):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)

class PipelineRequirementsException(Exception):
    """
    Exception indicating one or more requirements failures while attempting to build a pipeline.
    Contains a ProcessorRequirementsException list.
    """

    def __init__(self, processor_req_fails: list[ProcessorRequirementsException]) -> None:
        self._processor_req_fails = processor_req_fails
        self.build_message()

    @property
    def processor_req_fails(self) -> list[ProcessorRequirementsException]:
        return self._processor_req_fails

    def build_message(self) -> None:
        err_msg = io.StringIO()
        print(*[req_fail.message for req_fail in self.processor_req_fails], sep='\n', file=err_msg)
        self.message = '\n\n' + err_msg.getvalue()

    def __str__(self) -> str:
        return self.message


def _run_processor(processor: Processor, document) -> Document:
    """Dispatch across the heterogeneous processor registry.

    Processor's stable contract is Document -> Document.  Tokenization and
    language identification deliberately accept wider concrete inputs; the
    registry key establishes that correlation at runtime.
    """
    return processor.process(document)


def build_default_config_option(
        model_specs: Sequence[ModelSpecification],
    ) -> Optional[Tuple[str, bool]]:
    """
    Build a config option for a couple situations: lemma=identity, processor is a variant

    Returns the option name and value

    Refactored from build_default_config so that we can reuse it when
    downloading all models
    """
    # handle case when processor variants are used
    if any(model_spec.package in PROCESSOR_VARIANTS[model_spec.processor] for model_spec in model_specs):
        if len(model_specs) > 1:
            raise IllegalPackageError("Variant processor selected for {}, but multiple packages requested".format(model_specs[0].processor))
        return f"{model_specs[0].processor}_with_{model_specs[0].package}", True
    # handle case when identity is specified as lemmatizer
    elif any(model_spec.processor == LEMMA and model_spec.package == 'identity' for model_spec in model_specs):
        if len(model_specs) > 1:
            raise IllegalPackageError("Identity processor selected for lemma, but multiple packages requested")
        return f"{LEMMA}_use_identity", True
    return None

def filter_variants(model_specs: ProcessorEntries) -> list[ImmutableProcessorEntry]:
    filtered_model_specs: list[ImmutableProcessorEntry] = []
    for processor, specifications in model_specs:
        if not isinstance(processor, str) or isinstance(specifications, str):
            raise TypeError(
                "Processor list entries must contain a name and model specifications"
            )
        if build_default_config_option(specifications) is None:
            filtered_model_specs.append((processor, specifications))
    return filtered_model_specs

# given a language and models path, build a default configuration
def build_default_config(resources, lang, model_dir, load_list):
    default_config = {}
    for processor, model_specs in load_list:
        option = build_default_config_option(model_specs)
        if option is not None:
            # if an option is set for the model_specs, keep that option and ignore
            # the rest of the model spec
            default_config[option[0]] = option[1]
            continue

        model_paths = [os.path.join(model_dir, lang, processor, model_spec.package + '.pt') for model_spec in model_specs]
        dependencies = [model_spec.dependencies for model_spec in model_specs]

        # Special case for NER: load multiple models at once
        # The pattern will be:
        #   a list of ner_model_path
        #   a list of ner_dependencies
        #     where each item in ner_dependencies is a map
        #     the map may contain forward_charlm_path, backward_charlm_path, or any other deps
        # The user will be able to override the defaults using a semicolon separated string
        # TODO: at least use the same config pattern for all other models
        if processor == NER:
            default_config[f"{processor}_model_path"] = model_paths
            dependency_paths = []
            for dependency_block in dependencies:
                if not dependency_block:
                    dependency_paths.append({})
                    continue
                dependency_paths.append({})
                for dependency in dependency_block:
                    dep_processor, dep_model = dependency
                    dependency_paths[-1][f"{dep_processor}_path"] = os.path.join(model_dir, lang, dep_processor, dep_model + '.pt')
            default_config[f"{processor}_dependencies"] = dependency_paths
            continue

        if len(model_specs) > 1:
            raise IllegalPackageError("Specified multiple packages for {}, which currently only handles one package".format(processor))

        default_config[f"{processor}_model_path"] = model_paths[0]
        if not dependencies[0]: continue
        for dependency in dependencies[0]:
            dep_processor, dep_model = dependency
            default_config[f"{processor}_{dep_processor}_path"] = os.path.join(
                model_dir, lang, dep_processor, dep_model + '.pt'
            )

    return default_config

def normalize_download_method(download_method: Optional[DownloadMethodInput]) -> DownloadMethod:
    """
    Turn None -> DownloadMethod.NONE, strings to the corresponding enum
    """
    if download_method is None:
        return DownloadMethod.NONE
    elif isinstance(download_method, str):
        try:
            return DownloadMethod[download_method.upper()]
        except KeyError as e:
            raise ValueError("Unknown download method %s" % download_method) from e
    return download_method

class Pipeline:
    lang: str
    dir: str
    download_method: DownloadMethod
    foundation_cache: FoundationCache
    processors: dict[str, Processor]
    device: PipelineDevice

    def __init__(self,
                 lang: str = 'en',
                 dir: str = DEFAULT_MODEL_DIR,
                 package: Optional[Package] = 'default',
                 processors: Optional[Union[ProcessorName, Processors]] = {},
                 logging_level: Optional[str] = None,
                 verbose: Optional[bool] = None,
                 use_gpu: Optional[bool] = None,
                 model_dir: Optional[str] = None,
                 download_method: Optional[DownloadMethodInput] = DownloadMethod.DOWNLOAD_RESOURCES,
                 resources_url: str = DEFAULT_RESOURCES_URL,
                 resources_branch: Optional[str] = None,
                 resources_version: str = DEFAULT_RESOURCES_VERSION,
                 resources_filepath: Optional[Union[str, os.PathLike[str]]] = None,
                 proxies: Optional[Proxies] = None,
                 foundation_cache: Optional[FoundationCache] = None,
                 device: Optional[PipelineDevice] = None,
                 allow_unknown_language: bool = False,
                 **kwargs) -> None:
        self.lang, self.dir, self.kwargs = lang, dir, kwargs
        if model_dir is not None and dir == DEFAULT_MODEL_DIR:
            self.dir = model_dir

        # Only adjust logging if the caller explicitly requested it
        with logging_level_context(logging_level, verbose):
            self.download_method = normalize_download_method(download_method)
            if (self.download_method is DownloadMethod.DOWNLOAD_RESOURCES or
                (self.download_method is DownloadMethod.REUSE_RESOURCES and not os.path.exists(os.path.join(self.dir, "resources.json")))):
                logger.info("Checking for updates to resources.json in case models have been updated.  Note: this behavior can be turned off with download_method=None or download_method=DownloadMethod.REUSE_RESOURCES")
                download_resources_json(self.dir,
                                        resources_url=resources_url,
                                        resources_branch=resources_branch,
                                        resources_version=resources_version,
                                        resources_filepath=resources_filepath,
                                        proxies=proxies)

            # processors can use this to save on the effort of loading
            # large sub-models, such as pretrained embeddings, bert, etc
            if foundation_cache is None:
                self.foundation_cache = FoundationCache(local_files_only=(self.download_method is DownloadMethod.NONE))
            else:
                self.foundation_cache = FoundationCache(foundation_cache, local_files_only=(self.download_method is DownloadMethod.NONE))

            # process different pipeline parameters
            normalized_lang, normalized_dir, package, processors = process_pipeline_parameters(
                lang, self.dir, package, processors
            )
            if normalized_lang is None or normalized_dir is None:
                raise ValueError("Pipeline language and model directory cannot be None")
            lang = normalized_lang
            self.dir = normalized_dir

            # Load resources.json to obtain latest packages.
            logger.debug('Loading resource file...')
            resources = load_resources_json(self.dir, resources_filepath)
            requested_lang = lang
            lang, language_resources = resolve_language_resources(
                resources,
                requested_lang,
            )
            if language_resources is not None:
                if lang != requested_lang:
                    logger.info(f'"{requested_lang}" is an alias for "{lang}"')
                lang_name_value = language_resources.get('lang_name')
                if lang_name_value is None:
                    lang_name = ''
                elif isinstance(lang_name_value, str):
                    lang_name = lang_name_value
                else:
                    raise ValueError(
                        f'Invalid resources JSON: lang_name for {lang} must be a string'
                    )
            elif allow_unknown_language:
                logger.warning("Trying to create pipeline for unsupported language: %s", lang)
                lang_name = langcode_to_lang(lang)
            else:
                logger.warning("Unsupported language: %s  If trying to add a new language, consider using allow_unknown_language=True", lang)
                lang_name = langcode_to_lang(lang)

            # Maintain load list
            if language_resources is not None:
                self.load_list = maintain_processor_list(resources, lang, package, processors, maybe_add_mwt=(not kwargs.get("tokenize_pretokenized")))
                self.load_list = add_dependencies(resources, lang, self.load_list)
                if self.download_method is not DownloadMethod.NONE:
                    # skip processors which aren't downloaded from our collection
                    download_list = [
                        entry
                        for entry in self.load_list
                        if (isinstance(entry[0], str)
                            and entry[0] in language_resources)
                    ]
                    # skip variants
                    download_list = filter_variants(download_list)
                    # gather up the model list...
                    download_list = flatten_processor_list(download_list)
                    # download_models will skip models we already have
                    download_models(download_list,
                                    resources=resources,
                                    lang=lang,
                                    model_dir=self.dir,
                                    resources_version=resources_version,
                                    proxies=proxies,
                                    log_info=False)
            elif allow_unknown_language:
                if processors is None:
                    raise ValueError(
                        "Processors must be specified for an unknown language"
                    )
                self.load_list = [(proc, [ModelSpecification(processor=proc, package='default', dependencies=None)])
                                  for proc in processors]
            else:
                self.load_list = []
            self.load_list = self.update_kwargs(kwargs, self.load_list)
            if len(self.load_list) == 0:
                if language_resources is None or PACKAGES not in language_resources:
                    raise ValueError(f'No processors to load for language {lang}.  Language {lang} is currently unsupported')
                else:
                    raise ValueError('No processors to load for language {}.  Please check if your language or package is correctly set.'.format(lang))
            load_table = make_table(['Processor', 'Package'], [(row[0], ";".join(model_spec.package for model_spec in row[1])) for row in self.load_list])
            logger.info(f'Loading these models for language: {lang} ({lang_name}):\n{load_table}')

            self.config = build_default_config(resources, lang, self.dir, self.load_list)
            self.config.update(kwargs)

            # Load processors
            self.processors = {}

            # configs that are the same for all processors
            pipeline_level_configs = {'lang': lang, 'mode': 'predict'}

            if device is None:
                if use_gpu is None or use_gpu == True:
                    device = default_device()
                else:
                    device = 'cpu'
                if use_gpu == True and device == 'cpu':
                    logger.warning("GPU requested, but is not available!")
            self.device = device
            logger.info("Using device: {}".format(self.device))

            # set up processors
            pipeline_reqs_exceptions = []
            for item in self.load_list:
                processor_name, _ = item
                if not isinstance(processor_name, str):
                    raise RuntimeError(
                        "Pipeline processor entries must begin with a processor name"
                    )
                logger.info('Loading: ' + processor_name)
                curr_processor_config = self.filter_config(processor_name, self.config)
                curr_processor_config.update(pipeline_level_configs)
                # TODO: this is obviously a hack
                # a better solution overall would be to make a pretagged version of the pos annotator
                # and then subsequent modules can use those tags without knowing where those tags came from
                if "pretagged" in self.config and "pretagged" not in curr_processor_config:
                    curr_processor_config["pretagged"] = self.config["pretagged"]
                logger.debug('With settings: ')
                logger.debug(curr_processor_config)
                try:
                    # try to build processor, throw an exception if there is a requirements issue
                    self.processors[processor_name] = NAME_TO_PROCESSOR_CLASS[processor_name](config=curr_processor_config,
                                                                                              pipeline=self,
                                                                                              device=self.device)
                except ProcessorRequirementsException as e:
                    # if there was a requirements issue, add it to list which will be printed at end
                    pipeline_reqs_exceptions.append(e)
                    # add the broken processor to the loaded processors for the sake of analyzing the validity of the
                    # entire proposed pipeline, but at this point the pipeline will not be built successfully
                    self.processors[processor_name] = e.err_processor
                except FileNotFoundError as e:
                    # For a FileNotFoundError, we try to guess if there's
                    # a missing model directory or file.  If so, we
                    # suggest the user try to download the models
                    if 'model_path' in curr_processor_config:
                        configured_model_path = curr_processor_config['model_path']
                        model_path: Optional[str] = None
                        if isinstance(configured_model_path, (str, os.PathLike)):
                            path_value = os.fspath(configured_model_path)
                            if isinstance(path_value, str):
                                model_path = path_value
                        elif (isinstance(configured_model_path, (tuple, list))
                              and isinstance(e.filename, str)
                              and e.filename in configured_model_path):
                            model_path = e.filename

                        if model_path is not None:
                            model_dir, model_name = os.path.split(model_path)
                            lang_dir = os.path.dirname(model_dir)
                            if lang_dir and not os.path.exists(lang_dir):
                                # model files for this language can't be found in the expected directory
                                raise LanguageNotDownloadedError(lang, lang_dir, model_path) from e
                            if (language_resources is not None
                                    and processor_name not in language_resources):
                                # user asked for a model which doesn't exist for this language?
                                raise UnsupportedProcessorError(processor_name, lang) from e
                            if not os.path.exists(model_path):
                                model_name, _ = os.path.splitext(model_name)
                                # TODO: before recommending this, check that such a thing exists in resources.json.
                                # currently that case is handled by ignoring the model, anyway
                                raise FileNotFoundError('Could not find model file %s, although there are other models downloaded for language %s.  Perhaps you need to download a specific model.  Try: stanza.download(lang="%s",package=None,processors={"%s":"%s"})' % (model_path, lang, lang, processor_name, model_name)) from e

                    # if we couldn't find a more suitable description of the
                    # FileNotFoundError, just raise the old error
                    raise

            # if there are any processor exceptions, throw an exception to indicate pipeline build failure
            if pipeline_reqs_exceptions:
                logger.info('\n')
                raise PipelineRequirementsException(pipeline_reqs_exceptions)

            logger.info("Done loading processors!")

    @staticmethod
    def update_kwargs(kwargs, processor_list):
        processor_dict = {processor: [{'package': model_spec.package, 'dependencies': model_spec.dependencies} for model_spec in model_specs]
                          for (processor, model_specs) in processor_list}
        for key, value in kwargs.items():
            pieces = key.split('_', 1)
            if len(pieces) == 1:
                continue
            k, v = pieces
            if v == 'model_path':
                package = value if len(value) < 25 else value[:10]+ '...' + value[-10:]
                original_spec = processor_dict.get(k, [])
                if len(original_spec) > 0:
                    dependencies = original_spec[0].get('dependencies')
                else:
                    dependencies = None
                processor_dict[k] = [{'package': package, 'dependencies': dependencies}]
        processor_list = [(processor, [ModelSpecification(processor=processor, package=model_spec['package'], dependencies=model_spec['dependencies']) for model_spec in processor_dict[processor]]) for processor in processor_dict]
        processor_list = sort_processors(processor_list)
        return processor_list

    @staticmethod
    def filter_config(prefix, config_dict):
        filtered_dict = {}
        for key in config_dict.keys():
            pieces = key.split('_', 1)  # split tokenize_pretokenize to tokenize+pretokenize
            if len(pieces) == 1:
                continue
            k, v = pieces
            if k == prefix:
                filtered_dict[v] = config_dict[key]
        return filtered_dict

    @property
    def loaded_processors(self) -> list[Processor]:
        """
        Return all currently loaded processors in execution order.
        :return: list of Processor instances
        """
        return [self.processors[processor_name] for processor_name in PIPELINE_NAMES if self.processors.get(processor_name)]

    @overload
    def process(self, doc: _DocumentInputT, processors: Optional[Union[ProcessorName, ProcessorNames]] = None) -> _DocumentInputT:
        ...

    @overload
    def process(self, doc: _RawPipelineInputT,
                processors: Optional[Union[ProcessorName, ProcessorNames]] = None) -> Union[_RawPipelineInputT, Document]:
        ...

    def process(self, doc: PipelineInput, processors: Optional[Union[ProcessorName, ProcessorNames]] = None) -> PipelineInput:
        """
        Run the pipeline

        processors: allow for a list of processors used by this pipeline action
          can be list, tuple, set, or comma separated string
          if None, use all the processors this pipeline knows about
          MWT is added if necessary
          otherwise, no care is taken to make sure prerequisites are followed...
            some of the annotators, such as depparse, will check, but others
            will fail in some unusual manner or just have really bad results
        """
        assert any([isinstance(doc, str), isinstance(doc, list),
                    isinstance(doc, Document)]), 'input should be either str, list or Document'

        # empty bulk process
        if isinstance(doc, list) and len(doc) == 0:
            return []

        # determine whether we are in bulk processing mode for multiple documents
        bulk=(isinstance(doc, list) and len(doc) > 0 and isinstance(doc[0], Document))

        # various options to limit the processors used by this pipeline action
        if processors is None:
            processors = PIPELINE_NAMES
        elif not isinstance(processors, (str, Sequence, Set)):
            raise ValueError("Cannot process {} as a list of processors to run".format(type(processors)))
        else:
            if isinstance(processors, str):
                processors = {x for x in processors.split(",")}
            else:
                if not all(isinstance(processor, str) for processor in processors):
                    raise ValueError(
                        "Processor selections must contain only strings"
                    )
                processors = set(processors)
            if TOKENIZE in processors and MWT in self.processors and MWT not in processors:
                logger.debug("Requested processors for pipeline did not have mwt, but pipeline needs mwt, so mwt is added")
                processors.add(MWT)
            processors = [x for x in PIPELINE_NAMES if x in processors]

        for processor_name in processors:
            if self.processors.get(processor_name):
                processor = self.processors[processor_name]
                if bulk:
                    if not isinstance(doc, list):
                        raise RuntimeError("Bulk pipeline state was not a document list")
                    documents: list[Document] = []
                    for item in doc:
                        if not isinstance(item, Document):
                            raise TypeError(
                                "Bulk pipeline inputs must contain only Documents"
                            )
                        documents.append(item)
                    doc = processor.bulk_process(documents)
                else:
                    doc = _run_processor(processor, doc)
        return doc

    def process_conllu(self, doc: str, ignore_gapping: bool = True, processors: Optional[Union[ProcessorName, ProcessorNames]] = None) -> Document:
        """ Convenience method: treat the doc as a conllu text, convert it, and process it accordingly """
        if processors is None:
            processors = set(self.processors.keys())
            if TOKENIZE in processors:
                processors.remove(TOKENIZE)
            if MWT in processors:
                processors.remove(MWT)
        converted_doc = CoNLL.conll2doc(input_str=doc, ignore_gapping=ignore_gapping)
        return self.process(converted_doc, processors=processors)

    def process_many(self, docs: Iterable[PipelineBatchItem],
                     processors: Optional[Union[ProcessorName, ProcessorNames]] = None,
                     *args, **kwargs) -> list[Document]:
        """
        Process a collection of documents or texts and return a list of Documents.

        This is a convenience wrapper around the existing bulk processing logic which:
          - Accepts any iterable of strings or Document objects
          - Preserves the input order
          - Always returns a list of Documents
        """
        if docs is None:
            raise ValueError("docs must be an iterable of strings or Documents")
        # Support any iterable, not just lists
        if not isinstance(docs, Iterable) or isinstance(docs, (str, bytes)):
            raise ValueError("docs must be an iterable of strings or Documents, not a single string")
        materialized_docs = list(docs)
        if len(materialized_docs) == 0:
            return []
        if processors is None:
            result = self.bulk_process(materialized_docs, *args, **kwargs)
        else:
            result = self.bulk_process(
                materialized_docs,
                processors,
                *args,
                **kwargs,
            )
        # bulk_process already preserves Documents; ensure we always return a list
        if isinstance(result, list):
            return result
        return list(result)

    def bulk_process(self, docs: Iterable[PipelineBatchItem],
                     processors: Optional[Union[ProcessorName, ProcessorNames]] = None,
                     *args, **kwargs) -> list[Document]:
        """
        Run the pipeline in bulk processing mode

        Expects a list of str or a list of Docs
        """
        # Wrap each text as a Document unless it is already such a document
        wrapped_docs = [doc if isinstance(doc, Document) else Document([], text=doc) for doc in docs]
        if processors is None:
            return self.process(wrapped_docs, *args, **kwargs)
        return self.process(wrapped_docs, processors, *args, **kwargs)

    def stream(self, docs: Iterable[PipelineBatchItem], batch_size: int = 50,
               processors: Optional[Union[ProcessorName, ProcessorNames]] = None,
               *args, **kwargs) -> Iterator[Document]:
        """
        Go through an iterator of documents in batches, yield processed documents

        sentence indices will be counted across the entire iterator
        """
        document_iterator = iter(docs)
        def next_batch() -> list[PipelineBatchItem]:
            batch = []
            for _ in range(batch_size):
                try:
                    next_doc = next(document_iterator)
                    batch.append(next_doc)
                except StopIteration:
                    return batch
            return batch

        sentence_start_index = 0
        batch = next_batch()
        while batch:
            if processors is None:
                processed_batch = self.bulk_process(batch, *args, **kwargs)
            else:
                processed_batch = self.bulk_process(
                    batch,
                    processors,
                    *args,
                    **kwargs,
                )
            for doc in processed_batch:
                doc.reindex_sentences(sentence_start_index)
                sentence_start_index += len(doc.sentences)
                yield doc
            batch = next_batch()

    def __str__(self) -> str:
        """
        Assemble the processors in order to make a simple description of the pipeline
        """
        processors = ["%s=%s" % (x, str(self.processors[x])) for x in PIPELINE_NAMES if x in self.processors]
        return "<Pipeline: %s>" % ", ".join(processors)

    @overload
    def __call__(self, doc: _DocumentInputT, processors: Optional[Union[ProcessorName, ProcessorNames]] = None) -> _DocumentInputT:
        ...

    @overload
    def __call__(self, doc: _RawPipelineInputT,
                 processors: Optional[Union[ProcessorName, ProcessorNames]] = None) -> Union[_RawPipelineInputT, Document]:
        ...

    def __call__(self, doc: PipelineInput, processors: Optional[Union[ProcessorName, ProcessorNames]] = None) -> PipelineInput:
        return self.process(doc, processors)

def main() -> None:
    # TODO: can add a bunch more arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='en', help='Language of the pipeline to use')
    parser.add_argument('--input_file', type=str, required=True, help='Input file to read')
    parser.add_argument('--processors', type=str, default='tokenize,pos,lemma,depparse', help='Processors to use')
    parser.add_argument('--package', type=str, default='default', help='Which package to use')
    parser.add_argument('--tokenize_no_ssplit', default=False, action='store_true', help="Don't ssplit")
    parser.add_argument('--tokenize_pretokenized', default=False, action='store_true', help="Text is pretokenized")
    args, _ = parser.parse_known_args()

    try:
        doc = CoNLL.conll2doc(args.input_file)
        tokenize_pretokenized = True
    except CoNLLError:
        logger.debug("Input file %s does not appear to be a conllu file.  Will read it as a text file")
        with open(args.input_file, encoding="utf-8") as fin:
            doc = fin.read()
        tokenize_pretokenized = args.tokenize_pretokenized

    if args.tokenize_no_ssplit and tokenize_pretokenized:
        pipe = Pipeline(args.lang, package=args.package, processors=args.processors,
                        tokenize_no_ssplit=True, tokenize_pretokenized=True)
    elif args.tokenize_no_ssplit:
        pipe = Pipeline(args.lang, package=args.package, processors=args.processors,
                        tokenize_no_ssplit=True)
    elif tokenize_pretokenized:
        pipe = Pipeline(args.lang, package=args.package, processors=args.processors,
                        tokenize_pretokenized=True)
    else:
        pipe = Pipeline(args.lang, package=args.package, processors=args.processors)

    doc = pipe(doc)

    print("{:C}".format(doc))


if __name__ == '__main__':
    main()
