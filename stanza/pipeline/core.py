"""
Pipeline that runs tokenize,mwt,pos,lemma,depparse
"""

import argparse
import collections
from enum import Enum
import io
import itertools
import sys
import logging
import json
import os

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
from stanza.pipeline.depparse_processor import DepparseProcessor
from stanza.pipeline.sentiment_processor import SentimentProcessor
from stanza.pipeline.ner_processor import NERProcessor
from stanza.resources.common import DEFAULT_MODEL_DIR, DEFAULT_RESOURCES_URL, DEFAULT_RESOURCES_VERSION, ModelSpecification, add_dependencies, add_mwt, download_models, download_resources_json, flatten_processor_list, load_resources_json, maintain_processor_list, process_pipeline_parameters, set_logging_level, sort_processors
from stanza.utils.helper_func import make_table

logger = logging.getLogger('stanza')

class DownloadMethod(Enum):
    """
    Determines a couple options on how to download resources for the pipeline.

    NONE will not download anything, probably resulting in failure if the resources aren't already in place.
    REUSE_RESOURCES will reuse the existing resources.json and models, but will download any missing models.
    DOWNLOAD_RESOURCES will download a new resources.json and will overwrite any out of date models.
    """
    NONE = 1
    REUSE_RESOURCES = 2
    DOWNLOAD_RESOURCES = 3

class LanguageNotDownloadedError(FileNotFoundError):
    def __init__(self, lang, lang_dir, model_path):
        super().__init__(f'Could not find the model file {model_path}.  The expected model directory {lang_dir} is missing.  Perhaps you need to run stanza.download("{lang}")')
        self.lang = lang
        self.lang_dir = lang_dir
        self.model_path = model_path

class UnsupportedProcessorError(FileNotFoundError):
    def __init__(self, processor, lang):
        super().__init__(f'Processor {processor} is not known for language {lang}.  If you have created your own model, please specify the {processor}_model_path parameter when creating the pipeline.')
        self.processor = processor
        self.lang = lang

class IllegalPackageError(ValueError):
    def __init__(self, msg):
        super().__init__(msg)

class PipelineRequirementsException(Exception):
    """
    Exception indicating one or more requirements failures while attempting to build a pipeline.
    Contains a ProcessorRequirementsException list.
    """

    def __init__(self, processor_req_fails):
        self._processor_req_fails = processor_req_fails
        self.build_message()

    @property
    def processor_req_fails(self):
        return self._processor_req_fails

    def build_message(self):
        err_msg = io.StringIO()
        print(*[req_fail.message for req_fail in self.processor_req_fails], sep='\n', file=err_msg)
        self.message = '\n\n' + err_msg.getvalue()

    def __str__(self):
        return self.message

def build_default_config_option(model_specs):
    """
    Build a config option for a couple situations: lemma=identity, processor is a variant

    Returns the option name and value

    Refactored from build_default_config so that we can reuse it when
    downloading all models
    """
    # handle case when processor variants are used
    if any(model_spec.package in PROCESSOR_VARIANTS[model_spec.processor] for model_spec in model_specs):
        if len(model_specs) > 1:
            raise IllegalPackageError("Variant processor selected for {}, but multiple packages requested".format(model_spec.processor))
        return f"{model_specs[0].processor}_with_{model_specs[0].package}", True
    # handle case when identity is specified as lemmatizer
    elif any(model_spec.processor == LEMMA and model_spec.package == 'identity' for model_spec in model_specs):
        if len(model_specs) > 1:
            raise IllegalPackageError("Identity processor selected for lemma, but multiple packages requested")
        return f"{LEMMA}_use_identity", True
    return None

def filter_variants(model_specs):
    return [(key, value) for (key, value) in model_specs if build_default_config_option(value) is None]

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

def normalize_download_method(download_method):
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

    def __init__(self,
                 lang='en',
                 dir=DEFAULT_MODEL_DIR,
                 package='default',
                 processors={},
                 logging_level=None,
                 verbose=None,
                 use_gpu=None,
                 model_dir=None,
                 download_method=DownloadMethod.DOWNLOAD_RESOURCES,
                 resources_url=DEFAULT_RESOURCES_URL,
                 resources_branch=None,
                 resources_version=DEFAULT_RESOURCES_VERSION,
                 resources_filepath=None,
                 proxies=None,
                 foundation_cache=None,
                 device=None,
                 allow_unknown_language=False,
                 **kwargs):
        self.lang, self.dir, self.kwargs = lang, dir, kwargs
        if model_dir is not None and dir == DEFAULT_MODEL_DIR:
            self.dir = model_dir

        # set global logging level
        set_logging_level(logging_level, verbose)

        # processors can use this to save on the effort of loading
        # large sub-models, such as pretrained embeddings, bert, etc
        if foundation_cache is None:
            self.foundation_cache = FoundationCache()
        else:
            self.foundation_cache = foundation_cache

        download_method = normalize_download_method(download_method)
        if (download_method is DownloadMethod.DOWNLOAD_RESOURCES or
            (download_method is DownloadMethod.REUSE_RESOURCES and not os.path.exists(os.path.join(self.dir, "resources.json")))):
            logger.info("Checking for updates to resources.json in case models have been updated.  Note: this behavior can be turned off with download_method=None or download_method=DownloadMethod.REUSE_RESOURCES")
            download_resources_json(self.dir,
                                    resources_url=resources_url,
                                    resources_branch=resources_branch,
                                    resources_version=resources_version,
                                    resources_filepath=resources_filepath,
                                    proxies=proxies)

        # process different pipeline parameters
        lang, self.dir, package, processors = process_pipeline_parameters(lang, self.dir, package, processors)

        # Load resources.json to obtain latest packages.
        logger.debug('Loading resource file...')
        resources = load_resources_json(self.dir, resources_filepath)
        if lang in resources:
            if 'alias' in resources[lang]:
                logger.info(f'"{lang}" is an alias for "{resources[lang]["alias"]}"')
                lang = resources[lang]['alias']
            lang_name = resources[lang]['lang_name'] if 'lang_name' in resources[lang] else ''
        else:
            logger.warning(f'Unsupported language: {lang}.')

        # Maintain load list
        if lang in resources:
            self.load_list = maintain_processor_list(resources, lang, package, processors, maybe_add_mwt=(not kwargs.get("tokenize_pretokenized")))
            self.load_list = add_dependencies(resources, lang, self.load_list)
            if download_method is not DownloadMethod.NONE:
                # skip processors which aren't downloaded from our collection
                download_list = [x for x in self.load_list if x[0] in resources.get(lang, {})]
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
            self.load_list = [(proc, [ModelSpecification(processor=proc, package='default', dependencies=None)])
                              for proc in list(processors.keys())]
            lang_name = langcode_to_lang(lang)
        else:
            self.load_list = []
        self.load_list = self.update_kwargs(kwargs, self.load_list)
        if len(self.load_list) == 0:
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
                    model_path = curr_processor_config['model_path']
                    if e.filename == model_path or (isinstance(model_path, (tuple, list)) and e.filename in model_path):
                        model_path = e.filename
                    model_dir, model_name = os.path.split(model_path)
                    lang_dir = os.path.dirname(model_dir)
                    if not os.path.exists(lang_dir):
                        # model files for this language can't be found in the expected directory
                        raise LanguageNotDownloadedError(lang, lang_dir, model_path) from e
                    if processor_name not in resources[lang]:
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
    def loaded_processors(self):
        """
        Return all currently loaded processors in execution order.
        :return: list of Processor instances
        """
        return [self.processors[processor_name] for processor_name in PIPELINE_NAMES if self.processors.get(processor_name)]

    def process(self, doc, processors=None):
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
        elif not isinstance(processors, (str, list, tuple, set)):
            raise ValueError("Cannot process {} as a list of processors to run".format(type(processors)))
        else:
            if isinstance(processors, str):
                processors = {x for x in processors.split(",")}
            else:
                processors = set(processors)
            if TOKENIZE in processors and MWT in self.processors and MWT not in processors:
                logger.debug("Requested processors for pipeline did not have mwt, but pipeline needs mwt, so mwt is added")
                processors.add(MWT)
            processors = [x for x in PIPELINE_NAMES if x in processors]

        for processor_name in processors:
            if self.processors.get(processor_name):
                process = self.processors[processor_name].bulk_process if bulk else self.processors[processor_name].process
                doc = process(doc)
        return doc

    def bulk_process(self, docs, *args, **kwargs):
        """
        Run the pipeline in bulk processing mode

        Expects a list of str or a list of Docs
        """
        # Wrap each text as a Document unless it is already such a document
        docs = [doc if isinstance(doc, Document) else Document([], text=doc) for doc in docs]
        return self.process(docs, *args, **kwargs)

    def stream(self, docs, batch_size=50, *args, **kwargs):
        """
        Go through an iterator of documents in batches, yield processed documents

        sentence indices will be counted across the entire iterator
        """
        if not isinstance(docs, collections.abc.Iterator):
            docs = iter(docs)
        def next_batch():
            batch = []
            for _ in range(batch_size):
                try:
                    next_doc = next(docs)
                    batch.append(next_doc)
                except StopIteration:
                    return batch
            return batch

        sentence_start_index = 0
        batch = next_batch()
        while batch:
            batch = self.bulk_process(batch, *args, **kwargs)
            for doc in batch:
                doc.reindex_sentences(sentence_start_index)
                sentence_start_index += len(doc.sentences)
                yield doc
            batch = next_batch()

    def __str__(self):
        """
        Assemble the processors in order to make a simple description of the pipeline
        """
        processors = ["%s=%s" % (x, str(self.processors[x])) for x in PIPELINE_NAMES if x in self.processors]
        return "<Pipeline: %s>" % ", ".join(processors)

    def __call__(self, doc, processors=None):
        return self.process(doc, processors)

def main():
    # TODO: can add a bunch more arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='en', help='Language of the pipeline to use')
    parser.add_argument('--input_file', type=str, required=True, help='Input file to read')
    parser.add_argument('--processors', type=str, default='tokenize,pos,lemma,depparse', help='Processors to use')
    args = parser.parse_args()

    with open(args.input_file, encoding="utf-8") as fin:
        text = fin.read()

    pipe = Pipeline(args.lang, processors=args.processors)
    doc = pipe(text)

    print("{:C}".format(doc))


if __name__ == '__main__':
    main()
