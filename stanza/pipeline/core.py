"""
Pipeline that runs tokenize,mwt,pos,lemma,depparse
"""

import io
import itertools
import sys
import torch
import logging
import json
import os

from distutils.util import strtobool
from stanza.pipeline._constants import *
from stanza.models.common.doc import Document
from stanza.pipeline.processor import Processor, ProcessorRequirementsException
from stanza.pipeline.tokenize_processor import TokenizeProcessor
from stanza.pipeline.mwt_processor import MWTProcessor
from stanza.pipeline.pos_processor import POSProcessor
from stanza.pipeline.lemma_processor import LemmaProcessor
from stanza.pipeline.depparse_processor import DepparseProcessor
from stanza.pipeline.sentiment_processor import SentimentProcessor
from stanza.pipeline.ner_processor import NERProcessor
from stanza.utils.resources import DEFAULT_MODEL_DIR, PIPELINE_NAMES, \
    maintain_processor_list, add_dependencies, build_default_config, set_logging_level, process_pipeline_parameters, sort_processors
from stanza.utils.helper_func import make_table

logger = logging.getLogger('stanza')

NAME_TO_PROCESSOR_CLASS = {TOKENIZE: TokenizeProcessor, MWT: MWTProcessor, POS: POSProcessor,
                           LEMMA: LemmaProcessor, DEPPARSE: DepparseProcessor, NER: NERProcessor,
                           SENTIMENT: SentimentProcessor}

# list of settings for each processor
PROCESSOR_SETTINGS = {
    TOKENIZE: ['batch_size', 'pretokenized', 'no_ssplit'],
    MWT: ['batch_size', 'dict_only', 'ensemble_dict'],
    POS: ['batch_size'],
    LEMMA: ['batch_size', 'beam_size', 'dict_only', 'ensemble_dict', 'use_identity'],
    DEPPARSE: ['batch_size', 'pretagged'],
    NER: ['batch_size']
} # TODO: ducumentation

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


class Pipeline:
    
    def __init__(self, lang='en', dir=DEFAULT_MODEL_DIR, package='default', processors={}, logging_level='INFO', verbose=None, use_gpu=True, **kwargs):
        self.lang, self.dir, self.kwargs = lang, dir, kwargs
        
        # set global logging level
        set_logging_level(logging_level, verbose)
        self.logging_level = logging.getLevelName(logger.level)
        # process different pipeline parameters
        lang, dir, package, processors = process_pipeline_parameters(lang, dir, package, processors)

        # Load resources.json to obtain latest packages.
        logger.debug('Loading resource file...')
        resources_filepath = os.path.join(dir, 'resources.json')
        if not os.path.exists(resources_filepath):
            raise Exception(f"Resources file not found at: {resources_filepath}. Try to download the model again.")
        with open(resources_filepath) as infile:
            resources = json.load(infile)
        if lang in resources:
            if 'alias' in resources[lang]:
                logger.info(f'"{lang}" is an alias for "{resources[lang]["alias"]}"')
                lang = resources[lang]['alias']
            lang_name = resources[lang]['lang_name'] if 'lang_name' in resources[lang] else ''
        else:
            logger.warning(f'Unsupported language: {lang}.')

        # Maintain load list
        self.load_list = maintain_processor_list(resources, lang, package, processors) if lang in resources else []
        self.load_list = add_dependencies(resources, lang, self.load_list) if lang in resources else []
        self.load_list = self.update_kwargs(kwargs, self.load_list)
        if len(self.load_list) == 0: raise Exception('No processor to load. Please check if your language or package is correctly set.')
        load_table = make_table(['Processor', 'Package'], [row[:2] for row in self.load_list])
        logger.info(f'Loading these models for language: {lang} ({lang_name}):\n{load_table}')

        self.config = build_default_config(resources, lang, dir, self.load_list)
        self.config.update(kwargs)

        # Load processors
        self.processors = {}

        # configs that are the same for all processors
        pipeline_level_configs = {'lang': lang, 'mode': 'predict'}
        self.use_gpu = torch.cuda.is_available() and use_gpu
        logger.info("Use device: {}".format("gpu" if self.use_gpu else "cpu"))
        
        # set up processors
        pipeline_reqs_exceptions = []
        for item in self.load_list:
            processor_name, _, _ = item
            logger.info('Loading: ' + processor_name)
            curr_processor_config = self.filter_config(processor_name, self.config)
            curr_processor_config.update(pipeline_level_configs)
            logger.debug('With settings: ')
            logger.debug(curr_processor_config)
            try:
                # try to build processor, throw an exception if there is a requirements issue
                self.processors[processor_name] = NAME_TO_PROCESSOR_CLASS[processor_name](config=curr_processor_config,
                                                                                          pipeline=self,
                                                                                          use_gpu=self.use_gpu)
            except ProcessorRequirementsException as e:
                # if there was a requirements issue, add it to list which will be printed at end
                pipeline_reqs_exceptions.append(e)
                # add the broken processor to the loaded processors for the sake of analyzing the validity of the
                # entire proposed pipeline, but at this point the pipeline will not be built successfully
                self.processors[processor_name] = e.err_processor

        # if there are any processor exceptions, throw an exception to indicate pipeline build failure
        if pipeline_reqs_exceptions:
            logger.info('\n')
            raise PipelineRequirementsException(pipeline_reqs_exceptions)

        logger.info("Done loading processors!")

    def update_kwargs(self, kwargs, processor_list):
        processor_dict = {processor: {'package': package, 'dependencies': dependencies} for (processor, package, dependencies) in processor_list}
        for key, value in kwargs.items():
            k, v = key.split('_', 1)
            if v == 'model_path':
                package = value if len(value) < 25 else value[:10]+ '...' + value[-10:]
                dependencies = processor_dict.get(k, {}).get('dependencies')
                processor_dict[k] = {'package': package, 'dependencies': dependencies}
        processor_list = [[processor, processor_dict[processor]['package'], processor_dict[processor]['dependencies']] for processor in processor_dict]
        processor_list = sort_processors(processor_list)
        return processor_list

    def filter_config(self, prefix, config_dict):
        filtered_dict = {}
        for key in config_dict.keys():
            k, v = key.split('_', 1) # split tokenize_pretokenize to tokenize+pretokenize
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

    def process(self, doc):
        # run the pipeline
        for processor_name in PIPELINE_NAMES:
            if self.processors.get(processor_name):
                doc = self.processors[processor_name].process(doc)
        return doc

    def __call__(self, doc):
        assert any([isinstance(doc, str), isinstance(doc, list),
                    isinstance(doc, Document)]), 'input should be either str, list or Document'
        doc = self.process(doc)
        return doc

