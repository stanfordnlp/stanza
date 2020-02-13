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
from stanfordnlp.pipeline._constants import *
from stanfordnlp.models.common.doc import Document
from stanfordnlp.pipeline.processor import Processor, ProcessorRequirementsException
from stanfordnlp.pipeline.tokenize_processor import TokenizeProcessor
from stanfordnlp.pipeline.mwt_processor import MWTProcessor
from stanfordnlp.pipeline.pos_processor import POSProcessor
from stanfordnlp.pipeline.lemma_processor import LemmaProcessor
from stanfordnlp.pipeline.depparse_processor import DepparseProcessor
from stanfordnlp.pipeline.ner_processor import NERProcessor
from stanfordnlp.utils.resources import DEFAULT_MODEL_DIR, DEFAULT_DOWNLOAD_VERSION, DEFAULT_RESOURCES_FILE, DEFAULT_DEPENDENCIES, PIPELINE_NAMES, maintain_processor_list, add_dependencies, make_table, build_default_config

logger = logging.getLogger(__name__)

NAME_TO_PROCESSOR_CLASS = {TOKENIZE: TokenizeProcessor, MWT: MWTProcessor, POS: POSProcessor,
                           LEMMA: LemmaProcessor, DEPPARSE: DepparseProcessor, NER: NERProcessor}

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
    
    def __init__(self, lang='en', dir=DEFAULT_MODEL_DIR, package='default', processors={}, version=DEFAULT_DOWNLOAD_VERSION, \
            logging_level='INFO', verbose=None, use_gpu=True, **kwargs):
        # Check verbose for easy logging control
        if verbose == False:
            logging_level = 'ERROR'
        elif verbose == True:
            logging_level = 'INFO'
        
        # Set logging level
        logging_level = logging_level.upper()
        all_levels = ['DEBUG', 'INFO', 'WARNING', 'WARN', 'ERROR', 'CRITICAL', 'FATAL']
        if logging_level not in all_levels:
            raise Exception(f"Unrecognized logging level for pipeline: {logging_level}. Must be one of {', '.join(all_levels)}.")
        logger.setLevel(logging_level)

        # Load resources.json to obtain latest packages.
        logger.info('Loading resource file...')
        resources_filepath = os.path.join(dir, DEFAULT_RESOURCES_FILE)
        if not os.path.exists(resources_filepath):
            raise Exception(f"Resources file not found at: {resources_filepath}. Try to download the model again.")
        with open(resources_filepath) as infile:
            resources = json.load(infile)
        if lang not in resources:
            raise Exception(f'Unsupported language: {lang}.')

        # Special case: processors is str, compatible with older verson
        if isinstance(processors, str):
            processors = {processor.strip(): package for processor in processors.split(',')}
            package = None

        # Maintain load list
        self.load_list = maintain_processor_list(resources, lang, package, processors)
        self.load_list = add_dependencies(resources, lang, self.load_list)
        load_table = make_table(['Processor', 'Model'], [row[:2] for row in self.load_list])
        logger.info(f'Load list:\n{load_table}')
        
        # Load processors
        self.use_gpu = torch.cuda.is_available() and use_gpu
        logger.info("Use device: {}".format("gpu" if self.use_gpu else "cpu"))

        # shorthand = default_treebanks[lang] if treebank is None else treebank
        self.config = build_default_config(resources, lang, dir, self.load_list)
        self.config.update(kwargs)

        self.processors = {}

        # configs that are the same for all processors
        pipeline_level_configs = {'lang': lang, 'mode': 'predict'}
        
        # set up processors
        pipeline_reqs_exceptions = []
        for item in self.load_list:
            processor_name, model, _ = item
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

