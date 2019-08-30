"""
Pipeline that runs tokenize,mwt,pos,lemma,depparse
"""

import io
import itertools
import logging
import sys
import torch

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
from stanfordnlp.utils.resources import DEFAULT_MODEL_DIR, default_treebanks, mwt_languages, build_default_config

DEFAULT_PROCESSORS_LIST = f'{TOKENIZE},{MWT},{POS},{LEMMA},{DEPPARSE}'

NAME_TO_PROCESSOR_CLASS = {TOKENIZE: TokenizeProcessor, MWT: MWTProcessor, POS: POSProcessor,
        LEMMA: LemmaProcessor, DEPPARSE: DepparseProcessor, NER: NERProcessor}

PIPELINE_SETTINGS = ['lang', 'shorthand', 'mode']

# list of settings for each processor
PROCESSOR_SETTINGS = {
    TOKENIZE: ['batch_size', 'pretokenized'],
    MWT: ['batch_size', 'dict_only', 'ensemble_dict'],
    POS: ['batch_size'],
    LEMMA: ['batch_size', 'beam_size', 'dict_only', 'ensemble_dict', 'use_identity'],
    DEPPARSE: ['batch_size'],
    NER: ['batch_size']
}

PROCESSOR_SETTINGS_LIST = \
    ['_'.join(psp) for k, v in PROCESSOR_SETTINGS.items() for psp in itertools.product([k], v)]

BOOLEAN_PROCESSOR_SETTINGS = {
    TOKENIZE: ['pretokenized'],
    MWT: ['dict_only', 'ensemble_dict'],
    LEMMA: ['dict_only', 'edit', 'ensemble_dict', 'use_identity']
}

BOOLEAN_PROCESSOR_SETTINGS_LIST = \
    ['_'.join(psp) for k, v in BOOLEAN_PROCESSOR_SETTINGS.items() for psp in itertools.product([k], v)]


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

logging.basicConfig(stream=sys.stderr)

class Pipeline:

    def __init__(self, processors=DEFAULT_PROCESSORS_LIST, lang='en', models_dir=DEFAULT_MODEL_DIR, treebank=None,
                 use_gpu=True, logger=None, **kwargs):
        shorthand = default_treebanks[lang] if treebank is None else treebank
        config = build_default_config(shorthand, models_dir)
        config.update(kwargs)
        self.config = config
        self.config['processors'] = processors
        self.config['lang'] = lang
        self.config['shorthand'] = shorthand
        self.config['models_dir'] = models_dir
        self.processor_names = [n.strip() for n in self.config['processors'].split(',')]
        self.processors = {TOKENIZE: None, MWT: None, LEMMA: None, POS: None, DEPPARSE: None}
        # always use GPU if a GPU device can be found, unless use_gpu is explicitly set to be False
        self.use_gpu = torch.cuda.is_available() and use_gpu
        logger = logging.getLogger() if logger is None else logger
        logger.info("Use device: {}".format("gpu" if self.use_gpu else "cpu"))
        # configs that are the same for all processors
        pipeline_level_configs = {'lang': self.config['lang'], 'shorthand': self.config['shorthand'], 'mode': 'predict'}
        self.standardize_config_values()
        # set up processors
        pipeline_reqs_exceptions = []
        for processor_name in self.processor_names:
            if processor_name == MWT and self.config['shorthand'] not in mwt_languages:
                continue
            logger.info('---')
            logger.info('Loading: ' + processor_name)
            curr_processor_config = self.filter_config(processor_name, self.config)
            curr_processor_config.update(pipeline_level_configs)
            logger.debug('With settings: ')
            logger.debug(curr_processor_config)
            try:
                # try to build processor, throw an exception if there is a requirements issue
                self.processors[processor_name] = NAME_TO_PROCESSOR_CLASS[processor_name](config=curr_processor_config,
                                                                                          pipeline=self,
                                                                                          use_gpu=self.use_gpu,
                                                                                          logger=logger)
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
        logger.info('---')

    def filter_config(self, prefix, config_dict):
        filtered_dict = {}
        for key in config_dict.keys():
            if key.split('_')[0] == prefix:
                filtered_dict['_'.join(key.split('_')[1:])] = config_dict[key]
        return filtered_dict

    @property
    def loaded_processors(self):
        """
        Return all currently loaded processors in execution order.
        :return: list of Processor instances
        """
        return [self.processors[processor_name] for processor_name in self.processor_names
                if self.processors.get(processor_name)]

    def standardize_config_values(self):
        """
        Standardize config settings
        1.) for boolean settings, convert string values to True or False using distutils.util.strtobool
        """
        standardized_entries = {}
        for key, val in self.config.items():
            if key in BOOLEAN_PROCESSOR_SETTINGS_LIST and isinstance(val, str):
                standardized_entries[key] = bool(strtobool(val))
        self.config.update(standardized_entries)

    def process(self, doc):
        # run the pipeline
        for processor_name in self.processor_names:
            if self.processors[processor_name] is not None:
                doc = self.processors[processor_name].process(doc)
        return doc

    def __call__(self, doc):
        assert any([isinstance(doc, str), isinstance(doc, list)]), 'input should be either str or list'
        doc = self.process(doc)
        return doc
