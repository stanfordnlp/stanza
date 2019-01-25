"""
Pipeline that runs tokenize,mwt,pos,lemma,depparse
"""

import torch

from stanfordnlp.pipeline.doc import Document
from stanfordnlp.pipeline.tokenize_processor import TokenizeProcessor
from stanfordnlp.pipeline.mwt_processor import MWTProcessor
from stanfordnlp.pipeline.pos_processor import POSProcessor
from stanfordnlp.pipeline.lemma_processor import LemmaProcessor
from stanfordnlp.pipeline.depparse_processor import DepparseProcessor
from stanfordnlp.utils.resources import DEFAULT_MODEL_DIR, default_treebanks, mwt_languages, build_default_config

DEFAULT_PROCESSORS_LIST = 'tokenize,mwt,pos,lemma,depparse'

NAME_TO_PROCESSOR_CLASS = {'tokenize': TokenizeProcessor, 'mwt': MWTProcessor, 'pos': POSProcessor,
                           'lemma': LemmaProcessor, 'depparse': DepparseProcessor}

PIPELINE_SETTINGS = ['lang', 'shorthand', 'mode']


class Pipeline:

    def __init__(self, processors=DEFAULT_PROCESSORS_LIST, lang='en', models_dir=DEFAULT_MODEL_DIR, treebank=None,
                 cpu=False, **kwargs):
        shorthand = default_treebanks[lang] if treebank is None else treebank
        config = build_default_config(shorthand, models_dir)
        config.update(kwargs)
        self.config = config
        self.config['processors'] = processors
        self.config['lang'] = lang
        self.config['shorthand'] = shorthand
        self.config['models_dir'] = models_dir
        self.processor_names = self.config['processors'].split(',')
        self.processors = {'tokenize': None, 'mwt': None, 'lemma': None, 'pos': None, 'depparse': None}
        # always use GPU if a GPU device can be found, unless cpu is set to be True
        self.use_gpu = torch.cuda.is_available() and not cpu
        print("Use device: {}".format("gpu" if self.use_gpu else "cpu"))
        # configs that are the same for all processors
        pipeline_level_configs = {'lang': self.config['lang'], 'shorthand': self.config['shorthand'], 'mode': 'predict'}
        # set up processors
        for processor_name in self.processor_names:
            if processor_name == 'mwt' and self.config['shorthand'] not in mwt_languages:
                continue
            print('---')
            print('Loading: ' + processor_name)
            curr_processor_config = self.filter_config(processor_name, self.config)
            curr_processor_config.update(pipeline_level_configs)
            print('With settings: ')
            print(curr_processor_config)
            self.processors[processor_name] = NAME_TO_PROCESSOR_CLASS[processor_name](config=curr_processor_config,
                                                                                      use_gpu=self.use_gpu)
        print("Done loading processors!")
        print('---')

    def filter_config(self, prefix, config_dict):
        filtered_dict = {}
        for key in config_dict.keys():
            if key.split('.')[0] == prefix:
                filtered_dict[key.split('.')[1]] = config_dict[key]
        return filtered_dict

    def process(self, doc):
        # run the pipeline
        for processor_name in self.processor_names:
            if self.processors[processor_name] is not None:
                self.processors[processor_name].process(doc)
        doc.load_annotations()

    def __call__(self, doc_str):
        doc = Document(doc_str)
        self.process(doc)
        return doc
