"""
Pipeline that runs tokenize,mwt,lemma,pos,depparse
"""

from stanfordnlp.pipeline.doc import Document
from stanfordnlp.pipeline.tokenize_processor import TokenizeProcessor
from stanfordnlp.pipeline.mwt_processor import MWTProcessor
from stanfordnlp.pipeline.pos_processor import POSProcessor
from stanfordnlp.pipeline.lemma_processor import LemmaProcessor
from stanfordnlp.pipeline.depparse_processor import DepparseProcessor
from stanfordnlp.utils.resources import DEFAULT_MODEL_DIR, default_treebanks, mwt_languages

class Pipeline:

    def __init__(self, processors='tokenize,mwt,lemma,pos,depparse', lang='en', save_dir=DEFAULT_MODEL_DIR, shorthand=None, **kwargs):
        self.config = kwargs
        self.config['processors'] = processors
        self.config['lang'] = lang
        self.config['shorthand'] = default_treebanks[lang] if shorthand is None else shorthand
        self.config['save_dir'] = save_dir
        self.processor_names = self.config['processors'].split(',')
        self.processors = {'tokenize': None, 'mwt': None, 'lemma': None, 'pos': None, 'depparse': None}
        # set up processors
        if 'tokenize' in self.processor_names:
            print('loading tokenizer...')
            tokenize_config = self.filter_config('tokenize', self.config)
            tokenize_config['model_path'] = f'{self.config["save_dir"]}/{self.config["shorthand"]}_models/{self.config["shorthand"]}_tokenizer.pt'
            print('with settings')
            print(tokenize_config)
            self.processors['tokenize'] = TokenizeProcessor(config=tokenize_config)
        if 'mwt' in self.processor_names and self.config['shorthand'] in mwt_languages:
            mwt_config = self.filter_config('mwt', self.config)
            mwt_config['model_path'] = f'{self.config["save_dir"]}/{self.config["shorthand"]}_models/{self.config["shorthand"]}_mwt_expander.pt'
            print('loading mwt expander...')
            print('with settings')
            print(mwt_config)
            self.processors['mwt'] = MWTProcessor(config=mwt_config)
        if 'pos' in self.processor_names:
            pos_config = self.filter_config('pos', self.config)
            pos_config['model_path'] = f'{self.config["save_dir"]}/{self.config["shorthand"]}_models/{self.config["shorthand"]}_tagger.pt'
            pos_config['pretrain_path'] = f'{self.config["save_dir"]}/{self.config["shorthand"]}_models/{self.config["shorthand"]}_tagger.pretrain.pt'
            print('loading part of speech tagger...')
            print('with settings')
            print(pos_config)
            self.processors['pos'] = POSProcessor(config=pos_config)
        if 'lemma' in self.processor_names:
            lemma_config = self.filter_config('lemma', self.config)
            lemma_config['model_path'] = f'{self.config["save_dir"]}/{self.config["shorthand"]}_models/{self.config["shorthand"]}_lemmatizer.pt'
            print('loading lemmatizer...')
            print('with settings')
            print(lemma_config)
            self.processors['lemma'] = LemmaProcessor(config=lemma_config)
        if 'depparse' in self.processor_names:
            depparse_config = self.filter_config('depparse', self.config)
            depparse_config['model_path'] = f'{self.config["save_dir"]}/{self.config["shorthand"]}_models/{self.config["shorthand"]}_parser.pt'
            depparse_config['pretrain_path'] = f'{self.config["save_dir"]}/{self.config["shorthand"]}_models/{self.config["shorthand"]}_parser.pretrain.pt'
            print('loading dependency parser...')
            print('with settings')
            print(depparse_config)
            self.processors['depparse'] = DepparseProcessor(config=depparse_config)
        print("done loading processors!")

    def filter_config(self, prefix, config_dict):
        filtered_dict = {}
        for key in config_dict.keys():
            if key.split('.')[0] == prefix:
                filtered_dict[key.split('.')[1]] = config_dict[key]
        return filtered_dict

    def process(self, doc):
        # run the pipeline
        for processor_name in ['tokenize', 'mwt', 'pos', 'lemma', 'depparse']:
            if self.processors[processor_name] is not None:
                self.processors[processor_name].process(doc)
        doc.load_annotations()

    def __call__(self, doc_str):
        doc = Document(doc_str)
        self.process(doc)
        return doc
