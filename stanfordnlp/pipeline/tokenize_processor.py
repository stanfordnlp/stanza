"""
Processor for performing tokenization
"""

import io
import logging

from stanfordnlp.models.tokenize.data import DataLoader
from stanfordnlp.models.tokenize.trainer import Trainer
from stanfordnlp.models.tokenize.utils import output_predictions
from stanfordnlp.pipeline._constants import *
from stanfordnlp.pipeline.processor import UDProcessor
from stanfordnlp.utils.postprocess_vietnamese_tokenizer_data import paras_to_chunks
from stanfordnlp.models.common import doc
from stanfordnlp.utils.spacy import SpacyTokenizer

logger = logging.getLogger('stanfordnlp')

# class for running the tokenizer
class TokenizeProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([TOKENIZE])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([])
    # default max sequence length
    MAX_SEQ_LENGTH_DEFAULT = 1000

    def _set_up_model(self, config, use_gpu):
        # set up trainer
        if config.get('pretokenized'):
            self._trainer = None
        elif config.get('with_spacy', False):
            self._trainer = None
            self._spacy_tokenizer = SpacyTokenizer(config.get('lang'))
            logger.info("Using spaCy as tokenizer")
        else:
            self._trainer = Trainer(model_file=config['model_path'], use_cuda=use_gpu)

    def process_pre_tokenized_text(self, input_src):
        """
        Pretokenized text can be provided in 2 manners:

        1.) str, tokenized by whitespace, sentence split by newline
        2.) list of token lists, each token list represents a sentence

        generate dictionary data structure
        """

        document = []
        if isinstance(input_src, str):
            sentences = [sent.rstrip(' ').split() for sent in input_src.rstrip('\n').split('\n') if sent]
        elif isinstance(input_src, list):
            sentences = input_src
        for sentence in sentences:
            sent = []
            for token_id, token in enumerate(sentence):
                sent.append({doc.ID: str(token_id + 1), doc.TEXT: token})
            document.append(sent)
        return document

    def process(self, document):
        assert isinstance(document, str) or (self.config.get('pretokenized') or self.config.get('no_ssplit', False)), \
            "If neither 'pretokenized' or 'no_ssplit' option is enabled, the input to the TokenizerProcessor must be a string."

        if self.config.get('pretokenized'):
            raw_text = None
            document = self.process_pre_tokenized_text(document)
        elif self.config.get('with_spacy', False):
            return self._spacy_tokenizer.tokenize(document)
        else:
            raw_text = document
            # set up batches
            if self.config.get('lang') == 'vi':
                # special processing is due for Vietnamese
                text = '\n\n'.join([x for x in document.split('\n\n')]).rstrip()
                dummy_labels = '\n\n'.join(['0' * len(x) for x in text.split('\n\n')])
                data = paras_to_chunks(text, dummy_labels)
                batches = DataLoader(self.config, input_data=data, vocab=self.vocab, evaluation=True)
            else:
                if isinstance(document, list):
                    document = '\n\n'.join(document)
                batches = DataLoader(self.config, input_text=document, vocab=self.vocab, evaluation=True)
            # get dict data
            _, _, _, document = output_predictions(None, self.trainer, batches, self.vocab, None,
                                   self.config.get('max_seqlen', TokenizeProcessor.MAX_SEQ_LENGTH_DEFAULT),
                                   orig_text = document,
                                   no_ssplit=self.config.get('no_ssplit', False))
        return doc.Document(document, raw_text)
