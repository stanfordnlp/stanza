"""
Processor for performing tokenization
"""

import io

from stanfordnlp.models.tokenize.data import DataLoader
from stanfordnlp.models.tokenize.trainer import Trainer
from stanfordnlp.models.tokenize.utils import output_predictions
from stanfordnlp.pipeline._constants import *
from stanfordnlp.pipeline.processor import UDProcessor
from stanfordnlp.utils.postprocess_vietnamese_tokenizer_data import paras_to_chunks
from stanfordnlp.pipeline.doc import Document


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
        else:
            self._trainer = Trainer(model_file=config['model_path'], use_cuda=use_gpu)

    def process_pre_tokenized_text(self, input_src):
        """
        Pretokenized text can be provided in 2 manners:

        1.) str, tokenized by whitespace, sentence split by newline
        2.) list of token lists, each token list represents a sentence

        generate dictionary data structure
        """

        doc = []
        if isinstance(input_src, str):
            sentences = [sent.rstrip(' ').split() for sent in doc.rstrip('\n').split('\n') if sent]
        elif isinstance(input_src, list):
            sentences = doc
        for sentence in sentences:
            sent = []
            for token_id, token in enumerate(sentence):
                sent.append({'id': str(token_id + 1), 'word': token})
            doc.append(sent)
        return doc

    def process(self, doc):
        if self.config.get('pretokenized'):
            doc = self.process_pre_tokenized_text(doc)
        else:
            # set up batches
            if self.config.get('lang') == 'vi':
                # special processing is due for Vietnamese
                text = '\n\n'.join([x for x in doc.split('\n\n')]).rstrip()
                dummy_labels = '\n\n'.join(['0' * len(x) for x in text.split('\n\n')])
                data = paras_to_chunks(text, dummy_labels)
                batches = DataLoader(self.config, input_data=data, vocab=self.vocab, evaluation=True)
            else:
                batches = DataLoader(self.config, input_text=doc, vocab=self.vocab, evaluation=True)
            # get dict data
            _, _, _, doc = output_predictions(None, self.trainer, batches, self.vocab, None,
                                   self.config.get('max_seqlen', TokenizeProcessor.MAX_SEQ_LENGTH_DEFAULT),
                                   orig_text = doc)
        return Document(doc)
