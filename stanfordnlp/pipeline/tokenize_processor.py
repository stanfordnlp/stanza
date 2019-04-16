"""
Processor for performing tokenization
"""

import io

from stanfordnlp.models.common import conll
from stanfordnlp.models.tokenize.data import DataLoader
from stanfordnlp.models.tokenize.trainer import Trainer
from stanfordnlp.models.tokenize.utils import output_predictions
from stanfordnlp.pipeline._constants import *
from stanfordnlp.pipeline.processor import UDProcessor
from stanfordnlp.utils.postprocess_vietnamese_tokenizer_data import paras_to_chunks


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

    def process_pre_tokenized_text(self, doc):
        """
        Pretokenized text can be provided in 2 manners:

        1.) str, tokenized by whitespace, sentence split by newline
        2.) list of token lists, each token list represents a sentence

        generate CoNLL-U output
        """
        conllu_output_string = ""
        if isinstance(doc.text, str):
            sentences = [sent.rstrip(' ').split() for sent in doc.text.rstrip('\n').split('\n') if sent]
        elif isinstance(doc.text, list):
            sentences = doc.text
        for sentence in sentences:
            for token_id, token in enumerate(sentence):
                conllu_data = ['_'] * conll.FIELD_NUM
                conllu_data[conll.FIELD_TO_IDX['id']] = str(token_id + 1)
                conllu_data[conll.FIELD_TO_IDX['word']] = token
                conllu_data[conll.FIELD_TO_IDX['head']] = str(token_id)
                conllu_output_string += ('\t'.join(conllu_data)+'\n')
            conllu_output_string += '\n'
        doc.conll_file = conll.CoNLLFile(input_str=conllu_output_string)

    def process(self, doc):
        if self.config.get('pretokenized'):
            self.process_pre_tokenized_text(doc)
        else:
            # set up batches
            if self.config.get('lang') == 'vi':
                # special processing is due for Vietnamese
                text = '\n\n'.join([x for x in doc.text.split('\n\n')]).rstrip()
                dummy_labels = '\n\n'.join(['0' * len(x) for x in text.split('\n\n')])
                data = paras_to_chunks(text, dummy_labels)
                batches = DataLoader(self.config, input_data=data, vocab=self.vocab, evaluation=True)
            else:
                batches = DataLoader(self.config, input_text=doc.text, vocab=self.vocab, evaluation=True)
            # set up StringIO to get conllu data, run output predictions, set doc's conll file
            with io.StringIO() as conll_output_string:
                output_predictions(conll_output_string, self.trainer, batches, self.vocab, None,
                                   self.config.get('max_seqlen', TokenizeProcessor.MAX_SEQ_LENGTH_DEFAULT))
                # set conll file for doc
                doc.conll_file = conll.CoNLLFile(input_str=conll_output_string.getvalue())

