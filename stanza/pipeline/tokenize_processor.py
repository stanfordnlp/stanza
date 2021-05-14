"""
Processor for performing tokenization
"""

import io
import logging

from stanza.models.tokenization.data import DataLoader, NEWLINE_WHITESPACE_RE
from stanza.models.tokenization.trainer import Trainer
from stanza.models.tokenization.utils import output_predictions
from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor
from stanza.pipeline.registry import PROCESSOR_VARIANTS
from stanza.models.common import doc

# these imports trigger the "register_variant" decorations
from stanza.pipeline.external.jieba import JiebaTokenizer
from stanza.pipeline.external.spacy import SpacyTokenizer
from stanza.pipeline.external.sudachipy import SudachiPyTokenizer
from stanza.pipeline.external.pythainlp import PyThaiNLPTokenizer

logger = logging.getLogger('stanza')

# class for running the tokenizer
@register_processor(name=TOKENIZE)
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

        document = []
        if isinstance(input_src, str):
            sentences = [sent.strip().split() for sent in input_src.strip().split('\n') if len(sent.strip()) > 0]
        elif isinstance(input_src, list):
            sentences = input_src
        idx = 0
        for sentence in sentences:
            sent = []
            for token_id, token in enumerate(sentence):
                sent.append({doc.ID: (token_id + 1, ), doc.TEXT: token, doc.MISC: f'start_char={idx}|end_char={idx + len(token)}'})
                idx += len(token) + 1
            document.append(sent)
        raw_text = ' '.join([' '.join(sentence) for sentence in sentences])
        return raw_text, document

    def process(self, document):
        assert isinstance(document, str) or isinstance(document, doc.Document) or (self.config.get('pretokenized') or self.config.get('no_ssplit', False)), \
            "If neither 'pretokenized' or 'no_ssplit' option is enabled, the input to the TokenizerProcessor must be a string or a Document object."

        if isinstance(document, doc.Document):
            if self.config.get('pretokenized'):
                return document
            document = document.text

        if self.config.get('pretokenized'):
            raw_text, document = self.process_pre_tokenized_text(document)
            return doc.Document(document, raw_text)

        if hasattr(self, '_variant'):
            return self._variant.process(document)

        raw_text = '\n\n'.join(document) if isinstance(document, list) else document
        # set up batches
        batches = DataLoader(self.config, input_text=raw_text, vocab=self.vocab, evaluation=True)
        # get dict data
        _, _, _, document = output_predictions(None, self.trainer, batches, self.vocab, None,
                                               self.config.get('max_seqlen', TokenizeProcessor.MAX_SEQ_LENGTH_DEFAULT),
                                               orig_text=raw_text,
                                               no_ssplit=self.config.get('no_ssplit', False))
        return doc.Document(document, raw_text)

    def bulk_process(self, docs):
        """
        The tokenizer cannot use UDProcessor's sentence-level cross-document batching interface, and requires special handling.
        Essentially, this method concatenates the text of multiple documents with "\n\n", tokenizes it with the neural tokenizer,
        then splits the result into the original Documents and recovers the original character offsets.
        """
        if hasattr(self, '_variant'):
            return self._variant.bulk_process(docs)

        if self.config.get('pretokenized'):
            res = []
            for document in docs:
                raw_text, document = self.process_pre_tokenized_text(document.text)
                res.append(doc.Document(document, raw_text))
            return res

        combined_text = '\n\n'.join([thisdoc.text for thisdoc in docs])
        processed_combined = self.process(doc.Document([], text=combined_text))

        # postprocess sentences and tokens to reset back pointers and char offsets
        charoffset = 0
        sentst = senten = 0
        for thisdoc in docs:
            while senten < len(processed_combined.sentences) and processed_combined.sentences[senten].tokens[-1].end_char - charoffset <= len(thisdoc.text):
                senten += 1

            sentences = processed_combined.sentences[sentst:senten]
            thisdoc.sentences = sentences
            for sent in sentences:
                # fix doc back pointers for sentences
                sent._doc = thisdoc

                # fix char offsets for tokens and words
                for token in sent.tokens:
                    token._start_char -= charoffset
                    token._end_char -= charoffset
                    if token.words:  # not-yet-processed MWT can leave empty tokens
                        for word in token.words:
                            word._start_char -= charoffset
                            word._end_char -= charoffset

            thisdoc.num_tokens = sum(len(sent.tokens) for sent in sentences)
            thisdoc.num_words = sum(len(sent.words) for sent in sentences)
            sentst = senten

            charoffset += len(thisdoc.text) + 2

        return docs
