"""
Processors related to SudachiPy in the pipeline.

GitHub Home: https://github.com/WorksApplications/SudachiPy
"""

import re

from stanza.models.common import doc
from stanza.pipeline._constants import TOKENIZE
from stanza.pipeline.processor import ProcessorVariant, register_processor_variant

def check_sudachipy():
    """
    Import necessary components from SudachiPy to perform tokenization.
    """
    try:
        import sudachipy
        import sudachidict_core
    except ImportError:
        raise ImportError(
            "Both sudachipy and sudachidict_core libraries are required. "
            "Try install them with `pip install sudachipy sudachidict_core`. "
            "Go to https://github.com/WorksApplications/SudachiPy for more information."
        )
    return True

@register_processor_variant(TOKENIZE, 'sudachipy')
class SudachiPyTokenizer(ProcessorVariant):
    def __init__(self, config):
        """ Construct a SudachiPy-based tokenizer.

        Note that this tokenizer uses regex for sentence segmentation.
        """
        if config['lang'] != 'ja':
            raise Exception("SudachiPy tokenizer is only allowed in Japanese pipelines.")

        check_sudachipy()
        from sudachipy import tokenizer
        from sudachipy import dictionary

        self.tokenizer = dictionary.Dictionary().create()
        self.no_ssplit = config.get('no_ssplit', False)

    def process(self, document):
        """ Tokenize a document with the SudachiPy tokenizer and wrap the results into a Doc object.
        """
        if isinstance(document, doc.Document):
            text = document.text
        else:
            text = document
        if not isinstance(text, str):
            raise Exception("Must supply a string or Stanza Document object to the SudachiPy tokenizer.")

        # we use the default sudachipy tokenization mode (i.e., mode C)
        # more config needs to be added to support other modes

        tokens = self.tokenizer.tokenize(text)

        sentences = []
        current_sentence = []
        for token in tokens:
            token_text = token.surface()
            # by default sudachipy will output whitespace as a token
            # we need to skip these tokens to be consistent with other tokenizers
            if token_text.isspace():
                continue
            start = token.begin()
            end = token.end()

            token_entry = {
                doc.TEXT: token_text,
                doc.MISC: f"{doc.START_CHAR}={start}|{doc.END_CHAR}={end}"
            }
            current_sentence.append(token_entry)

            if not self.no_ssplit and token_text in ['。', '！', '？', '!', '?']:
                sentences.append(current_sentence)
                current_sentence = []

        if len(current_sentence) > 0:
            sentences.append(current_sentence)

        return doc.Document(sentences, text)
