"""
Processors related to PyThaiNLP in the pipeline.

GitHub Home: https://github.com/PyThaiNLP/pythainlp
"""

from stanza.models.common import doc
from stanza.pipeline._constants import TOKENIZE
from stanza.pipeline.processor import ProcessorVariant, register_processor_variant

def check_pythainlp():
    """
    Import necessary components from pythainlp to perform tokenization.
    """
    try:
        import pythainlp
    except ImportError:
        raise ImportError(
            "The pythainlp library is required. "
            "Try to install it with `pip install pythainlp`. "
            "Go to https://github.com/PyThaiNLP/pythainlp for more information."
        )
    return True

@register_processor_variant(TOKENIZE, 'pythainlp')
class PyThaiNLPTokenizer(ProcessorVariant):
    def __init__(self, config):
        """ Construct a PyThaiNLP-based tokenizer.

        Note that we always uses the default tokenizer of PyThaiNLP for sentence and word segmentation.
        Currently this is a CRF model for sentence segmentation and a dictionary-based model (newmm) for word segmentation.
        """
        if config['lang'] != 'th':
            raise Exception("PyThaiNLP tokenizer is only allowed in Thai pipeline.")

        check_pythainlp()
        from pythainlp.tokenize import sent_tokenize as pythai_sent_tokenize
        from pythainlp.tokenize import word_tokenize as pythai_word_tokenize

        self.pythai_sent_tokenize = pythai_sent_tokenize
        self.pythai_word_tokenize = pythai_word_tokenize
        self.no_ssplit = config.get('no_ssplit', False)
    
    def process(self, document):
        """ Tokenize a document with the PyThaiNLP tokenizer and wrap the results into a Doc object.
        """
        if isinstance(document, doc.Document):
            text = document.text
        else:
            text = document
        if not isinstance(text, str):
            raise Exception("Must supply a string or Stanza Document object to the PyThaiNLP tokenizer.")

        sentences = []
        current_sentence = []
        offset = 0

        if self.no_ssplit:
            # skip sentence segmentation
            sent_strs = [text]
        else:
            sent_strs = self.pythai_sent_tokenize(text, engine='crfcut')
        for sent_str in sent_strs:
            for token_str in self.pythai_word_tokenize(sent_str, engine='newmm'):
                # by default pythainlp will output whitespace as a token
                # we need to skip these tokens to be consistent with other tokenizers
                if token_str.isspace():
                    offset += len(token_str)
                    continue
                
                # create token entry
                token_entry = {
                    doc.TEXT: token_str,
                    doc.MISC: f"{doc.START_CHAR}={offset}|{doc.END_CHAR}={offset+len(token_str)}"
                }
                current_sentence.append(token_entry)
                offset += len(token_str)
            
            # finish sentence
            sentences.append(current_sentence)
            current_sentence = []

        if len(current_sentence) > 0:
            sentences.append(current_sentence)

        return doc.Document(sentences, text)