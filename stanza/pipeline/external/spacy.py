"""
Processors related to spaCy in the pipeline.
"""

from stanza.models.common import doc
from stanza.pipeline._constants import TOKENIZE
from stanza.pipeline.processor import ProcessorVariant, register_processor_variant

def check_spacy():
    """
    Import necessary components from spaCy to perform tokenization.
    """
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "spaCy is used but not installed on your machine. Go to https://spacy.io/usage for installation instructions."
        )
    return True

@register_processor_variant(TOKENIZE, 'spacy')
class SpacyTokenizer(ProcessorVariant):
    def __init__(self, config):
        """ Construct a spaCy-based tokenizer by loading the spaCy pipeline.
        """
        if config['lang'] != 'en':
            raise Exception("spaCy tokenizer is currently only allowed in English pipeline.")

        try:
            import spacy
            from spacy.lang.en import English
        except ImportError:
            raise ImportError(
                "spaCy 2.0+ is used but not installed on your machine. Go to https://spacy.io/usage for installation instructions."
            )

        # Create a Tokenizer with the default settings for English
        # including punctuation rules and exceptions
        self.nlp = English()
        # by default spacy uses dependency parser to do ssplit
        # we need to add a sentencizer for fast rule-based ssplit
        sentencizer = self.nlp.create_pipe('sentencizer')
        self.nlp.add_pipe(sentencizer)

    def process(self, text):
        """ Tokenize a document with the spaCy tokenizer and wrap the results into a Doc object.
        """
        if not isinstance(text, str):
            raise Exception("Must supply a string to the spaCy tokenizer.")
        spacy_doc = self.nlp(text)

        sentences = []
        for sent in spacy_doc.sents:
            tokens = []
            for tok in sent:
                token_entry = {
                    doc.TEXT: tok.text,
                    doc.MISC: f"{doc.START_CHAR}={tok.idx}|{doc.END_CHAR}={tok.idx+len(tok.text)}"
                }
                tokens.append(token_entry)
            sentences.append(tokens)

        return doc.Document(sentences, text)
