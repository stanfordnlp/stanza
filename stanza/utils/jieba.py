"""
Utilities related to using Jieba in the pipeline.
"""

import re

from stanza.models.common import doc

def check_jieba():
    """
    Import necessary components from Jieba to perform tokenization.
    """
    try:
        import jieba
    except ImportError:
        raise ImportError(
            "Jieba is used but not installed on your machine. Go to https://pypi.org/project/jieba/ for installation instructions."
        )
    return True

class JiebaTokenizer():
    def __init__(self, lang='zh-hans'):
        """ Construct a Jieba-based tokenizer by loading the Jieba pipeline.

        Note that this tokenizer uses regex for sentence segmentation.
        """
        if lang not in ['zh', 'zh-hans', 'zh-hant']:
            raise Exception("Jieba tokenizer is currently only allowed in Chinese (simplified or traditional) pipelines.")

        check_jieba()
        import jieba
        self.nlp = jieba

    def tokenize(self, text):
        """ Tokenize a document with the Jieba tokenizer and wrap the results into a Doc object.
        """
        if not isinstance(text, str):
            raise Exception("Must supply a string to the Jieba tokenizer.")
        tokens = self.nlp.cut(text, cut_all=False)

        sentences = []
        current_sentence = []
        offset = 0
        for token in tokens:
            if re.match('\s+', token):
                offset += len(token)
                continue

            token_entry = {
                doc.TEXT: token,
                doc.MISC: f"{doc.START_CHAR}={offset}|{doc.END_CHAR}={offset+len(token)}"
            }
            current_sentence.append(token_entry)
            offset += len(token)

            if token in ['。', '！', '？', '!', '?']:
                sentences.append(current_sentence)
                current_sentence = []

        if len(current_sentence) > 0:
            sentences.append(current_sentence)

        return doc.Document(sentences, text)
