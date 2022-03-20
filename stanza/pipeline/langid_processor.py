"""
Processor for determining language of text.
"""

import emoji
import re
import stanza
import torch

from stanza.models.common.doc import Document
from stanza.models.langid.model import LangIDBiLSTM
from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor


@register_processor(name=LANGID)
class LangIDProcessor(UDProcessor):
    """
    Class for detecting language of text.
    """

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([LANGID])
    
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([])
    
    # default max sequence length
    MAX_SEQ_LENGTH_DEFAULT = 1000

    def _set_up_model(self, config, pipeline, use_gpu):
        batch_size = config.get("batch_size", 64)
        self._model = LangIDBiLSTM.load(path=config["model_path"], use_cuda=use_gpu,
                                        batch_size=batch_size, lang_subset=config.get("lang_subset"))
        self._device = torch.device("cuda") if use_gpu else None
        self._char_index = self._model.char_to_idx
        self._clean_text = config.get("clean_text")

    def _text_to_tensor(self, docs):
        """
        Map list of strings to batch tensor. Assumed all docs are same length.
        """

        all_docs = []
        for doc in docs:
            doc_chars = [self._char_index.get(c, self._char_index["UNK"]) for c in list(doc)]
            all_docs.append(doc_chars)
        return torch.tensor(all_docs, device=self._device, dtype=torch.long)

    def _id_langs(self, batch_tensor):
        """
        Identify languages for each sequence in a batch tensor
        """
        predictions = self._model.prediction_scores(batch_tensor)
        prediction_labels = [self._model.idx_to_tag[prediction] for prediction in predictions]

        return prediction_labels

    # regexes for cleaning text
    http_regex = re.compile("https?:\/\/t\.co/[a-zA-Z0-9]+")
    handle_regex = re.compile("@[a-zA-Z0-9_]+")
    hashtag_regex = re.compile("#[a-zA-Z]+")
    punctuation_regex = re.compile("[!.]+")
    all_regexes = [http_regex, handle_regex, hashtag_regex, punctuation_regex]
    
    @staticmethod
    def clean_text(text):
        """
        Process text to improve language id performance. Main emphasis is on tweets, this method removes shortened
        urls, hashtags, handles, and punctuation and emoji.
        """

        for regex in LangIDProcessor.all_regexes:
            text = regex.sub(" ", text)

        text = emoji.get_emoji_regexp().sub(" ", text)

        if text.strip():
            text = text.strip()

        return text

    def _process_list(self, docs):
        """
        Identify language of list of strings or Documents
        """

        if len(docs) == 0:
            # TO DO: what standard do we want for bad input, such as empty list?
            # TO DO: more handling of bad input
            return

        if isinstance(docs[0], str):
            docs = [Document([], text) for text in docs]

        docs_by_length = {}
        for doc in docs:
            text = LangIDProcessor.clean_text(doc.text) if self._clean_text else doc.text
            doc_length = len(text)
            if doc_length not in docs_by_length:
                docs_by_length[doc_length] = []
            docs_by_length[doc_length].append((doc, text))

        for doc_length in docs_by_length:
            inputs = [doc[1] for doc in docs_by_length[doc_length]]
            predictions = self._id_langs(self._text_to_tensor(inputs))
            for doc, lang in zip(docs_by_length[doc_length], predictions):
                doc[0].lang = lang

        return docs

    def process(self, doc):
        """
        Handle single str or Document
        """
        
        wrapped_doc = [doc]
        return self._process_list(wrapped_doc)[0]

    def bulk_process(self, docs):
        """
        Handle list of strings or Documents
        """

        return self._process_list(docs)

