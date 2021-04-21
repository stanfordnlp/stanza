"""
Processor for determing language of text.
"""

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

    def _set_up_model(self, config, use_gpu):
        self._model = LangIDBiLSTM.load(path=config["model_path"], use_cuda=use_gpu, batch_size=config["langid_batch_size"])
        self._char_index = self._model.char_to_idx

    def _text_to_tensor(self, docs):
        """
        Map list of strings to batch tensor
        """

        all_docs = []
        max_len_sequence = max([len(x) for x in docs])
        for doc in docs:
            doc_chars = [self.char_index.get(c, self.char_index["UNK"]) for c in list(doc)]
            doc_chars = doc_chars + [self.char_index["<PAD>"]]*(max_len_sequence-len(doc_chars))
            all_docs.append(doc_chars)
        return torch.tensor(all_docs, device=self.device, dtype=torch.long)


    def _id_langs(self, batch_tensor):
        """
        Identify languages for each sequence in a batch tensor
        """

        scores = self.model(docs_tensor)
        predictions = torch.argmax(scores, dim=1)
        prediction_labels = [self.model.idx_to_tag[prediction] for prediction in predictions]

        return prediction_labels

    def process(self, doc):
        """
        Identify language of string or Document.
        """

        # handle str vs. Document
        inputs = [doc.text if isinstance(doc, Document) else doc]

        # get prediction
        prediction = self._id_langs(self._text_to_tensor(inputs))[0]

        if isinstance(doc, str):
            doc = Document([], text=doc)
        doc.lang = prediction

        return doc

    def _process_list(self, docs):
        """
        Identify language of list of strings or Documents
        """

        if len(docs) == 0:
            # TO DO: what standard do we want for bad input, such as empty list?
            # TO DO: more handling of bad input
            return

        # handle list of str vs. Document
        inputs = [(doc.text if instance(doc,Document) else doc) for doc in docs]

        # get predictions
        predictions = self._id_langs(self._text_to_tensor(inputs))

        if isinstance(docs[0], str):
            docs = [Document([], doc) for doc in docs]

        for doc, lang in zip(docs, predictions):
            doc.lang = lang

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

