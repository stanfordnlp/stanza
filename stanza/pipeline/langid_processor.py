"""
Processor for determining language of text.
"""

from __future__ import annotations

from collections.abc import Sequence
import emoji
import re
from typing import ClassVar, Union

import torch

from stanza.models.common.doc import Document
from stanza.models.langid.model import LangIDBiLSTM
from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor

LangIDInput = Union[str, Document]
LangIDBatch = Union[list[str], list[Document]]


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

    _model: LangIDBiLSTM
    _char_index: dict[str, int]
    _clean_text: bool

    def _set_up_model(self, config, pipeline, device) -> None:
        batch_size = config.get("batch_size", 64)
        self._model = LangIDBiLSTM.load(path=config["model_path"], device=device,
                                        batch_size=batch_size, lang_subset=config.get("lang_subset"))
        self._char_index = self._model.char_to_idx
        self._clean_text = bool(config.get("clean_text"))

    def _text_to_tensor(self, docs: Sequence[str]) -> torch.Tensor:
        """
        Map list of strings to batch tensor. Assumed all docs are same length.
        """

        device = next(self._model.parameters()).device
        all_docs: list[list[int]] = []
        for document in docs:
            doc_chars = [
                self._char_index.get(character, self._char_index["UNK"])
                for character in document
            ]
            all_docs.append(doc_chars)
        return torch.tensor(all_docs, device=device, dtype=torch.long)

    def _id_langs(self, batch_tensor: torch.Tensor) -> list[str]:
        """
        Identify languages for each sequence in a batch tensor
        """
        predictions = self._model.prediction_scores(batch_tensor)
        prediction_labels = [
            self._model.idx_to_tag[int(prediction)]
            for prediction in predictions
        ]

        return prediction_labels

    # regexes for cleaning text
    http_regex = re.compile(r"https?:\/\/t\.co/[a-zA-Z0-9]+")
    handle_regex = re.compile("@[a-zA-Z0-9_]+")
    hashtag_regex = re.compile("#[a-zA-Z]+")
    punctuation_regex = re.compile("[!.]+")
    all_regexes: ClassVar[list[re.Pattern[str]]] = [
        http_regex,
        handle_regex,
        hashtag_regex,
        punctuation_regex,
    ]

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Process text to improve language id performance. Main emphasis is on tweets, this method removes shortened
        urls, hashtags, handles, and punctuation and emoji.
        """

        for regex in LangIDProcessor.all_regexes:
            text = regex.sub(" ", text)

        text = emoji.emojize(text)
        text = emoji.replace_emoji(text, replace=' ')

        if text.strip():
            text = text.strip()

        return text

    def _process_list(self, docs: LangIDBatch) -> list[Document]:
        """
        Identify language of list of strings or Documents
        """

        if len(docs) == 0:
            return []

        if isinstance(docs[0], str):
            documents: list[Document] = []
            for text in docs:
                if not isinstance(text, str):
                    raise TypeError(
                        "Language ID batches cannot mix strings and Documents"
                    )
                documents.append(Document([], text))
        else:
            documents = []
            for document in docs:
                if not isinstance(document, Document):
                    raise TypeError(
                        "Language ID batches cannot mix strings and Documents"
                    )
                documents.append(document)

        docs_by_length: dict[int, list[tuple[Document, str]]] = {}
        for document in documents:
            document_text = document.text
            if not isinstance(document_text, str):
                raise TypeError(
                    "Language ID Documents must contain string text"
                )
            text = (
                LangIDProcessor.clean_text(document_text)
                if self._clean_text
                else document_text
            )
            doc_length = len(text)
            if doc_length not in docs_by_length:
                docs_by_length[doc_length] = []
            docs_by_length[doc_length].append((document, text))

        for doc_length in docs_by_length:
            inputs = [doc[1] for doc in docs_by_length[doc_length]]
            predictions = self._id_langs(self._text_to_tensor(inputs))
            for doc, lang in zip(docs_by_length[doc_length], predictions):
                doc[0].lang = lang

        return documents

    def process(self, doc: LangIDInput) -> Document:
        """
        Handle single str or Document
        """

        if isinstance(doc, str):
            return self._process_list([doc])[0]
        return self._process_list([doc])[0]

    def bulk_process(self, docs: LangIDBatch) -> list[Document]:
        """
        Handle list of strings or Documents
        """

        return self._process_list(docs)
