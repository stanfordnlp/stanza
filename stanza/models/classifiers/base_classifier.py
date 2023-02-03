from abc import ABC, abstractmethod

import logging

import torch
import torch.nn as nn

from stanza.models.common.utils import split_into_batches, sort_with_indices, unsort

"""
A base classifier type

Currently, has the ability to process text or other inputs in a manner
suitable for the particular model type.
In other words, the CNNClassifier processes lists of words,
and the ConstituencyClassifier processes trees
"""

logger = logging.getLogger('stanza')

class BaseClassifier(ABC, nn.Module):
    @abstractmethod
    def extract_sentences(self, doc):
        """
        Extract the sentences or the relevant information in the sentences from a document
        """

    def preprocess_sentences(self, sentences):
        """
        By default, don't do anything
        """
        return sentences

    def label_sentences(self, sentences, batch_size=None):
        """
        Given a list of sentences, return the model's results on that text.
        """
        self.eval()

        sentences = self.preprocess_sentences(sentences)

        if batch_size is None:
            intervals = [(0, len(sentences))]
            orig_idx = None
        else:
            sentences, orig_idx = sort_with_indices(sentences, key=len, reverse=True)
            intervals = split_into_batches(sentences, batch_size)
        labels = []
        for interval in intervals:
            if interval[1] - interval[0] == 0:
                # this can happen for empty text
                continue
            output = self(sentences[interval[0]:interval[1]])
            predicted = torch.argmax(output, dim=1)
            labels.extend(predicted.tolist())

        if orig_idx:
            sentences = unsort(sentences, orig_idx)
            labels = unsort(labels, orig_idx)

        logger.debug("Found labels")
        for (label, sentence) in zip(labels, sentences):
            logger.debug((label, sentence))

        return labels
