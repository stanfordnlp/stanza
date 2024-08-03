"""Functions related to BERT or similar models"""

import logging
from typing import List, Tuple

import numpy as np                                 # type: ignore
from transformers import AutoModel, AutoTokenizer  # type: ignore

from stanza.models.coref.config import Config
from stanza.models.coref.const import Doc


logger = logging.getLogger('stanza')

def get_subwords_batches(doc: Doc,
                         config: Config,
                         tok: AutoTokenizer
                         ) -> np.ndarray:
    """
    Turns a list of subwords to a list of lists of subword indices
    of max length == batch_size (or shorter, as batch boundaries
    should match sentence boundaries). Each batch is enclosed in cls and sep
    special tokens.

    Returns:
        batches of bert tokens [n_batches, batch_size]
    """
    batch_size = config.bert_window_size - 2  # to save space for CLS and SEP

    subwords: List[str] = doc["subwords"]
    subwords_batches = []
    start, end = 0, 0

    while end < len(subwords):
        # to prevent the case where a batch_size step forward
        # doesn't capture more than 1 sentence, we will just cut
        # that sequence
        prev_end = end
        end = min(end + batch_size, len(subwords))

        # Move back till we hit a sentence end
        if end < len(subwords):
            sent_id = doc["sent_id"][doc["word_id"][end]]
            while end and doc["sent_id"][doc["word_id"][end - 1]] == sent_id:
                end -= 1

        # this occurs IFF there was no sentence end found throughout
        # the forward scan; this means that our sentence was waay too
        # long (i.e. longer than the max length of the transformer.
        #
        # if so, we give up and just chop the sentence off at the max length
        # that was given
        if end == prev_end:
            end = min(end + batch_size, len(subwords))

        length = end - start
        if tok.cls_token == None or tok.sep_token == None:
            batch = [tok.eos_token] + subwords[start:end] + [tok.eos_token]
        else:
            batch = [tok.cls_token] + subwords[start:end] + [tok.sep_token]

        # Padding to desired length
        batch += [tok.pad_token] * (batch_size - length)

        subwords_batches.append([tok.convert_tokens_to_ids(token)
                                 for token in batch])
        start += length

    return np.array(subwords_batches)
