from collections import namedtuple
from enum import Enum
import logging
import os

from stanza.models.common.vocab import VOCAB_PREFIX
from stanza.models.pos.vocab import XPOSVocab, WordVocab

class XPOSType(Enum):
    XPOS     = 1
    WORD     = 2

XPOSDescription = namedtuple('XPOSDescription', ['xpos_type', 'sep'])
DEFAULT_KEY = XPOSDescription(XPOSType.WORD, None)

logger = logging.getLogger('stanza')

def filter_data(data, idx):
    data_filtered = []
    for sentence in data:
        flag = True
        for token in sentence:
            if token[idx] is None:
                flag = False
        if flag: data_filtered.append(sentence)
    return data_filtered

def choose_simplest_factory(data, shorthand):
    logger.info(f'Original length = {len(data)}')
    data = filter_data(data, idx=2)
    logger.info(f'Filtered length = {len(data)}')
    vocab = WordVocab(data, shorthand, idx=2, ignore=["_"])
    key = DEFAULT_KEY
    best_size = len(vocab) - len(VOCAB_PREFIX)
    if best_size > 20:
        for sep in ['', '-', '+', '|', ',', ':']: # separators
            vocab = XPOSVocab(data, shorthand, idx=2, sep=sep)
            length = sum(len(x) - len(VOCAB_PREFIX) for x in vocab._id2unit.values())
            if length < best_size:
                key = XPOSDescription(XPOSType.XPOS, sep)
                best_size = length
    return key

def build_xpos_vocab(description, data, shorthand):
    if description.xpos_type is XPOSType.WORD:
        return WordVocab(data, shorthand, idx=2, ignore=["_"])

    return XPOSVocab(data, shorthand, idx=2, sep=description.sep)
