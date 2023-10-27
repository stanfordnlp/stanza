"""
Keeps BERT, charlm, word embedings in a cache to save memory
"""

import logging
import threading

from stanza.models.common import bert_embedding
from stanza.models.common.char_model import CharacterLanguageModel
from stanza.models.common.pretrain import Pretrain

logger = logging.getLogger('stanza')

class FoundationCache:
    def __init__(self, other=None):
        if other is None:
            self.bert = {}
            self.charlms = {}
            self.pretrains = {}
            # future proof the module by using a lock for the glorious day
            # when the GIL is finally gone
            self.lock = threading.Lock()
        else:
            self.bert = other.bert
            self.charlms = other.charlms
            self.pretrains = other.pretrains
            self.lock = other.lock

    def load_bert(self, transformer_name):
        """
        Load a transformer only once

        Uses a lock for thread safety
        """
        if transformer_name is None:
            return None, None
        with self.lock:
            if transformer_name not in self.bert:
                model, tokenizer = bert_embedding.load_bert(transformer_name)
                self.bert[transformer_name] = (model, tokenizer)
            else:
                logger.debug("Reusing bert %s", transformer_name)

            return self.bert[transformer_name]

    def load_charlm(self, filename):
        if not filename:
            return None

        with self.lock:
            if filename not in self.charlms:
                logger.debug("Loading charlm from %s", filename)
                self.charlms[filename] = CharacterLanguageModel.load(filename, finetune=False)
            else:
                logger.debug("Reusing charlm from %s", filename)

            return self.charlms[filename]

    def load_pretrain(self, filename):
        """
        Load a pretrained word embedding only once

        Uses a lock for thread safety
        """
        if filename is None:
            return None
        with self.lock:
            if filename not in self.pretrains:
                logger.debug("Loading pretrain %s", filename)
                self.pretrains[filename] = Pretrain(filename)
            else:
                logger.debug("Reusing pretrain %s", filename)

            return self.pretrains[filename]

class NoTransformerFoundationCache(FoundationCache):
    """
    Uses the underlying FoundationCache, but hiding the transformer.

    Useful for when loading a downstream model such as POS which has a
    finetuned transformer, and we don't want the transformer reused
    since it will then have the finetuned weights for other models
    which don't want them
    """
    def load_bert(self, transformer_name):
        return load_bert(transformer_name)

def load_bert(model_name, foundation_cache=None):
    """
    Load a bert, possibly using a foundation cache, ignoring it if not present
    """
    if foundation_cache is None:
        return bert_embedding.load_bert(model_name)
    else:
        return foundation_cache.load_bert(model_name)

def load_charlm(charlm_file, foundation_cache=None, finetune=False):
    if not charlm_file:
        return None

    if finetune:
        # can't use the cache in the case of a model which will be finetuned
        # and the numbers will be different for other users of the model
        return CharacterLanguageModel.load(charlm_file, finetune=True)

    if foundation_cache is not None:
        return foundation_cache.load_charlm(charlm_file)

    logger.debug("Loading charlm from %s", charlm_file)
    return CharacterLanguageModel.load(charlm_file, finetune=False)

def load_pretrain(filename, foundation_cache=None):
    if not filename:
        return None

    if foundation_cache is not None:
        return foundation_cache.load_pretrain(filename)

    logger.debug("Loading pretrain from %s", filename)
    return Pretrain(filename)
