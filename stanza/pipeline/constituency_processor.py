"""Processor that attaches a constituency tree to a sentence

The model used is a generally a model trained on the Stanford
Sentiment Treebank or some similar dataset.  When run, this processor
attaches a score in the form of a string to each sentence in the
document.

TODO: a possible way to generalize this would be to make it a
ClassifierProcessor and have "sentiment" be an option.
"""

import stanza.models.constituency.trainer as trainer

from stanza.models.common import doc
from stanza.models.common.utils import get_tqdm
from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor

tqdm = get_tqdm()

@register_processor(CONSTITUENCY)
class ConstituencyProcessor(UDProcessor):
    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([CONSTITUENCY])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE, POS])

    # default batch size, measured in sentences
    DEFAULT_BATCH_SIZE = 50

    def _set_up_requires(self):
        self._pretagged = self._config.get('pretagged')
        if self._pretagged:
            self._requires = set()
        else:
            self._requires = self.__class__.REQUIRES_DEFAULT

    def _set_up_model(self, config, pipeline, use_gpu):
        # set up model
        # pretrain and charlm paths are args from the config
        # bert (if used) will be chosen from the model save file
        args = {
            "wordvec_pretrain_file": config.get('pretrain_path', None),
            "charlm_forward_file": config.get('forward_charlm_path', None),
            "charlm_backward_file": config.get('backward_charlm_path', None),
            "cuda": use_gpu,
        }
        self._model = trainer.Trainer.load(filename=config['model_path'],
                                           args=args,
                                           foundation_cache=pipeline.foundation_cache)
        self._model.model.eval()
        # batch size counted as sentences
        self._batch_size = int(config.get('batch_size', ConstituencyProcessor.DEFAULT_BATCH_SIZE))
        self._tqdm = 'tqdm' in config and config['tqdm']

    def process(self, document):
        sentences = document.sentences

        if self._model.uses_xpos():
            words = [[(w.text, w.xpos) for w in s.words] for s in sentences]
        else:
            words = [[(w.text, w.upos) for w in s.words] for s in sentences]
        if self._tqdm:
            words = tqdm(words)

        trees = trainer.parse_tagged_words(self._model.model, words, self._batch_size)
        document.set(CONSTITUENCY, trees, to_sentence=True)
        return document

    def get_constituents(self):
        """
        Return a set of the constituents known by this model

        For a pipeline, this can be queried with
          pipeline.processors["constituency"].get_constituents()
        """
        return set(self._model.model.constituents)
