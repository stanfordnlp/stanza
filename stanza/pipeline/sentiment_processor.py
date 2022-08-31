"""Processor that attaches a sentiment score to a sentence

The model used is a generally a model trained on the Stanford
Sentiment Treebank or some similar dataset.  When run, this processor
attaches a score in the form of a string to each sentence in the
document.

TODO: a possible way to generalize this would be to make it a
ClassifierProcessor and have "sentiment" be an option.
"""

from types import SimpleNamespace

import stanza.models.classifiers.cnn_classifier as cnn_classifier
from stanza.models.classifiers.trainer import Trainer

from stanza.models.common import doc
from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor

@register_processor(SENTIMENT)
class SentimentProcessor(UDProcessor):
    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([SENTIMENT])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])

    # default batch size, measured in words per batch
    DEFAULT_BATCH_SIZE = 5000

    def _set_up_model(self, config, pipeline, use_gpu):
        # get pretrained word vectors
        pretrain_path = config.get('pretrain_path', None)
        forward_charlm_path = config.get('forward_charlm_path', None)
        backward_charlm_path = config.get('backward_charlm_path', None)
        # elmo does not have a convenient way to download intermediate
        # models the way stanza downloads charlms & pretrains or
        # transformers downloads bert etc
        # however, elmo in general is not as good as using a
        # transformer, so it is unlikely we will ever fix this
        args = SimpleNamespace(cuda = use_gpu,
                               charlm_forward_file = forward_charlm_path,
                               charlm_backward_file = backward_charlm_path,
                               wordvec_pretrain_file = pretrain_path,
                               elmo_model = None,
                               use_elmo = False)
        # set up model
        trainer = Trainer.load(filename=config['model_path'],
                               args=args,
                               foundation_cache=pipeline.foundation_cache)
        self._model = trainer.model
        # batch size counted as words
        self._batch_size = config.get('batch_size', SentimentProcessor.DEFAULT_BATCH_SIZE)

    def process(self, document):
        sentences = document.sentences
        # TODO: tokens or words better here?
        text = [[token.text for token in sentence.tokens] for sentence in sentences]
        labels = cnn_classifier.label_text(self._model, text, batch_size=self._batch_size)
        # TODO: allow a classifier processor for any attribute, not just sentiment
        document.set(SENTIMENT, labels, to_sentence=True)
        return document
