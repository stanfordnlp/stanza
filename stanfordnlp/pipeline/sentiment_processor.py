"""
Processor that attaches a sentiment score to a sentence
"""

from stanfordnlp.models.common import doc
from stanfordnlp.models.common.pretrain import Pretrain
from stanfordnlp.pipeline._constants import *
from stanfordnlp.pipeline.processor import UDProcessor

class SentimentProcessor(UDProcessor):
    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([SENTIMENT])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _set_up_model(self, config, use_gpu):
        # get pretrained word vectors
        self._pretrain = Pretrain(config['pretrain_path'])
        # set up model
        self._model = classifier.load(model_file=config['model_path'], pretrain=self.pretrain)

        # TODO: move this call to load()
        if use_gpu:
            self._model.cuda()

    def process(self, document):
        # TODO: go from document to words, call the classifier, attach the result



