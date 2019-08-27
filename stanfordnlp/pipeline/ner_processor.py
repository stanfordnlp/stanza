"""
Processor for performing named entity tagging.
"""

from stanfordnlp.models.common import doc
from stanfordnlp.models.common.pretrain import Pretrain
from stanfordnlp.models.common.utils import unsort
from stanfordnlp.models.ner.data import DataLoader
from stanfordnlp.models.ner.trainer import Trainer
from stanfordnlp.pipeline._constants import *
from stanfordnlp.pipeline.processor import UDProcessor


class NERProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([NER])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _set_up_model(self, config, use_gpu):
        # get pretrained word vectors
        self._pretrain = Pretrain(config['pretrain_path'])
        # set up trainer
        self._trainer = Trainer(pretrain=self.pretrain, model_file=config['model_path'], use_cuda=use_gpu)

    def process(self, document):
        batch = DataLoader(
            document, self.config['batch_size'], self.config, self.pretrain, vocab=self.vocab, evaluation=True)
        preds = []
        for i, b in enumerate(batch):
            preds += self.trainer.predict(b)
        batch.doc.set([doc.NER], [y for x in preds for y in x])
        # TODO: collect entities into document attribute
        return batch.doc
