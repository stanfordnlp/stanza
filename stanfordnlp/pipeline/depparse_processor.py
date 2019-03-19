"""
Processor for performing dependency parsing
"""

from stanfordnlp.models.common.pretrain import Pretrain
from stanfordnlp.models.common.utils import unsort
from stanfordnlp.models.depparse.data import DataLoader
from stanfordnlp.models.depparse.trainer import Trainer
from stanfordnlp.pipeline._constants import *
from stanfordnlp.pipeline.processor import UDProcessor


class DepparseProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([DEPPARSE])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE, POS])

    def _set_up_model(self, config, use_gpu):
        self._pretrain = Pretrain(config['pretrain_path'])
        self._trainer = Trainer(pretrain=self._pretrain, model_file=config['model_path'], use_cuda=use_gpu)

    def process(self, doc):
        batch = DataLoader(
            doc, self._config['batch_size'], self._config, self._pretrain, vocab=self._vocab, evaluation=True, sort_during_eval=True)
        preds = []
        for i, b in enumerate(batch):
            preds += self._trainer.predict(b)
        preds = unsort(preds, batch.data_orig_idx)
        batch.conll.set(['head', 'deprel'], [y for x in preds for y in x])
