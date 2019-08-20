"""
Processor for performing multi-word-token expansion
"""

import io

from stanfordnlp.models.common import conll
from stanfordnlp.models.mwt.data import DataLoader
from stanfordnlp.models.mwt.trainer import Trainer
from stanfordnlp.pipeline._constants import *
from stanfordnlp.pipeline.processor import UDProcessor


class MWTProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([MWT])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _set_up_model(self, config, use_gpu):
        self._trainer = Trainer(model_file=config['model_path'], use_cuda=use_gpu)

    def process(self, doc):
        batch = DataLoader(doc, self.config['batch_size'], self.config, vocab=self.vocab, evaluation=True)
        if len(batch) > 0:
            dict_preds = self.trainer.predict_dict(batch.doc.get_mwt_expansions(evaluation=True))
            # decide trainer type and run eval
            if self.config['dict_only']:
                preds = dict_preds
            else:
                preds = []
                for i, b in enumerate(batch):
                    preds += self.trainer.predict(b)

                if self.config.get('ensemble_dict', False):
                    preds = self.trainer.ensemble(batch.doc.get_mwt_expansions(evaluation=True), preds)
        else:
            # skip eval if dev data does not exist
            preds = []

        batch.doc.set_mwt(preds)
        return batch.doc

