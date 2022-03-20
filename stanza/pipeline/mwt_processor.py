"""
Processor for performing multi-word-token expansion
"""

import io

from stanza.models.mwt.data import DataLoader
from stanza.models.mwt.trainer import Trainer
from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor

@register_processor(MWT)
class MWTProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([MWT])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _set_up_model(self, config, pipeline, use_gpu):
        self._trainer = Trainer(model_file=config['model_path'], use_cuda=use_gpu)

    def process(self, document):
        batch = DataLoader(document, self.config['batch_size'], self.config, vocab=self.vocab, evaluation=True)
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

        batch.doc.set_mwt_expansions(preds)
        return batch.doc

    def bulk_process(self, docs):
        """
        MWT processor counts some statistics on the individual docs, so we need to separately redo those stats
        """
        docs = super().bulk_process(docs)
        for doc in docs:
            doc._count_words()
        return docs
