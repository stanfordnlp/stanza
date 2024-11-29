"""
Processor for performing multi-word-token expansion
"""

import io

import torch

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

    def _set_up_model(self, config, pipeline, device):
        self._trainer = Trainer(model_file=config['model_path'], device=device)

    def build_batch(self, document):
        return DataLoader(document, self.config['batch_size'], self.config, vocab=self.vocab, evaluation=True, expand_unk_vocab=True)

    def process(self, document):
        batch = self.build_batch(document)

        # process the rest
        expansions = batch.doc.get_mwt_expansions(evaluation=True)
        if len(batch) > 0:
            # decide trainer type and run eval
            if self.config['dict_only']:
                preds = self.trainer.predict_dict(expansions)
            else:
                with torch.no_grad():
                    preds = []
                    for i, b in enumerate(batch.to_loader()):
                        preds += self.trainer.predict(b, never_decode_unk=True, vocab=batch.vocab)

                if self.config.get('ensemble_dict', False):
                    preds = self.trainer.ensemble(expansions, preds)
        else:
            # skip eval if dev data does not exist
            preds = []

        batch.doc.set_mwt_expansions(preds, process_manual_expanded=False)
        return batch.doc

    def bulk_process(self, docs):
        """
        MWT processor counts some statistics on the individual docs, so we need to separately redo those stats
        """
        docs = super().bulk_process(docs)
        for doc in docs:
            doc._count_words()
        return docs
