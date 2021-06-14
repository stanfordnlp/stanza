"""
Processor for performing named entity tagging.
"""
import logging

from stanza.models.common import doc
from stanza.models.common.utils import unsort
from stanza.models.ner.data import DataLoader
from stanza.models.ner.trainer import Trainer
from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor

logger = logging.getLogger('stanza')

@register_processor(name=NER)
class NERProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([NER])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _set_up_model(self, config, use_gpu):
        # set up trainer
        args = {'charlm_forward_file': config.get('forward_charlm_path', None),
                'charlm_backward_file': config.get('backward_charlm_path', None)}
        self._trainer = Trainer(args=args, model_file=config['model_path'], use_cuda=use_gpu)

    def process(self, document):
        # set up a eval-only data loader and skip tag preprocessing
        batch = DataLoader(
            document, self.config['batch_size'], self.config, vocab=self.vocab, evaluation=True, preprocess_tags=False)
        preds = []
        for i, b in enumerate(batch):
            preds += self.trainer.predict(b)
        batch.doc.set([doc.NER], [y for x in preds for y in x], to_token=True)
        # collect entities into document attribute
        total = len(batch.doc.build_ents())
        logger.debug(f'{total} entities found in document.')
        return batch.doc

    def bulk_process(self, docs):
        """
        NER processor has a collation step after running inference
        """
        docs = super().bulk_process(docs)
        for doc in docs:
            doc.build_ents()
        return docs
