"""
Processor for performing dependency parsing
"""

from stanza.models.common import doc
from stanza.models.common.pretrain import Pretrain
from stanza.models.common.utils import unsort
from stanza.models.depparse.data import DataLoader
from stanza.models.depparse.trainer import Trainer
from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor

DEFAULT_SEPARATE_BATCH=150

@register_processor(name=DEPPARSE)
class DepparseProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([DEPPARSE])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE, POS, LEMMA])

    def __init__(self, config, pipeline, use_gpu):
        self._pretagged = None
        super().__init__(config, pipeline, use_gpu)

    def _set_up_requires(self):
        self._pretagged = self._config.get('pretagged')
        if self._pretagged:
            self._requires = set()
        else:
            self._requires = self.__class__.REQUIRES_DEFAULT

    def _set_up_model(self, config, use_gpu):
        self._pretrain = Pretrain(config['pretrain_path']) if 'pretrain_path' in config else None
        self._trainer = Trainer(pretrain=self.pretrain, model_file=config['model_path'], use_cuda=use_gpu)

    def process(self, document):
        try:
            batch = DataLoader(document, self.config['batch_size'], self.config, self.pretrain, vocab=self.vocab, evaluation=True,
                               sort_during_eval=self.config.get('sort_during_eval', True),
                               min_length_to_batch_separately=self.config.get('min_length_to_batch_separately', DEFAULT_SEPARATE_BATCH))
            preds = []
            for i, b in enumerate(batch):
                preds += self.trainer.predict(b)
            if batch.data_orig_idx is not None:
                preds = unsort(preds, batch.data_orig_idx)
            batch.doc.set([doc.HEAD, doc.DEPREL], [y for x in preds for y in x])
            # build dependencies based on predictions
            for sentence in batch.doc.sentences:
                sentence.build_dependencies()
            return batch.doc
        except RuntimeError as e:
            if str(e).startswith("CUDA out of memory. Tried to allocate"):
                new_message = str(e) + " ... You may be able to compensate for this by separating long sentences into their own batch with a parameter such as depparse_min_length_to_batch_separately=150 or by limiting the overall batch size with depparse_batch_size=400."
                raise RuntimeError(new_message) from e
            else:
                raise
