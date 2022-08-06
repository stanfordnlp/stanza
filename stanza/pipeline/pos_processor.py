"""
Processor for performing part-of-speech tagging
"""

from stanza.models.common import doc
from stanza.models.common.utils import get_tqdm, unsort
from stanza.models.common.vocab import VOCAB_PREFIX, CompositeVocab
from stanza.models.pos.data import DataLoader
from stanza.models.pos.trainer import Trainer
from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor

tqdm = get_tqdm()

@register_processor(name=POS)
class POSProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([POS])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _set_up_model(self, config, pipeline, use_gpu):
        # get pretrained word vectors
        self._pretrain = pipeline.foundation_cache.load_pretrain(config['pretrain_path']) if 'pretrain_path' in config else None
        args = {'charlm_forward_file': config.get('forward_charlm_path', None),
                'charlm_backward_file': config.get('backward_charlm_path', None)}
        # set up trainer
        self._trainer = Trainer(pretrain=self.pretrain, model_file=config['model_path'], use_cuda=use_gpu, args=args)
        self._tqdm = 'tqdm' in config and config['tqdm']

    def __str__(self):
        return "POSProcessor(%s)" % self.config['model_path']

    def get_known_xpos(self):
        """
        Returns the xpos tags known by this model
        """
        if isinstance(self.vocab['xpos'], CompositeVocab):
            if len(self.vocab['xpos']) == 1:
                return [k for k in self.vocab['xpos'][0]._unit2id.keys() if k not in VOCAB_PREFIX]
            else:
                return {k: v.keys() - VOCAB_PREFIX for k, v in self.vocab['xpos']._unit2id.items()}
        return [k for k in self.vocab['xpos']._unit2id.keys() if k not in VOCAB_PREFIX]

    def is_composite_xpos(self):
        """
        Returns if the xpos tags are part of a composite vocab
        """
        return isinstance(self.vocab['xpos'], CompositeVocab)

    def get_known_upos(self):
        """
        Returns the upos tags known by this model
        """
        keys = [k for k in self.vocab['upos']._unit2id.keys() if k not in VOCAB_PREFIX]
        return keys

    def get_known_feats(self):
        """
        Returns the features known by this model
        """
        values = {k: v.keys() - VOCAB_PREFIX for k, v in self.vocab['feats']._unit2id.items()}
        return values

    def process(self, document):
        batch = DataLoader(
            document, self.config['batch_size'], self.config, self.pretrain, vocab=self.vocab, evaluation=True,
            sort_during_eval=True)
        preds = []

        if self._tqdm:
            for i, b in enumerate(tqdm(batch)):
                preds += self.trainer.predict(b)
        else:
            for i, b in enumerate(batch):
                preds += self.trainer.predict(b)

        preds = unsort(preds, batch.data_orig_idx)
        batch.doc.set([doc.UPOS, doc.XPOS, doc.FEATS], [y for x in preds for y in x])
        return batch.doc
