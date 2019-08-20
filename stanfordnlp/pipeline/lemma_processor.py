"""
Processor for performing lemmatization
"""

from stanfordnlp.models.lemma.data import DataLoader
from stanfordnlp.models.lemma.trainer import Trainer
from stanfordnlp.pipeline._constants import *
from stanfordnlp.pipeline.processor import UDProcessor


class LemmaProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([LEMMA])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])
    # default batch size
    DEFAULT_BATCH_SIZE = 5000

    def __init__(self, config, pipeline, use_gpu, logger=None):
        # run lemmatizer in identity mode
        self._use_identity = None
        super().__init__(config, pipeline, use_gpu, logger=logger)

    @property
    def use_identity(self):
        return self._use_identity

    def _set_up_model(self, config, use_gpu):
        if config.get('use_identity') in ['True', True]:
            self._use_identity = True
            self._config = config
            self.config['batch_size'] = LemmaProcessor.DEFAULT_BATCH_SIZE
        else:
            self._use_identity = False
            self._trainer = Trainer(model_file=config['model_path'], use_cuda=use_gpu, logger=self.logger)

    def _set_up_requires(self):
        if self.config.get('pos') and not self.use_identity:
            self._requires = LemmaProcessor.REQUIRES_DEFAULT.union(set([POS]))
        else:
            self._requires = LemmaProcessor.REQUIRES_DEFAULT

    def process(self, doc):
        if not self.use_identity:
            batch = DataLoader(doc, self.config['batch_size'], self.config, vocab=self.vocab, evaluation=True)
        else:
            batch = DataLoader(doc, self.config['batch_size'], self.config, evaluation=True, conll_only=True)
        if self.use_identity:
            preds = [word.text for sent in batch.doc.sentences for word in sent.words]
        elif self.config.get('dict_only', False):
            preds = self.trainer.predict_dict(batch.doc.get(['text', 'upos']))
        else:
            if self.config.get('ensemble_dict', False):
                # skip the seq2seq model when we can
                skip = self.trainer.skip_seq2seq(batch.doc.get(['text', 'upos']))
                seq2seq_batch = DataLoader(doc, self.config['batch_size'], self.config, vocab=self.vocab,
                                           evaluation=True, skip=skip)
            else:
                seq2seq_batch = batch

            preds = []
            edits = []
            for i, b in enumerate(seq2seq_batch):
                ps, es = self.trainer.predict(b, self.config['beam_size'])
                preds += ps
                if es is not None:
                    edits += es

            if self.config.get('ensemble_dict', False):
                preds = self.trainer.postprocess([x for x, y in zip(batch.doc.get(['text']), skip) if not y], preds, edits=edits)
                # expand seq2seq predictions to the same size as all words
                i = 0
                preds1 = []
                for s in skip:
                    if s:
                        preds1.append('')
                    else:
                        preds1.append(preds[i])
                        i += 1
                preds = self.trainer.ensemble(batch.doc.get(['text', 'upos']), preds1)
            else:
                preds = self.trainer.postprocess(batch.doc.get(['text']), preds, edits=edits)

        # map empty string lemmas to '_'
        preds = [max([(len(x), x), (0, '_')])[1] for x in preds]
        batch.doc.set(['lemma'], preds)
        return batch.doc

