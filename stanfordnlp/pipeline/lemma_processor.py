from stanfordnlp.models.common.conll import FIELD_TO_IDX
from stanfordnlp.models.lemma.data import DataLoader
from stanfordnlp.models.lemma.trainer import Trainer
from stanfordnlp.pipeline._constants import *
from stanfordnlp.pipeline.processor import UDProcessor


class LemmaProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([LEMMA])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])

    def __init__(self, config, pipeline, use_gpu):
        # check if in identity mode
        self._pipeline = pipeline
        if config.get('use_identity') in ['True', True]:
            self._use_identity = True
            self._config = config
        else:
            self._use_identity = False
            self._trainer = Trainer(model_file=config['model_path'], use_cuda=use_gpu)
            self._build_final_config(config)
        self._set_provides()
        self._set_requires()
        self._check_requirements()

    def _set_requires(self):
        if self._config['pos']:
            self._requires = LemmaProcessor.REQUIRES_DEFAULT.union(set([POS]))
        else:
            self._requires = LemmaProcessor.REQUIRES_DEFAULT

    def process(self, doc):
        if not self._use_identity:
            batch = DataLoader(doc, self._config['batch_size'], self._config, vocab=self._vocab, evaluation=True)
        else:
            batch = DataLoader(doc, self._config['batch_size'], self._config, evaluation=True, conll_only=True)
        if self._use_identity:
            preds = [ln[FIELD_TO_IDX['word']] for sent in batch.conll.sents for ln in sent if '-' not in ln[0]]
        elif self._config.get('dict_only', False):
            preds = self._trainer.predict_dict(batch.conll.get(['word', 'upos']))
        else:
            preds = []
            edits = []
            for i, b in enumerate(batch):
                ps, es = self._trainer.predict(b, self._config['beam_size'])
                preds += ps
                if es is not None:
                    edits += es
            preds = self._trainer.postprocess(batch.conll.get(['word']), preds, edits=edits)

            if self._config.get('ensemble_dict', False):
                preds = self._trainer.ensemble(batch.conll.get(['word', 'upos']), preds)
        
        # map empty string lemmas to '_'
        preds = [max([(len(x), x), (0, '_')])[1] for x in preds]
        batch.conll.set(['lemma'], preds)

