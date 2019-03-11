from stanfordnlp.models.common.pretrain import Pretrain
from stanfordnlp.models.depparse.data import DataLoader
from stanfordnlp.models.depparse.trainer import Trainer
from stanfordnlp.pipeline._constants import *
from stanfordnlp.pipeline.processor import UDProcessor


class DepparseProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([DEPPARSE])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE, POS])

    def __init__(self, config, pipeline, use_gpu):
        # set up configurations
        self._pipeline = pipeline
        # get pretrained word vectors
        self._pretrain = Pretrain(config['pretrain_path'])
        # set up trainer
        self._trainer = Trainer(pretrain=self._pretrain, model_file=config['model_path'], use_cuda=use_gpu)
        self._build_final_config(config)
        self._set_provides()
        self._set_requires()
        self._check_requirements()

    def process(self, doc):
        batch = DataLoader(
            doc, self._config['batch_size'], self._config, self._pretrain, vocab=self._vocab, evaluation=True)
        preds = []
        for i, b in enumerate(batch):
            preds += self._trainer.predict(b)
        batch.conll.set(['head', 'deprel'], [y for x in preds for y in x])
        return batch.conll

