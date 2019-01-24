from stanfordnlp.models.common.pretrain import Pretrain
from stanfordnlp.models.depparse.data import DataLoader
from stanfordnlp.models.depparse.trainer import Trainer
from stanfordnlp.pipeline.processor import UDProcessor


class DepparseProcessor(UDProcessor):

    def __init__(self, config, use_gpu):
        # set up configurations
        # get pretrained word vectors
        self.pretrain = Pretrain(config['pretrain_path'])
        # set up trainer
        self.trainer = Trainer(pretrain=self.pretrain, model_file=config['model_path'], use_cuda=use_gpu)
        self.build_final_config(config)

    def process(self, doc):
        batch = DataLoader(
            doc, self.config['batch_size'], self.config, self.pretrain, vocab=self.vocab, evaluation=True)
        preds = []
        for i, b in enumerate(batch):
            preds += self.trainer.predict(b)
        batch.conll.set(['head', 'deprel'], [y for x in preds for y in x])
        return batch.conll

