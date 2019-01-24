from stanfordnlp.models.common.pretrain import Pretrain
from stanfordnlp.models.pos.data import DataLoader
from stanfordnlp.models.pos.trainer import Trainer
from stanfordnlp.pipeline.processor import UDProcessor

POS_MODEL_OPTIONS = ['adapt_eval_interval', 'batch_size', 'beta2', 'char', 'char_emb_dim', 'char_hidden_dim',
                      'char_num_layers', 'char_rec_dropout', 'composite_deep_biaff_hidden_dim', 'deep_biaff_hidden_dim',
                      'dropout', 'eval_interval', 'hidden_dim', 'log_step', 'lr', 'max_grad_norm', 'max_steps',
                      'max_steps_before_stop', 'num_layers', 'optim', 'pretrain', 'rec_dropout', 'seed', 'share_hid',
                      'tag_emb_dim', 'transformed_dim', 'word_dropout', 'word_emb_dim']


class POSProcessor(UDProcessor):

    def __init__(self, config, use_gpu):
        # set up configurations
        # get pretrained word vectors
        self.model_options = POS_MODEL_OPTIONS
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
        batch.conll.set(['upos', 'xpos', 'feats'], [y for x in preds for y in x])


