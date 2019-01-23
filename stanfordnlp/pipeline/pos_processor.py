from stanfordnlp.models.common.pretrain import Pretrain
from stanfordnlp.models.pos.data import DataLoader
from stanfordnlp.models.pos.trainer import Trainer
from stanfordnlp.pipeline.processor import Processor

DEFAULT_POS_CONFIG = {
                      'data_dir': 'data/pos',
                      'wordvec_dir': 'extern_data/word2vec',
                      'train_file': None,
                      'eval_file': 'pre_pos_content.conllu',
                      'output_file': 'post_pos_content.conllu',
                      'gold_file': 'pre_pos_content.conllu',
                      'pretrain_path': 'saved_models/pos/en_ewt.pretrain.pt',
                      'model_path': 'saved_models/pos/en_ewt_tagger.pt',
                      'mode': 'predict',
                      'lang': 'en_ewt',
                      'shorthand': 'en_ewt',
                      'best_param': False,
                      'char': True,
                      'pretrain': True,
                      'share_hid': False,
                      'sample_train': 1.0,
                      'optim': 'adam',
                      'lr': 0.003,
                      'beta2': 0.95,
                      'max_steps': 50000,
                      'eval_interval': 100,
                      'adapt_eval_interval': True,
                      'max_steps_before_stop': 3000,
                      'batch_size': 5000,
                      'max_grad_norm': 1.0,
                      'log_step': 20,
                      'save_dir': 'saved_models/pos',
                      'save_name': None,
                      'seed': 1234,
                      'cuda': True,
                      'cpu': False
                      }


class POSProcessor(Processor):

    def __init__(self, config={}):
        # set up configurations
        self.args = DEFAULT_POS_CONFIG
        for key in config.keys():
            self.args[key] = config[key]
        # get pretrained word vectors
        self.pretrain = Pretrain(self.args['pretrain_path'])
        # set up trainer
        self.trainer = Trainer(pretrain=self.pretrain, model_file=self.args['model_path'])
        self.loaded_args, self.vocab = self.trainer.args, self.trainer.vocab
        for k in self.args:
            if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand'] or k == 'mode':
                self.loaded_args[k] = self.args[k]
        self.loaded_args['cuda'] = self.args['cuda'] and not self.args['cpu']

    def process(self, doc):
        batch = DataLoader(
            doc, self.loaded_args['batch_size'], self.loaded_args, self.pretrain, vocab=self.vocab, evaluation=True)
        preds = []
        for i, b in enumerate(batch):
            preds += self.trainer.predict(b)
        batch.conll.set(['upos', 'xpos', 'feats'], [y for x in preds for y in x])


