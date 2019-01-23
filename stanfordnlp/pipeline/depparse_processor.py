import random
import torch

from stanfordnlp.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanfordnlp.models.common import conll
from stanfordnlp.models.common.pretrain import Pretrain
from stanfordnlp.models.common.vocab import PAD_ID, VOCAB_PREFIX, ROOT_ID, CompositeVocab
from stanfordnlp.models.depparse.data import DataLoader
from stanfordnlp.models.depparse.trainer import Trainer
from stanfordnlp.models.pos.vocab import CharVocab, WordVocab, XPOSVocab, FeatureVocab, MultiVocab
from stanfordnlp.models.pos.xpos_vocab_factory import xpos_vocab_factory

DEFAULT_DEPPARSE_CONFIG = {
    'data_dir': 'data/depparse', 'wordvec_dir': 'extern_data/word2vec', 'train_file': None,
    'eval_file': 'parser_input.conllu',
    'output_file': 'parser_output.conllu',
    'gold_file': 'parser_input.conllu',
    'pretrain_path': 'saved_models/depparse/en_ewt.pretrain.pt',
    'model_path': 'saved_models/depparse/en_ewt_parser.pt',
    'mode': 'predict',
    'lang': 'en',
    'shorthand': 'en_ewt',
    'best_param': False,
    'hidden_dim': 400,
    'char_hidden_dim': 400,
    'deep_biaff_hidden_dim': 400,
    'composite_deep_biaff_hidden_dim': 100,
    'word_emb_dim': 75,
    'char_emb_dim': 100,
    'tag_emb_dim': 50,
    'transformed_dim': 125,
    'num_layers': 3,
    'char_num_layers': 1,
    'word_dropout': 0.33,
    'dropout': 0.5,
    'rec_dropout': 0,
    'char_rec_dropout': 0,
    'char': True,
    'pretrain': True,
    'linearization': True,
    'distance': True,
    'sample_train': 1.0,
    'optim': 'adam',
    'lr': 0.003,
    'beta2': 0.95,
    'max_steps': 50000,
    'eval_interval': 100,
    'max_steps_before_stop': 3000,
    'batch_size': 5000,
    'max_grad_norm': 1.0,
    'log_step': 20,
    'save_dir': 'saved_models/depparse',
    'save_name': None,
    'seed': 1234,
    'cuda': True,
    'cpu': False
}


class DepparseProcessor:

    def __init__(self, config={}):
        # set up configurations
        self.args = DEFAULT_DEPPARSE_CONFIG
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
        batch.conll.set(['head', 'deprel'], [y for x in preds for y in x])
        return batch.conll
