from stanfordnlp.models.common.conll import FIELD_TO_IDX
from stanfordnlp.models.lemma.data import DataLoader
from stanfordnlp.models.lemma.trainer import Trainer
from stanfordnlp.pipeline.processor import UDProcessor

LEMMA_MODEL_OPTIONS = ['alpha', 'attn_type', 'batch_size', 'beam_size', 'decay_epoch', 'dict_only', 'dropout',
                        'edit', 'emb_dim', 'emb_dropout', 'ensemble_dict', 'hidden_dim', 'log_step', 'lr', 'lr_decay',
                        'max_dec_len', 'max_grad_norm', 'num_edit', 'num_epoch', 'num_layers', 'optim', 'pos',
                        'pos_dim', 'pos_dropout', 'pos_vocab_size', 'seed', 'vocab_size']


class LemmaProcessor(UDProcessor):

    def __init__(self, config, use_gpu):
        self.model_options = LEMMA_MODEL_OPTIONS
        # check if in identity mode
        if config.get('use_identity') in ['True', True]:
            self.use_identity = True
            self.config = config
        else:
            self.use_identity = False
            self.trainer = Trainer(model_file=config['model_path'], use_cuda=use_gpu)
            self.build_final_config(config)

    def process(self, doc):
        if not self.use_identity:
            batch = DataLoader(doc, self.config['batch_size'], self.config, vocab=self.vocab, evaluation=True)
        else:
            batch = DataLoader(doc, self.config['batch_size'], self.config, evaluation=True, conll_only=True)
        if self.use_identity:
            preds = [ln[FIELD_TO_IDX['word']] for sent in batch.conll.sents for ln in sent if '-' not in ln[0]]
        elif self.config.get('dict_only', False):
            preds = self.trainer.predict_dict(batch.conll.get(['word', 'upos']))
        else:
            print("Running the seq2seq model...")
            preds = []
            edits = []
            for i, b in enumerate(batch):
                ps, es = self.trainer.predict(b, self.config['beam_size'])
                preds += ps
                if es is not None:
                    edits += es
            preds = self.trainer.postprocess(batch.conll.get(['word']), preds, edits=edits)

            if self.config.get('ensemble_dict', False):
                print("[Ensembling dict with seq2seq lemmatizer...]")
                preds = self.trainer.ensemble(batch.conll.get(['word', 'upos']), preds)
        
        # map empty string lemmas to '_'
        preds = [max([(len(x),x), (0, '_')])[1] for x in preds]
        batch.conll.set(['lemma'], preds)

