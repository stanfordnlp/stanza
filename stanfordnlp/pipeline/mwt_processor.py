import io

from stanfordnlp.models.common import conll
from stanfordnlp.models.mwt.data import DataLoader
from stanfordnlp.models.mwt.trainer import Trainer


DEFAULT_MWT_CONFIG = {
    'model_path': 'saved_models/mwt/fr_gsd_mwt_expander.pt',
    'cuda': True,
    'cpu': False
}


class MWTProcessor:

    def __init__(self, config={}):
        # set up configurations
        self.args = DEFAULT_MWT_CONFIG
        for key in config.keys():
            self.args[key] = config[key]
        use_cuda = self.args['cuda'] and not self.args['cpu']
        self.trainer = Trainer(model_file=self.args['model_path'], use_cuda=use_cuda)
        self.loaded_args, self.vocab = self.trainer.args, self.trainer.vocab
        for k in self.args:
            if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand']:
                self.loaded_args[k] = self.args[k]

    def process(self, doc):
        batch = DataLoader(doc, self.loaded_args['batch_size'], self.loaded_args, vocab=self.vocab, evaluation=True)
        if len(batch) > 0:
            dict_preds = self.trainer.predict_dict(batch.conll.get_mwt_expansion_cands())
            # decide trainer type and run eval
            if self.loaded_args['dict_only']:
                preds = dict_preds
            else:
                print("Running the seq2seq model...")
                preds = []
                for i, b in enumerate(batch):
                    preds += self.trainer.predict(b)

                if self.loaded_args.get('ensemble_dict', False):
                    preds = self.trainer.ensemble(batch.conll.get_mwt_expansion_cands(), preds)
        else:
            # skip eval if dev data does not exist
            preds = []

        conll_with_mwt = io.StringIO()
        batch.conll.write_conll_with_mwt_expansions(preds, conll_with_mwt, should_close=False)
        doc.conll_file = conll.CoNLLFile(input_str=conll_with_mwt.getvalue())
        conll_with_mwt.close()

