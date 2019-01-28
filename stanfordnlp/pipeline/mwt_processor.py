import io

from stanfordnlp.models.common import conll
from stanfordnlp.models.mwt.data import DataLoader
from stanfordnlp.models.mwt.trainer import Trainer
from stanfordnlp.pipeline.processor import UDProcessor


class MWTProcessor(UDProcessor):

    def __init__(self, config, use_gpu):
        # set up configurations
        self.trainer = Trainer(model_file=config['model_path'], use_cuda=use_gpu)
        self.build_final_config(config)

    def process(self, doc):
        batch = DataLoader(doc, self.config['batch_size'], self.config, vocab=self.vocab, evaluation=True)
        if len(batch) > 0:
            dict_preds = self.trainer.predict_dict(batch.conll.get_mwt_expansion_cands())
            # decide trainer type and run eval
            if self.config['dict_only']:
                preds = dict_preds
            else:
                preds = []
                for i, b in enumerate(batch):
                    preds += self.trainer.predict(b)

                if self.config.get('ensemble_dict', False):
                    preds = self.trainer.ensemble(batch.conll.get_mwt_expansion_cands(), preds)
        else:
            # skip eval if dev data does not exist
            preds = []

        with io.StringIO() as conll_with_mwt:
            batch.conll.write_conll_with_mwt_expansions(preds, conll_with_mwt)
            doc.conll_file = conll.CoNLLFile(input_str=conll_with_mwt.getvalue())

