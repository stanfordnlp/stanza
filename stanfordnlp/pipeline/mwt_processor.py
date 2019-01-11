import random
import torch

import stanfordnlp.models.common.seq2seq_constant as constant
from stanfordnlp.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanfordnlp.models.common import conll
from stanfordnlp.models.mwt.data import DataLoader
from stanfordnlp.models.mwt.trainer import Trainer
from stanfordnlp.models.mwt.vocab import Vocab

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
        self.trainer = Trainer(model_file=self.args['model_path'])
        self.loaded_args, self.vocab = self.trainer.args, self.trainer.vocab
        for k in self.args:
            if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand']:
                self.loaded_args[k] = self.args[k]
        self.loaded_args['cuda'] = self.args['cuda'] and not self.args['cpu']

    def process(self, doc):
        batch = DataLoader(doc, self.loaded_args['batch_size'], self.loaded_args, vocab=self.vocab, evaluation=True)
        dict_preds = self.trainer.predict_dict(batch.conll.get_mwt_expansion_cands())
        if self.loaded_args['dict_only']:
            preds = dict_preds
        else:
            print("Running the seq2seq model...")
            preds = []
            for i, b in enumerate(batch):
                preds += self.trainer.predict(b)
            if self.loaded_args.get('ensemble_dict', False):
                preds = self.trainer.ensemble(batch.conll.get_mwt_expansion_cands(), preds)
        updated_conllu = self.generate_conll_with_mwt_expansions(batch.conll, preds)
        doc.conll_file = conll.CoNLLFile(input_str=updated_conllu)

    def generate_conll_with_mwt_expansions(self, conll_file, expansions):
        idx = 0
        count = 0
        return_string = ''
        for sent in conll_file.sents:
            for ln in sent:
                idx += 1
                if "MWT=Yes" not in ln[-1]:
                    return_string += ("{}\t{}".format(idx, "\t".join(ln[1:6] + [str(idx-1)] + ln[7:])))
                    return_string += '\n'
                else:
                    # print MWT expansion
                    expanded = [x for x in expansions[count].split(' ') if len(x) > 0]
                    count += 1
                    endidx = idx + len(expanded) - 1

                    return_string += ("{}-{}\t{}".format(idx, endidx, "\t".join(['_' if i == 5 or i == 8 else x for i, x in enumerate(ln[1:])])))
                    return_string += '\n'
                    for e_i, e_word in enumerate(expanded):
                        return_string += ("{}\t{}\t{}".format(idx + e_i, e_word, "\t".join(['_'] * 4 + [str(idx + e_i - 1)] + ['_'] * 3)))
                        return_string += '\n'
                    idx = endidx

            return_string += '\n'
            idx = 0

        assert count == len(expansions), "{} {} {}".format(count, len(expansions), expansions)
        return return_string



