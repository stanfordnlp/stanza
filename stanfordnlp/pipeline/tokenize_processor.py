import io

from stanfordnlp.models.common import conll
from stanfordnlp.models.tokenize.data import DataLoader
from stanfordnlp.models.tokenize.trainer import Trainer
from stanfordnlp.models.tokenize.utils import output_predictions
from stanfordnlp.pipeline.processor import Processor


DEFAULT_TOKENIZE_CONFIG = {
    'mode': 'predict',
    'shorthand': 'en_ewt',
    'lang': 'en',
    'cuda': True,
    'cpu': False,
    'max_seqlen': 1000,
    'feat_funcs': ['space_before', 'capitalized', 'all_caps', 'numeric'],
    'feat_dim': 4,
    'model_path': 'saved_models/tokenize/en_ewt_tokenizer.pt'
}


# class for running the tokenizer
class TokenizeProcessor(Processor):

    def __init__(self, config=DEFAULT_TOKENIZE_CONFIG):
        # set up configurations
        self.args = DEFAULT_TOKENIZE_CONFIG
        for key in config.keys():
            self.args[key] = config[key]
        # set up trainer
        use_cuda = config.get('cuda', True) and not config.get('cpu', False)
        self.trainer = Trainer(model_file=self.args['model_path'])
        # set configurations from loaded model
        self.loaded_args, self.vocab = self.trainer.args, self.trainer.vocab
        for k in self.loaded_args:
            if not k.endswith('_file') and k not in ['cuda', 'mode', 'save_dir', 'save_name']:
                self.args[k] = self.loaded_args[k]

    def process(self, doc):
        # set up batches
        batches = DataLoader(self.args, input_text=doc.text, vocab=self.vocab, evaluation=True)
        # set up StringIO to get conllu data, run output predictions, set doc's conll file
        with io.StringIO() as conll_output_string:
            output_predictions(conll_output_string, self.trainer, batches, self.vocab, None, self.args['max_seqlen'],
                               should_close=False)
            # set conll file for doc
            doc.conll_file = conll.CoNLLFile(input_str=conll_output_string.getvalue())

