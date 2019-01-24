import io

from stanfordnlp.models.common import conll
from stanfordnlp.models.tokenize.data import DataLoader
from stanfordnlp.models.tokenize.trainer import Trainer
from stanfordnlp.models.tokenize.utils import output_predictions
from stanfordnlp.pipeline.processor import UDProcessor

TOKENIZE_MODEL_OPTIONS = ['anneal', 'anneal_after', 'batch_size', 'conv_filters', 'conv_res', 'dropout', 'emb_dim',
                          'feat_dim', 'feat_funcs', 'hidden_dim', 'hier_invtemp', 'hierarchical', 'input_dropout',
                          'lr0', 'max_grad_norm', 'max_seqlen', 'residual', 'rnn_layers', 'seed', 'shuffle_steps',
                          'steps', 'tok_noise', 'unit_dropout', 'vocab_size', 'weight_decay']


# class for running the tokenizer
class TokenizeProcessor(UDProcessor):

    def __init__(self, config, use_gpu):
        # set up configurations
        # set up trainer
        self.model_options = TOKENIZE_MODEL_OPTIONS
        self.trainer = Trainer(model_file=config['model_path'], use_cuda=use_gpu)
        self.build_final_config(config)

    def process(self, doc):
        # set up batches
        batches = DataLoader(self.config, input_text=doc.text, vocab=self.vocab, evaluation=True)
        # set up StringIO to get conllu data, run output predictions, set doc's conll file
        with io.StringIO() as conll_output_string:
            output_predictions(conll_output_string, self.trainer, batches, self.vocab, None, self.config['max_seqlen'])
            # set conll file for doc
            doc.conll_file = conll.CoNLLFile(input_str=conll_output_string.getvalue())

