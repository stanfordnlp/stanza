import io

from stanfordnlp.models.common import conll
from stanfordnlp.models.tokenize.data import DataLoader
from stanfordnlp.models.tokenize.trainer import Trainer
from stanfordnlp.models.tokenize.utils import output_predictions
from stanfordnlp.pipeline.processor import UDProcessor
from stanfordnlp.utils.postprocess_vietnamese_tokenizer_data import paras_to_chunks


# class for running the tokenizer
class TokenizeProcessor(UDProcessor):

    def __init__(self, config, use_gpu):
        # set up configurations
        # set up trainer
        self.trainer = Trainer(model_file=config['model_path'], use_cuda=use_gpu)
        self.build_final_config(config)

    def process(self, doc):
        # set up batches
        if self.config['lang'] == 'vi':
            # special processing is due for Vietnamese
            text = '\n\n'.join([x for x in doc.text.split('\n\n')]).rstrip()
            dummy_labels = '\n\n'.join(['0' * len(x) for x in text.split('\n\n')])
            data = paras_to_chunks(text, dummy_labels)
            batches = DataLoader(self.config, input_data=data, vocab=self.vocab, evaluation=True)
        else:
            batches = DataLoader(self.config, input_text=doc.text, vocab=self.vocab, evaluation=True)
        # set up StringIO to get conllu data, run output predictions, set doc's conll file
        with io.StringIO() as conll_output_string:
            output_predictions(conll_output_string, self.trainer, batches, self.vocab, None, self.config['max_seqlen'])
            # set conll file for doc
            doc.conll_file = conll.CoNLLFile(input_str=conll_output_string.getvalue())

