"""Processor that attaches a constituency tree to a sentence

The model used is a generally a model trained on the Stanford
Sentiment Treebank or some similar dataset.  When run, this processor
attaches a score in the form of a string to each sentence in the
document.

TODO: a possible way to generalize this would be to make it a
ClassifierProcessor and have "sentiment" be an option.
"""

import stanza.models.constituency.trainer as trainer

from stanza.models.common import doc
from stanza.models.common.pretrain import Pretrain
from stanza.models.common.utils import get_tqdm
from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor

tqdm = get_tqdm()

@register_processor(CONSTITUENCY)
class ConstituencyProcessor(UDProcessor):
    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([CONSTITUENCY])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE, POS])

    # default batch size, measured in sentences
    DEFAULT_BATCH_SIZE = 50

    def _set_up_requires(self):
        self._pretagged = self._config.get('pretagged')
        if self._pretagged:
            self._requires = set()
        else:
            self._requires = self.__class__.REQUIRES_DEFAULT

    def _set_up_model(self, config, use_gpu):
        # get pretrained word vectors
        pretrain_path = config.get('pretrain_path', None)
        self._pretrain = Pretrain(pretrain_path) if pretrain_path else None
        # set up model
        charlm_forward_file = config.get('forward_charlm_path', None)
        charlm_backward_file = config.get('backward_charlm_path', None)
        bert_name = config.get('bert_model', None)
        bert_model, bert_tokenizer = trainer.load_bert(bert_name)
        self._model = trainer.Trainer.load(filename=config['model_path'],
                                           pt=self._pretrain,
                                           forward_charlm=trainer.load_charlm(charlm_forward_file),
                                           backward_charlm=trainer.load_charlm(charlm_backward_file),
                                           bert_model=bert_model,
                                           bert_tokenizer=bert_tokenizer,
                                           use_gpu=use_gpu)
        self._model.model.eval()
        # batch size counted as sentences
        self._batch_size = config.get('batch_size', ConstituencyProcessor.DEFAULT_BATCH_SIZE)
        self._tqdm = 'tqdm' in config and config['tqdm']

    def process(self, document):
        sentences = document.sentences

        if self._model.uses_xpos():
            words = [[(w.text, w.xpos) for w in s.words] for s in sentences]
        else:
            words = [[(w.text, w.upos) for w in s.words] for s in sentences]
        if self._tqdm:
            words = tqdm(words)

        trees = trainer.parse_tagged_words(self._model.model, words, self._batch_size)
        document.set(CONSTITUENCY, trees, to_sentence=True)
        return document
