"""
Processor that attaches a constituency tree to a sentence
"""

from stanza.models.constituency.trainer import Trainer

from stanza.models.common import doc
from stanza.models.common.utils import sort_with_indices, unsort
from stanza.utils.get_tqdm import get_tqdm
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

    def _set_up_model(self, config, pipeline, device):
        # set up model
        # pretrain and charlm paths are args from the config
        # bert (if used) will be chosen from the model save file
        args = {
            "wordvec_pretrain_file": config.get('pretrain_path', None),
            "charlm_forward_file": config.get('forward_charlm_path', None),
            "charlm_backward_file": config.get('backward_charlm_path', None),
            "device": device,
        }
        trainer = Trainer.load(filename=config['model_path'],
                               args=args,
                               foundation_cache=pipeline.foundation_cache)
        self._trainer = trainer
        self._model = trainer.model
        self._model.eval()
        # batch size counted as sentences
        self._batch_size = int(config.get('batch_size', ConstituencyProcessor.DEFAULT_BATCH_SIZE))
        self._tqdm = 'tqdm' in config and config['tqdm']

    def _set_up_final_config(self, config):
        loaded_args = self._model.args
        loaded_args = {k: v for k, v in loaded_args.items() if not UDProcessor.filter_out_option(k)}
        loaded_args.update(config)
        self._config = loaded_args

    def process(self, document):
        sentences = document.sentences

        if self._model.uses_xpos():
            words = [[(w.text, w.xpos) for w in s.words] for s in sentences]
        else:
            words = [[(w.text, w.upos) for w in s.words] for s in sentences]
        words, original_indices = sort_with_indices(words, key=len, reverse=True)
        if self._tqdm:
            words = tqdm(words)

        trees = self._model.parse_tagged_words(words, self._batch_size)
        trees = unsort(trees, original_indices)
        document.set(CONSTITUENCY, trees, to_sentence=True)
        return document

    def get_constituents(self):
        """
        Return a set of the constituents known by this model

        For a pipeline, this can be queried with
          pipeline.processors["constituency"].get_constituents()
        """
        return set(self._model.constituents)
