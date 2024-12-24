"""
Processor for performing lemmatization
"""

from itertools import compress

import torch

from stanza.models.common import doc
from stanza.models.lemma.data import DataLoader
from stanza.models.lemma.trainer import Trainer
from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor

WORD_TAGS = [doc.TEXT, doc.UPOS]

@register_processor(name=LEMMA)
class LemmaProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([LEMMA])
    # set of processor requirements for this processor
    # pos will be added later for non-identity lemmatizerx
    REQUIRES_DEFAULT = set([TOKENIZE])
    # default batch size
    DEFAULT_BATCH_SIZE = 5000

    def __init__(self, config, pipeline, device):
        # run lemmatizer in identity mode
        self._use_identity = None
        self._pretagged = None
        super().__init__(config, pipeline, device)

    @property
    def use_identity(self):
        return self._use_identity

    def _set_up_model(self, config, pipeline, device):
        if config.get('use_identity') in ['True', True]:
            self._use_identity = True
            self._config = config
            self.config['batch_size'] = LemmaProcessor.DEFAULT_BATCH_SIZE
        else:
            # the lemmatizer only looks at one word when making
            # decisions, not the surrounding context
            # therefore, we can save some time by remembering what
            # we did the last time we saw any given word,pos
            # since a long running program will remember everything
            # (unless we go back and make it smarter)
            # we make this an option, not the default
            # TODO: need to update the cache to skip the contextual lemmatizer
            self.store_results = config.get('store_results', False)
            self._use_identity = False
            args = {'charlm_forward_file': config.get('forward_charlm_path', None),
                    'charlm_backward_file': config.get('backward_charlm_path', None)}
            lemma_classifier_args = dict(args)
            lemma_classifier_args['wordvec_pretrain_file'] = config.get('pretrain_path', None)
            self._trainer = Trainer(args=args, model_file=config['model_path'], device=device, foundation_cache=pipeline.foundation_cache, lemma_classifier_args=lemma_classifier_args)

    def _set_up_requires(self):
        self._pretagged = self._config.get('pretagged', None)
        if self._pretagged:
            self._requires = set()
        elif self.config.get('pos') and not self.use_identity:
            self._requires = LemmaProcessor.REQUIRES_DEFAULT.union(set([POS]))
        else:
            self._requires = LemmaProcessor.REQUIRES_DEFAULT

    def process(self, document):
        if not self.use_identity:
            batch = DataLoader(document, self.config['batch_size'], self.config, vocab=self.vocab, evaluation=True, expand_unk_vocab=True)
        else:
            batch = DataLoader(document, self.config['batch_size'], self.config, evaluation=True, conll_only=True)
        if self.use_identity:
            preds = [word.text for sent in batch.doc.sentences for word in sent.words]
        elif self.config.get('dict_only', False):
            preds = self.trainer.predict_dict(batch.doc.get([doc.TEXT, doc.UPOS]))
        else:
            if self.config.get('ensemble_dict', False):
                # skip the seq2seq model when we can
                skip = self.trainer.skip_seq2seq(batch.doc.get([doc.TEXT, doc.UPOS]))
                # although there is no explicit use of caseless or lemma_caseless in this processor,
                # it shows up in the config which gets passed to the DataLoader,
                # possibly affecting its results
                seq2seq_batch = DataLoader(document, self.config['batch_size'], self.config, vocab=self.vocab,
                                           evaluation=True, skip=skip, expand_unk_vocab=True)
            else:
                seq2seq_batch = batch

            with torch.no_grad():
                preds = []
                edits = []
                for i, b in enumerate(seq2seq_batch):
                    ps, es = self.trainer.predict(b, self.config['beam_size'], seq2seq_batch.vocab)
                    preds += ps
                    if es is not None:
                        edits += es

            if self.config.get('ensemble_dict', False):
                word_tags = batch.doc.get(WORD_TAGS)
                words = [x[0] for x in word_tags]
                preds = self.trainer.postprocess([x for x, y in zip(words, skip) if not y], preds, edits=edits)
                if self.store_results:
                    new_word_tags = compress(word_tags, map(lambda x: not x, skip))
                    new_predictions = [(x[0], x[1], y) for x, y in zip(new_word_tags, preds)]
                    self.trainer.train_dict(new_predictions, update_word_dict=False)
                # expand seq2seq predictions to the same size as all words
                i = 0
                preds1 = []
                for s in skip:
                    if s:
                        preds1.append('')
                    else:
                        preds1.append(preds[i])
                        i += 1
                preds = self.trainer.ensemble(word_tags, preds1)
            else:
                preds = self.trainer.postprocess(batch.doc.get([doc.TEXT]), preds, edits=edits)

            if self.trainer.has_contextual_lemmatizers():
                preds = self.trainer.update_contextual_preds(batch.doc, preds)

        # map empty string lemmas to '_'
        preds = [max([(len(x), x), (0, '_')])[1] for x in preds]
        batch.doc.set([doc.LEMMA], preds)
        return batch.doc
