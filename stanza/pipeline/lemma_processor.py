"""
Processor for performing lemmatization
"""

from collections import OrderedDict
from itertools import compress

import torch

from stanza.models.common import doc
from stanza.models.lemma.data import DataLoader
from stanza.models.lemma.trainer import Trainer
from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor

WORD_TAGS = [doc.TEXT, doc.UPOS]

#: How many word, pos answers the run time cache keeps before evicting the least
#: recently used. Bounded so that a long running process cannot grow without limit,
#: which is what kept store_results off by default.
DEFAULT_STORE_RESULTS_SIZE = 10000

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
            # the cache skips the words the contextual lemmatizers decide,
            # see Trainer.drop_contextual
            self.store_results = config.get('store_results', True)
            self.store_results_size = int(config.get('store_results_size', DEFAULT_STORE_RESULTS_SIZE))
            # (word, pos) -> lemma, least recently used first. Kept here rather than in
            # the trainer's pos_dict: that dict is the dictionary the model shipped with,
            # so evicting from it would throw away model data, not cached answers.
            self._results = OrderedDict()
            self._use_identity = False
            args = {'charlm_forward_file': config.get('forward_charlm_path', None),
                    'charlm_backward_file': config.get('backward_charlm_path', None)}
            lemma_classifier_args = dict(args)
            lemma_classifier_args['wordvec_pretrain_file'] = config.get('pretrain_path', None)
            self._trainer = Trainer(args=args, model_file=config['model_path'], device=device, foundation_cache=pipeline.foundation_cache, lemma_classifier_args=lemma_classifier_args)

    def _remember(self, triples):
        """Keep (word, pos, lemma) answers, dropping the least recently used past the cap."""
        for word, pos, lemma in triples:
            key = (word, pos)
            self._results.pop(key, None)
            self._results[key] = lemma
        while len(self._results) > self.store_results_size:
            self._results.popitem(last=False)

    def _recall(self, word, pos):
        """A remembered lemma for this word and pos, or None; marks it recently used."""
        key = (word, pos)
        if key not in self._results:
            return None
        self._results.move_to_end(key)
        return self._results[key]

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
                word_tags_for_skip = batch.doc.get([doc.TEXT, doc.UPOS])
                skip = self.trainer.skip_seq2seq(word_tags_for_skip)
                if self.store_results:
                    # a remembered answer is as good as a dictionary hit for skipping
                    skip = [
                        was_skipped or self._recall(word, pos) is not None
                        for was_skipped, (word, pos) in zip(skip, word_tags_for_skip)
                    ]
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
                    # these predictions have not been through the contextual
                    # lemmatizers yet, which happens below, so the words those
                    # decide must not be remembered from one sentence
                    new_predictions = self.trainer.drop_contextual(new_predictions)
                    self._remember(new_predictions)
                # expand seq2seq predictions to the same size as all words
                i = 0
                preds1 = []
                for s in skip:
                    if s:
                        preds1.append('')
                    else:
                        preds1.append(preds[i])
                        i += 1
                if self.store_results:
                    # ensemble prefers the shipped dictionary, then whatever is passed
                    # here, so a remembered answer fills the slot of a word we skipped
                    preds1 = [
                        pred if pred else (self._recall(word, pos) or pred)
                        for pred, (word, pos) in zip(preds1, word_tags)
                    ]
                preds = self.trainer.ensemble(word_tags, preds1)
            else:
                preds = self.trainer.postprocess(batch.doc.get([doc.TEXT]), preds, edits=edits)

            if self.trainer.has_contextual_lemmatizers():
                preds = self.trainer.update_contextual_preds(batch.doc, preds)

        # map empty string lemmas to '_'
        preds = [max([(len(x), x), (0, '_')])[1] for x in preds]
        batch.doc.set([doc.LEMMA], preds)
        return batch.doc
