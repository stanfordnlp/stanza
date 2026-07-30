from collections import Counter
import random
import logging
import torch

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.sampler import Sampler

from stanza.models.common.bert_embedding import filter_data, needs_length_filter
from stanza.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all, get_augment_ratio
from stanza.models.common.utils import DEFAULT_WORD_CUTOFF, simplify_punct
from stanza.models.common.vocab import PAD_ID, VOCAB_PREFIX, ROOT_ID, CompositeVocab, CharVocab
from stanza.models.pos.vocab import WordVocab, XPOSVocab, FeatureVocab, MultiVocab
from stanza.models.pos.xpos_vocab_factory import xpos_vocab_factory
from stanza.models.common.doc import *

logger = logging.getLogger('stanza')

def data_to_batches(data, batch_size, eval_mode, sort_during_eval, min_length_to_batch_separately):
    """
    Given a list of lists, where the first element of each sublist
    represents the sentence, group the sentences into batches.

    During training mode (not eval_mode) the sentences are sorted by
    length with a bit of random shuffling.  During eval mode, the
    sentences are sorted by length if sort_during_eval is true.

    Refactored from the data structure in case other models could use
    it and for ease of testing.

    Returns (batches, original_order), where original_order is None
    when in train mode or when unsorted and represents the original
    location of each sentence in the sort
    """
    res = []

    if not eval_mode:
        # sort sentences (roughly) by length for better memory utilization
        data = sorted(data, key = lambda x: len(x[0]), reverse=random.random() > .5)
        data_orig_idx = None
    elif sort_during_eval:
        (data, ), data_orig_idx = sort_all([data], [len(x[0]) for x in data])
    else:
        data_orig_idx = None

    current = []
    currentlen = 0
    for x in data:
        if min_length_to_batch_separately is not None and len(x[0]) > min_length_to_batch_separately:
            if currentlen > 0:
                res.append(current)
                current = []
                currentlen = 0
            res.append([x])
        else:
            if len(x[0]) + currentlen > batch_size and currentlen > 0:
                res.append(current)
                current = []
                currentlen = 0
            current.append(x)
            currentlen += len(x[0])

    if currentlen > 0:
        res.append(current)

    return res, data_orig_idx


def record_ends_with_punct(record, punct_id):
    """
    True if this preprocessed sentence (as produced by Dataset.preprocess)
    currently ends with a PUNCT token.

    record[2] is the ROOT-prepended upos ID field, so record[2][-1] is the
    ID of the sentence's last real word; punct_id is whatever ID the
    Dataset's upos vocab assigned to the string 'PUNCT'.  Requires at
    least one real word (len(record[2]) > 1, since index 0 is ROOT).
    """
    upos = record[2]
    return len(upos) > 1 and upos[-1] == punct_id


def record_can_augment_nopunct(record, punct_id):
    """
    True if the sentence ends with PUNCT, has at least one other real
    word, and no other word's head depends on that final PUNCT token --
    removing it would otherwise leave a word's head pointing at a
    position that no longer exists.

    record[7] is the head field (no ROOT, one 1-indexed position per
    real word), so len(record[7]) is the sentence's real word count and
    len(record[7]) itself is the last word's own 1-indexed position.
    """
    if not record_ends_with_punct(record, punct_id):
        return False
    head = record[7]
    if len(head) <= 1:
        # augment_punct's original len(sentence) > 1 guard: don't reduce
        # a sentence to zero real words
        return False
    last_position = len(head)
    if any(h == last_position for h in head[:-1]):
        return False
    return True


class Dataset:
    """
    Sentence-level dataset for the dependency parser: owns vocab
    construction and preprocessing, and exposes one sentence's raw
    (un-tensorized) ID lists per __getitem__ call.

    This mirrors the split used in stanza.models.pos.data.Dataset.
    Preprocessing (mapping words/chars/tags to vocab IDs, ROOT-token
    prepending, sentence reversal) happens once, in __init__, since
    none of it is stochastic today. But __getitem__ re-reads from
    self.data on every call rather than handing out a cached tensor,
    so that a future per-item augmentation added here (comparable to
    the POS tagger's punctuation-drop augmentation in its __getitem__)
    would be re-rolled fresh every time a batch is materialized --
    once per sentence per epoch, rather than once at preprocessing time.
    """

    def __init__(self, doc, args, pretrain, vocab=None, evaluation=False, bert_tokenizer=None):
        self.args = args
        self.eval = evaluation
        self.doc = doc
        self.reversed = args.get('reversed', False)
        data = self.load_doc(doc)

        # handle vocab
        if vocab is None:
            self.vocab = self.init_vocab(data)
        else:
            self.vocab = vocab

        # filter out the long sentences if bert is used
        if self.args.get('bert_model', None) and needs_length_filter(self.args['bert_model']):
            data = filter_data(self.args['bert_model'], data, bert_tokenizer)

        # handle pretrain; pretrain vocab is used when args['pretrain'] == True and pretrain is not None
        self.pretrain_vocab = None
        if pretrain is not None and args['pretrain']:
            self.pretrain_vocab = pretrain.vocab

        # filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)
            logger.debug("Subsample training set with rate {:g}".format(args['sample_train']))

        self.data = self.preprocess(data, self.vocab, self.pretrain_vocab, args)
        # shuffle for training -- this only affects tie-breaking order for
        # equal-length sentences the first time batches are chunked, since
        # data_to_batches re-sorts by length (with its own random direction)
        # on every call; kept here, in the same place relative to
        # preprocessing, to match the previous implementation exactly
        if not self.eval:
            random.shuffle(self.data)
        self.num_examples = len(self.data)
        # length of each preprocessed sentence (word list, ROOT-inclusive),
        # used by DepparseBatchSampler for batching without needing to
        # re-fetch every sentence's full content just to chunk by length
        self.lengths = [len(sent[0]) for sent in self.data]

        # dynamic punctuation augmentation: some UD treebanks have every
        # sentence end in PUNCT (eg UD_Hebrew-HTB), which teaches the
        # tokenizer/parser to always expect a final punctuation mark and
        # misparse sentences that don't have one. Rather than permanently
        # duplicating a fraction of sentences with their final PUNCT
        # removed (which bloats the training set and skews its
        # length/content distribution), this augmentation is applied
        # fresh, per sentence, in __getitem__ -- see record_can_augment_nopunct
        # and __getitem__ below. The target ratio itself is still computed
        # once here, the same way it always was.
        self.punct_id = self.vocab['upos'].unit2id('PUNCT') if 'PUNCT' in self.vocab['upos'] else None
        augment_nopunct_arg = args.get('augment_nopunct', None)
        if self.eval or self.punct_id is None:
            self.augment_nopunct_ratio = 0.0
        elif augment_nopunct_arg is None:
            self.augment_nopunct_ratio = get_augment_ratio(
                self.data,
                lambda record: record_ends_with_punct(record, self.punct_id),
                lambda record: record_can_augment_nopunct(record, self.punct_id))
        elif augment_nopunct_arg <= 0:
            self.augment_nopunct_ratio = 0.0
        else:
            self.augment_nopunct_ratio = augment_nopunct_arg

    def init_vocab(self, data):
        assert self.eval == False # for eval vocab must exist
        cutoff = self.args['word_cutoff'] if self.args.get('word_cutoff') is not None else DEFAULT_WORD_CUTOFF
        charvocab = CharVocab(data, self.args['shorthand'])
        wordvocab = WordVocab(data, self.args['shorthand'], cutoff=cutoff, lower=True)
        uposvocab = WordVocab(data, self.args['shorthand'], idx=1)
        xposvocab = xpos_vocab_factory(data, self.args['shorthand'])
        featsvocab = FeatureVocab(data, self.args['shorthand'], idx=3)
        lemmavocab = WordVocab(data, self.args['shorthand'], cutoff=cutoff, idx=4, lower=True)
        deprelvocab = WordVocab(data, self.args['shorthand'], idx=6)
        vocab = MultiVocab({'char': charvocab,
                            'word': wordvocab,
                            'upos': uposvocab,
                            'xpos': xposvocab,
                            'feats': featsvocab,
                            'lemma': lemmavocab,
                            'deprel': deprelvocab})
        return vocab

    def preprocess(self, data, vocab, pretrain_vocab, args):
        processed = []
        xpos_replacement = [[ROOT_ID] * len(vocab['xpos'])] if isinstance(vocab['xpos'], CompositeVocab) else [ROOT_ID]
        feats_replacement = [[ROOT_ID] * len(vocab['feats'])]
        for sent in data:
            processed_sent = [[ROOT_ID] + vocab['word'].map([w[0] for w in sent])]
            processed_sent += [[[ROOT_ID]] + [vocab['char'].map([x for x in w[0]]) for w in sent]]
            processed_sent += [[ROOT_ID] + vocab['upos'].map([w[1] for w in sent])]
            processed_sent += [xpos_replacement + vocab['xpos'].map([w[2] for w in sent])]
            processed_sent += [feats_replacement + vocab['feats'].map([w[3] for w in sent])]
            if pretrain_vocab is not None:
                # always use lowercase lookup in pretrained vocab
                processed_sent += [[ROOT_ID] + pretrain_vocab.map([w[0].lower() for w in sent])]
            else:
                processed_sent += [[ROOT_ID] + [PAD_ID] * len(sent)]
            processed_sent += [[ROOT_ID] + vocab['lemma'].map([w[4] for w in sent])]
            processed_sent += [[to_int(w[5], ignore_error=self.eval) for w in sent]]
            processed_sent += [vocab['deprel'].map([w[6] for w in sent])]
            processed_sent.append([w[0] for w in sent])
            processed.append(processed_sent)
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """
        Returns one preprocessed sentence's raw ID lists.

        Deliberately re-reads from self.data on every call (rather than,
        say, the caller caching the result across epochs) so that the
        nopunct augmentation below (and any future per-item augmentation
        added here) is re-rolled fresh every access -- the same mechanism
        the POS tagger's Dataset.__getitem__ already uses for its own
        punctuation augmentation.
        """
        record = self.data[key]
        if self.augment_nopunct_ratio > 0 and record_can_augment_nopunct(record, self.punct_id):
            if random.random() < self.augment_nopunct_ratio:
                # drop the final (PUNCT) word from every field -- ROOT-
                # prepended fields (word/char/upos/xpos/feats/pretrain/
                # lemma) and non-ROOT-prepended fields (head/deprel/text)
                # all simply lose their last entry
                record = [field[:-1] for field in record]
        return record

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def load_doc(self, doc):
        data = doc.get([TEXT, UPOS, XPOS, FEATS, LEMMA, HEAD, DEPREL], as_sentences=True)
        data = self.resolve_none(data)
        data = simplify_punct(data)
        if self.reversed:
            data = self.reverse_sentences(data)
        return data

    def reverse_sentences(self, data):
        new_data = []
        for sentence in data:
            sentence = sentence[::-1]
            for word in sentence:
                if word[5] != 0 and word[5] != '_':
                    word[5] = len(sentence) + 1 - word[5]
            new_data.append(sentence)
        return new_data

    def resolve_none(self, data):
        # replace None to '_'
        for sent_idx in range(len(data)):
            for tok_idx in range(len(data[sent_idx])):
                for feat_idx in range(len(data[sent_idx][tok_idx])):
                    if data[sent_idx][tok_idx][feat_idx] is None:
                        data[sent_idx][tok_idx][feat_idx] = '_'
        return data

    def to_loader(self, batch_size=None, sort_during_eval=True, min_length_to_batch_separately=None, sampler=None, **kwargs):
        """
        Wrap this Dataset in a real torch.utils.data.DataLoader, mirroring
        stanza.models.pos.data.Dataset.to_loader. This is the single
        place that constructs that wrapping -- DataLoader.__iter__ below
        calls this too (passing its own already-built sampler), rather
        than duplicating the TorchDataLoader construction inline.

        This is an eval-only mechanism: batches are chunked once via
        DepparseBatchSampler in eval mode (deterministic order, no random
        sort direction, no augmentation), and iterating the returned
        DataLoader repeatedly always produces the same batches in the
        same order. For training -- where batches need to be reshuffled
        on every epoch and, in the multi-source case, drawn according to
        a weighted mix -- the parent DataLoader class's InfiniteBatch-based
        mechanism is used instead, since an ordinary torch DataLoader
        doesn't directly support that.

        Pass an existing DepparseBatchSampler via sampler= to reuse
        already-computed batches (as DataLoader.__iter__ does) instead of
        recomputing them; otherwise a fresh one is built from batch_size /
        sort_during_eval / min_length_to_batch_separately.

        Passing num_workers>0 in **kwargs lets torch prefetch and collate
        batches in separate worker processes while the model consumes the
        previous batch -- likely a small win given how cheap collating a
        single batch is here, but harmless to make available.
        """
        if sampler is None:
            if batch_size is None:
                raise ValueError("Must supply either batch_size or an existing sampler")
            sampler = DepparseBatchSampler(self, batch_size, eval_mode=True,
                                            sort_during_eval=sort_during_eval,
                                            min_length_to_batch_separately=min_length_to_batch_separately)
        return TorchDataLoader(self, batch_sampler=sampler, collate_fn=DataLoader.collate, **kwargs)


class DepparseBatchSampler(Sampler):
    """
    Batches Dataset indices using data_to_batches' existing token-budget
    chunking logic, so that DataLoader can fetch each sentence fresh from
    Dataset (via Dataset.__getitem__) rather than working from a single
    fully pre-tensorized copy computed once at construction time.

    Batches are computed once at construction (mirroring the previous
    DataLoader.__init__ behavior). Call reshuffle() to recompute them
    with a fresh sort/shuffle for a new training epoch -- this is a
    separate, explicit step rather than something that happens
    automatically on iteration, matching the previous implementation
    (where a DataLoader's caller, such as InfiniteBatch, decides when
    a new epoch's worth of batches should be built).
    """

    def __init__(self, dataset, batch_size, eval_mode, sort_during_eval, min_length_to_batch_separately):
        self.dataset = dataset
        self.batch_size = batch_size
        self.eval_mode = eval_mode
        self.sort_during_eval = sort_during_eval
        self.min_length_to_batch_separately = min_length_to_batch_separately
        self.batches, self.data_orig_idx = self._compute_batches()

    def _compute_batches(self):
        # data_to_batches only needs len(x[0]) per item to decide how to
        # chunk; hand it a (length-placeholder, dataset-index) proxy so
        # the existing, already-tested chunking function can be reused
        # unmodified, without needing to fetch every sentence's full
        # content just to compute batch membership
        proxy = [([None] * length, idx) for idx, length in enumerate(self.dataset.lengths)]
        chunked, data_orig_idx = data_to_batches(
            data=proxy, batch_size=self.batch_size, eval_mode=self.eval_mode,
            sort_during_eval=self.sort_during_eval,
            min_length_to_batch_separately=self.min_length_to_batch_separately)
        batches = [[item[1] for item in chunk] for chunk in chunked]
        return batches, data_orig_idx

    def reshuffle(self):
        """Recompute batches with a fresh shuffle/sort for a new epoch."""
        self.batches, self.data_orig_idx = self._compute_batches()
        random.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch


class DataLoader:
    """
    Batch-level wrapper around Dataset + DepparseBatchSampler.

    The external API (constructor signature, __len__, __getitem__,
    __iter__, reshuffle, set_batch_size, and the vocab/doc/data_orig_idx
    attributes) is unchanged from the previous single-class
    implementation, so existing callers (InfiniteBatch, the parser
    training loop, the depparse pipeline processor) do not need to
    change. Internally, each batch is now materialized by fetching its
    sentences fresh from self.dataset (see Dataset.__getitem__) rather
    than reading from a single fully-preprocessed copy computed once at
    construction time.
    """

    def __init__(self, doc, batch_size, args, pretrain, vocab=None, evaluation=False, sort_during_eval=False, min_length_to_batch_separately=None, bert_tokenizer=None):
        self.batch_size = batch_size
        self.min_length_to_batch_separately = min_length_to_batch_separately
        self.args = args
        self.eval = evaluation
        self.sort_during_eval = sort_during_eval

        self.dataset = Dataset(doc, args, pretrain, vocab=vocab, evaluation=evaluation, bert_tokenizer=bert_tokenizer)
        self.doc = self.dataset.doc
        self.vocab = self.dataset.vocab

        self.sampler = DepparseBatchSampler(self.dataset, batch_size, evaluation, sort_during_eval, min_length_to_batch_separately)
        # data_orig_idx might be None at train time, since we don't anticipate unsorting
        self.data_orig_idx = self.sampler.data_orig_idx
        logger.debug("{} batches created.".format(len(self.sampler)))

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.sampler):
            raise IndexError
        index_list = self.sampler.batches[key]
        # fetch each sentence fresh from the Dataset -- see Dataset.__getitem__
        batch = [self.dataset[idx] for idx in index_list]
        return self.collate(batch)

    @staticmethod
    def collate(batch):
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 10

        # sort sentences by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # sort words by lens for easy char-RNN operations
        batch_words = [w for sent in batch[1] for w in sent]
        word_lens = [len(x) for x in batch_words]
        batch_words, word_orig_idx = sort_all([batch_words], word_lens)
        batch_words = batch_words[0]
        word_lens = [len(x) for x in batch_words]

        # convert to tensors
        words = batch[0]
        words = get_long_tensor(words, batch_size)
        words_mask = torch.eq(words, PAD_ID)
        wordchars = get_long_tensor(batch_words, len(word_lens))
        wordchars_mask = torch.eq(wordchars, PAD_ID)

        upos = get_long_tensor(batch[2], batch_size)
        xpos = get_long_tensor(batch[3], batch_size)
        ufeats = get_long_tensor(batch[4], batch_size)
        pretrained = get_long_tensor(batch[5], batch_size)
        sentlens = [len(x) for x in batch[0]]
        lemma = get_long_tensor(batch[6], batch_size)
        head = get_long_tensor(batch[7], batch_size)
        deprel = get_long_tensor(batch[8], batch_size)
        text = batch[9]
        return words, words_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, orig_idx, word_orig_idx, sentlens, word_lens, text

    def __iter__(self):
        """
        Eval mode iterates via a real torch.utils.data.DataLoader (built
        from the same sampler and collate function used everywhere else
        here), matching stanza.models.pos.data.Dataset.to_loader's
        mechanism and allowing num_workers>0 to prefetch/collate batches
        in a separate process, if a caller wants that.

        Train mode keeps the previous manual loop: InfiniteBatch relies on
        reshuffle()-on-exhaustion and (for multi-source training) weighted
        draws across sources, neither of which an ordinary torch
        DataLoader directly supports, so there is no equivalent win from
        switching that path over.
        """
        if self.eval:
            loader = self.dataset.to_loader(sampler=self.sampler)
            for batch in loader:
                yield batch
        else:
            for i in range(self.__len__()):
                yield self.__getitem__(i)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.sampler.batch_size = batch_size

    def reshuffle(self):
        self.sampler.reshuffle()
        self.data_orig_idx = self.sampler.data_orig_idx

def to_int(string, ignore_error=False):
    try:
        res = int(string)
    except ValueError as err:
        if ignore_error:
            return 0
        else:
            raise err
    return res

class InfiniteBatch:
    def __init__(self, *batches, weights=None):
        self.batches = batches
        self.iterators = [iter(batch) for batch in self.batches]
        if weights is None:
            self.weights = [1.0 for _ in batches]
        else:
            assert len(weights) == len(batches), "Got a weights parameter of a different length from the batches parameter"
            self.weights = weights

        self.counts = Counter()

    def next_batch(self):
        if len(self.batches) == 1:
            batch_idx = 0
        else:
            batch_idx = random.choices(range(len(self.batches)), self.weights)[0]
        self.counts[batch_idx] += 1
        batch = next(self.iterators[batch_idx], None)
        if batch is None:
            self.batches[batch_idx].reshuffle()
            self.iterators[batch_idx] = iter(self.batches[batch_idx])
            batch = next(self.iterators[batch_idx])
        return batch

    def set_batch_size(self, batch_size):
        for batch in self.batches:
            batch.set_batch_size(batch_size)

    def reshuffle(self):
        for batch in self.batches:
            batch.reshuffle()
        self.iterators = [iter(batch) for batch in self.batches]
