import random
import logging
import copy
import torch
from collections import namedtuple

from torch.utils.data import DataLoader as DL
from torch.utils.data.sampler import Sampler
from torch.nn.utils.rnn import pad_sequence

from stanza.models.common.bert_embedding import filter_data, needs_length_filter
from stanza.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all, starts_with_initial_mark
from stanza.models.common.utils import DEFAULT_WORD_CUTOFF, simplify_punct
from stanza.models.common.vocab import PAD_ID, VOCAB_PREFIX, CharVocab
from stanza.models.pos.tag_columns import DEFAULT_TAG_COLUMNS, TagKind, extract_misc_value, tag_columns_from_args
from stanza.models.pos.vocab import WordVocab, XPOSVocab, FeatureVocab, MultiVocab
from stanza.models.pos.xpos_vocab_factory import xpos_vocab_factory
from stanza.models.common.doc import *

logger = logging.getLogger('stanza')

# tags is a list with one entry per TagColumn, in column order.  An
# entry is None when this dataset has no annotation for that column at
# all - the loss for that head is then skipped for the whole batch,
# which is why ShuffledDataset keeps each batch to a single dataset.
DataSample = namedtuple("DataSample", "word char tags pretrain text")
DataBatch = namedtuple("DataBatch", "words words_mask wordchars wordchars_mask tags pretrained orig_idx word_orig_idx lens word_lens text idx")

def build_tag_vocab(column, data, idx, shorthand):
    """Build the vocab for one tag column, at position idx of the per-word lists"""
    if column.kind is TagKind.WORD:
        return WordVocab(data, shorthand, idx=idx)
    if column.kind is TagKind.FEATURES:
        try:
            return FeatureVocab(data, shorthand, idx=idx)
        except ValueError as e:
            raise ValueError("Unable to build the '%s' vocab.  Please check that column of your data for an error which may match the following description." % column.name) from e
    if column.kind is TagKind.AUTO:
        return xpos_vocab_factory(data, shorthand, idx=idx, column_name=column.name)
    raise ValueError("Unhandled tag column kind %s for column %s" % (column.kind, column.name))

class Dataset:
    def __init__(self, doc, args, pretrain, vocab=None, evaluation=False, sort_during_eval=False, bert_tokenizer=None, tag_columns=None, **kwargs):
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.sort_during_eval = sort_during_eval
        self.doc = doc

        if tag_columns is None:
            tag_columns = tag_columns_from_args(args)
        self.tag_columns = tuple(tag_columns)
        self.tag_names = [x.name for x in self.tag_columns]

        if vocab is None:
            self.vocab = Dataset.init_vocab([doc], args, self.tag_columns)
        else:
            self.vocab = vocab

        # has_tags[name] is False when the column is entirely absent
        # from this document, in which case the head is not trained on
        # this dataset's batches
        raw = self.load_doc(self.doc, self.tag_columns)
        self.has_tags = {name: False for name in self.tag_names}
        for sentence in raw:
            for word in sentence:
                for idx, name in enumerate(self.tag_names, 1):
                    if not self.has_tags[name] and word[idx] is not None and word[idx] != '_':
                        self.has_tags[name] = True

        data = raw
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

        # dynamic leading-¿/¡ drop: some UD treebanks (Spanish, Catalan) have
        # every training sentence with a leading inverted question or
        # exclamation mark, which the model never learns to do without.
        #
        # Eligibility is checked against the raw sentence data (before
        # self.vocab['word'] is even queried), not by asking whether ¿/¡
        # is IN self.vocab['word'].  self.vocab['word'] is built with a
        # frequency cutoff (DEFAULT_WORD_CUTOFF, or word_cutoff if set) --
        # a word appearing fewer times than the cutoff is left out of the
        # vocab and maps to UNK instead.  In a small treebank, ¿/¡ could
        # easily appear only a handful of times and fall under that
        # cutoff, which would make a vocab-containment check wrongly say
        # "ineligible" even though the mark is genuinely present in the
        # data.  Scanning the raw sentences directly avoids that failure
        # mode, so the augmentation still triggers correctly even on
        # small treebanks.  starts_with_initial_mark is shared with the
        # dependency parser's equivalent eligibility check.
        self.drop_initial_punct_eligible = not self.eval and any(
            starts_with_initial_mark([w[0] for w in sent]) for sent in data)
        self.drop_initial_punct_ratio = args.get('drop_initial_punct_prob', 0.20) if self.drop_initial_punct_eligible else 0.0

        data = self.preprocess(data, self.vocab, self.pretrain_vocab, args)

        self.data = data

        self.num_examples = len(data)
        self.__punct_tags = self.vocab["upos"].map(["PUNCT"])
        self.augment_nopunct = self.args.get("augment_nopunct", 0.0)

    # The three native columns are addressed by name often enough,
    # both here and in tagger.py, that it is worth keeping the old
    # attribute names working.  Anything beyond them goes through
    # has_tags directly.
    @property
    def has_upos(self):
        return self.has_tags['upos']

    @has_upos.setter
    def has_upos(self, value):
        self.has_tags['upos'] = value

    @property
    def has_xpos(self):
        return self.has_tags['xpos']

    @has_xpos.setter
    def has_xpos(self, value):
        self.has_tags['xpos'] = value

    @property
    def has_feats(self):
        return self.has_tags['feats']

    @has_feats.setter
    def has_feats(self, value):
        self.has_tags['feats'] = value

    @staticmethod
    def init_vocab(docs, args, tag_columns=None):
        if tag_columns is None:
            tag_columns = tag_columns_from_args(args)
        cutoff = args['word_cutoff'] if args.get('word_cutoff') is not None else DEFAULT_WORD_CUTOFF
        data = [x for doc in docs for x in Dataset.load_doc(doc, tag_columns)]
        charvocab = CharVocab(data, args['shorthand'])
        wordvocab = WordVocab(data, args['shorthand'], cutoff=cutoff, lower=True)
        vocabs = {'char': charvocab,
                  'word': wordvocab}
        for idx, column in enumerate(tag_columns, 1):
            # a column named 'char' or 'word', or two columns with the
            # same name, would otherwise replace an entry already in
            # here and leave the model looking words up in a tagset
            if column.name in vocabs:
                raise ValueError("Cannot build a vocab for the tag column '%s': that name is already used by another vocab" % column.name)
            vocabs[column.name] = build_tag_vocab(column, data, idx, args['shorthand'])
        return MultiVocab(vocabs)

    def preprocess(self, data, vocab, pretrain_vocab, args):
        processed = []
        for sent in data:
            processed_sent = DataSample(
                word = [vocab['word'].map([w[0] for w in sent])],
                char = [[vocab['char'].map([x for x in w[0]]) for w in sent]],
                tags = [vocab[name].map([w[idx] for w in sent])
                        for idx, name in enumerate(self.tag_names, 1)],
                pretrain = ([pretrain_vocab.map([w[0].lower() for w in sent])]
                            if pretrain_vocab is not None
                           else [[PAD_ID] * len(sent)]),
                text = [w[0] for w in sent]
            )
            processed.append(processed_sent)

        return processed

    def __len__(self):
        return len(self.data)

    def __mask(self, upos):
        """Returns a torch boolean about which elements should be masked out"""

        # creates all false mask
        mask = torch.zeros_like(upos, dtype=torch.bool)

        ### augmentation 1: punctuation augmentation ###
        # tags that needs to be checked, currently only PUNCT
        if random.uniform(0,1) < self.augment_nopunct:
            for i in self.__punct_tags:
                # generate a mask for the last element
                last_element = torch.zeros_like(upos, dtype=torch.bool)
                last_element[..., -1] = True
                # we or the bitmask against the existing mask
                # if it satisfies, we remove the word by masking it
                # to true
                #
                # if your input is just a lone punctuation, we perform
                # no masking
                if not torch.all(upos.eq(torch.tensor([[i]]))):
                    mask |= ((upos == i) & (last_element))

        return mask

    def __getitem__(self, key):
        """Retrieves a sample from the dataset.

        Retrieves a sample from the dataset. This function, for the
        most part, is spent performing ad-hoc data augmentation and
        restoration. It receives a DataSample object from the storage,
        and returns an almost-identical DataSample object that may
        have been augmented with /possibly/ (depending on augment_punct
        settings) PUNCT chopped.

        **Important Note**
        ------------------
        If you would like to load the data into a model, please convert
        this Dataset object into a DataLoader via self.to_loader(). Then,
        you can use the resulting object like any other PyTorch data
        loader. As masks are calculated ad-hoc given the batch, the samples
        returned from this object doesn't have the appropriate masking.

        Motivation
        ----------
        Why is this here? Every time you call next(iter(dataloader)), it calls
        this function. Therefore, if we augmented each sample on each iteration,
        the model will see dynamically generated augmentation.
        Furthermore, PyTorch dataloader handles shuffling natively.

        Parameters
        ----------
        key : int
            the integer ID to from which to retrieve the key.

        Returns
        -------
        DataSample
            The sample of data you requested, with augmentation.
        """
        # get a sample of the input data
        sample = self.data[key]

        # some data augmentation requires constructing a mask based on upos.
        # For instance, sometimes we'd like to mask out ending sentence punctuation.
        # We copy the other items here so that any edits made because
        # of the mask don't clobber the version owned by the Dataset
        # convert to tensors
        # TODO: only store single lists per data entry?
        words = torch.tensor(sample.word[0])
        # convert the rest to tensors.  a column this dataset has no
        # annotation for stays None the whole way through, so that
        # collate can turn the whole batch's entry into None and the
        # model can skip that head
        tags = [torch.tensor(tag) if self.has_tags[name] else None
                for name, tag in zip(self.tag_names, sample.tags)]
        pretrained = torch.tensor(sample.pretrain[0])

        # and deal with char & raw_text
        char = sample.char[0]
        raw_text = sample.text

        # some data augmentation requires constructing a mask based on
        # which upos. For instance, sometimes we'd like to mask out ending
        # sentence punctuation. The mask is True if we want to remove the element
        upos = tags[0]
        if self.has_tags['upos'] and upos is not None and not self.eval:
            # perform actual masking
            mask = self.__mask(upos)
        else:
            # dummy mask that's all false
            mask = None
        if mask is not None:
            mask_index = mask.nonzero()

            # mask out the elements that we need to mask out
            for mask in mask_index:
                mask = mask.item()
                words[mask] = PAD_ID
                for tag in tags:
                    if tag is not None:
                        # works for a flat column and for a composite
                        # one, where the tensor has a second dimension
                        tag[mask, ...] = PAD_ID
                pretrained[mask] = PAD_ID
                char = char[:mask] + char[mask+1:]
                raw_text = raw_text[:mask] + raw_text[mask+1:]

        # dynamic leading-¿/¡ drop (see drop_initial_punct_eligible in
        # __init__).  Unlike the trailing-punct mask above, this can't be
        # done by masking a position in place: removing the FIRST word
        # has to shift every later position back by one, so every field
        # is sliced consistently rather than one element being replaced
        # with a placeholder while the rest stay put.
        if (self.drop_initial_punct_ratio > 0 and starts_with_initial_mark(raw_text)
                and random.uniform(0, 1) < self.drop_initial_punct_ratio):
            words = words[1:]
            tags = [tag[1:] if tag is not None else None for tag in tags]
            pretrained = pretrained[1:]
            char = char[1:]
            raw_text = raw_text[1:]

        # get each character from the input sentnece
        # chars = [w for sent in char for w in sent]

        return DataSample(words, char, tags, pretrained, raw_text), key

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def to_loader(self, **kwargs):
        """Converts self to a DataLoader """

        return DL(self,
                  collate_fn=self.collate_fn,
                  **kwargs)

    def to_length_limited_loader(self, batch_size, maximum_tokens):
        sampler = LengthLimitedBatchSampler(self, batch_size, maximum_tokens)
        return DL(self,
                  collate_fn=self.collate_fn,
                  batch_sampler = sampler)

    def collate_fn(self, data):
        """Function used by DataLoader to pack data

        An instance method, as the collation needs to know how many
        entries to expect in each sample's tags.
        """
        (data, idx) = zip(*data)
        (words, wordchars, tags, pretrained, text) = zip(*data)

        # tags arrives as one tuple per sample; flip it to one tuple
        # per column, which is the shape everything downstream wants
        tags = list(zip(*tags)) if len(self.tag_names) > 0 else []

        # collate_fn is given a list of length batch size
        batch_size = len(data)

        # sort sentences by lens for easy RNN operations
        lens = [torch.sum(x != PAD_ID) for x in words]
        to_sort = [words, wordchars] + list(tags) + [pretrained, text]
        sorted_all, orig_idx = sort_all(to_sort, lens)
        words, wordchars = sorted_all[0], sorted_all[1]
        tags = sorted_all[2:2+len(tags)]
        pretrained, text = sorted_all[-2], sorted_all[-1]
        lens = [torch.sum(x != PAD_ID) for x in words] # we need to reinterpret lengths for the RNN

        # combine all words into one large list, and sort for easy charRNN ops
        wordchars = [w for sent in wordchars for w in sent]
        word_lens = [len(x) for x in wordchars]
        (wordchars,), word_orig_idx = sort_all([wordchars], word_lens)
        word_lens = [len(x) for x in wordchars] # we need to reinterpret lengths for the RNN

        # We now pad everything
        words = pad_sequence(words, True, PAD_ID)
        tags = [pad_sequence(tag, True, PAD_ID) if None not in tag else None
                for tag in tags]
        pretrained = pad_sequence(pretrained, True, PAD_ID)
        wordchars = get_long_tensor(wordchars, len(word_lens))

        # and finally create masks for the padding indices
        words_mask = torch.eq(words, PAD_ID)
        wordchars_mask = torch.eq(wordchars, PAD_ID)

        return DataBatch(words, words_mask, wordchars, wordchars_mask, tags,
                         pretrained, orig_idx, word_orig_idx, lens, word_lens, text, idx)

    @staticmethod
    def load_doc(doc, tag_columns=DEFAULT_TAG_COLUMNS):
        """
        Read the text and each tag column out of a Document

        The result is one list per word, with the text at index 0 and
        the columns following in order, so a column's index is its
        position in tag_columns plus one.  That is the number every
        vocab builder wants.
        """
        # a column which lives in MISC asks for MISC and then picks its
        # key back out.  asking for MISC more than once is harmless
        fields = [TEXT] + [MISC if column.misc_key else column.field for column in tag_columns]
        data = doc.get(fields, as_sentences=True)
        for idx, column in enumerate(tag_columns, 1):
            if not column.misc_key:
                continue
            for sentence in data:
                for word in sentence:
                    word[idx] = extract_misc_value(word[idx], column.misc_key)
        data = Dataset.resolve_none(data)
        data = simplify_punct(data)
        return data

    @staticmethod
    def resolve_none(data):
        # replace None to '_'
        for sent_idx in range(len(data)):
            for tok_idx in range(len(data[sent_idx])):
                for feat_idx in range(len(data[sent_idx][tok_idx])):
                    if data[sent_idx][tok_idx][feat_idx] is None:
                        data[sent_idx][tok_idx][feat_idx] = '_'
        return data

class LengthLimitedBatchSampler(Sampler):
    """
    Batches up the text in batches of batch_size, but cuts off each time a batch reaches maximum_tokens

    Intent is to avoid GPU OOM in situations where one sentence is significantly longer than expected,
    leaving a batch too large to fit in the GPU

    Sentences which are longer than maximum_tokens by themselves are put in their own batches
    """
    def __init__(self, data, batch_size, maximum_tokens):
        """
        Precalculate the batches, making it so len and iter just read off the precalculated batches
        """
        self.data = data
        self.batch_size = batch_size
        self.maximum_tokens = maximum_tokens

        self.batches = []
        current_batch = []
        current_length = 0

        for item, item_idx in data:
            item_len = len(item.word)
            if maximum_tokens and item_len > maximum_tokens:
                if len(current_batch) > 0:
                    self.batches.append(current_batch)
                    current_batch = []
                    current_length = 0
                self.batches.append([item_idx])
                continue
            if len(current_batch) + 1 > batch_size or (maximum_tokens and item_len + current_length > maximum_tokens):
                self.batches.append(current_batch)
                current_batch = []
                current_length = 0
            current_batch.append(item_idx)
            current_length += item_len

        if len(current_batch) > 0:
            self.batches.append(current_batch)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            current_batch = []
            for idx in batch:
                current_batch.append(idx)
            yield current_batch


class ShuffledDataset:
    """A wrapper around one or more datasets which shuffles the data in batch_size chunks

    This means that if multiple datasets are passed in, the batches
    from each dataset are shuffled together, with one batch being
    entirely members of the same dataset.

    The main use case of this is that in the tagger, there are cases
    where batches from different datasets will have different
    properties, such as having or not having UPOS tags.  We found that
    it is actually somewhat tricky to make the model's loss function
    (in model.py) properly represent batches with mixed w/ and w/o
    property, whereas keeping one entire batch together makes it a lot
    easier to process.

    The mechanism for the shuffling is that the iterator first makes a
    list long enough to represent each batch from each dataset,
    tracking the index of the dataset it is coming from, then shuffles
    that list.  Another alternative would be to use a weighted
    randomization approach, but this is very simple and the memory
    requirements are not too onerous.

    Note that the batch indices are wasteful in the case of only one
    underlying dataset, which is actually the most common use case,
    but the overhead is small enough that it probably isn't worth
    special casing the one dataset version.
    """
    def __init__(self, datasets, batch_size, ratios=None):
        self.batch_size = batch_size
        self.datasets = datasets
        self.loaders = [x.to_loader(batch_size=self.batch_size, shuffle=True) for x in self.datasets]

        if ratios is None:
            ratios = [1.0] * len(self.datasets)
        if len(ratios) != len(self.datasets):
            raise ValueError("Got %d ratios for %d datasets" % (len(ratios), len(self.datasets)))
        if any(x < 0 for x in ratios):
            raise ValueError("Dataset ratios cannot be negative: %s" % (ratios,))
        self.ratios = list(ratios)

        # how many batches an epoch takes from each dataset.  A ratio
        # of 1.0 is one pass.  The point of a smaller ratio is that a
        # dataset which is much larger than the others would otherwise
        # supply most of the batches, and the heads which only that
        # dataset trains would dominate the shared layers.
        #
        # A dataset asked for more batches than it has is iterated
        # again, reshuffled, rather than repeating the same batches.
        #
        # A loss weight is not a substitute for this.  The batches
        # here are each entirely from one dataset, so a per-dataset
        # loss weight scales a whole batch's gradient uniformly, and
        # Adam's second moment normalization then removes most of that
        # scaling again.  Changing how often a dataset is sampled is
        # the knob that actually moves.
        # max(1, ...) so that a small dataset with a small ratio still
        # appears, but not for an empty dataset: there would be no
        # batch to hand out.  A dataset can be empty here even though
        # the tagger checks the total, for example if it was the
        # shorter half of a bert length filter
        self.batch_counts = [0 if ratio == 0 or len(loader) == 0 else max(1, round(ratio * len(loader)))
                             for ratio, loader in zip(self.ratios, self.loaders)]
        if any(x != 1.0 for x in self.ratios):
            logger.info("Batches per epoch, after ratios %s: %s (was %s)",
                        self.ratios, self.batch_counts, [len(x) for x in self.loaders])

    def __iter__(self):
        iterators = [iter(x) for x in self.loaders]
        indices = [idx for idx, count in enumerate(self.batch_counts) for _ in range(count)]
        random.shuffle(indices)

        for idx in indices:
            try:
                yield next(iterators[idx])
            except StopIteration:
                # asked for more batches than this dataset has, so go
                # around again with a fresh shuffle
                iterators[idx] = iter(self.loaders[idx])
                yield next(iterators[idx])

    def num_batches(self):
        """How many batches one pass of __iter__ yields"""
        return sum(self.batch_counts)

    def __len__(self):
        return sum(len(x) for x in self.datasets)
