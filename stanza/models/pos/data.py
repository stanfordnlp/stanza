import random
import logging
import copy
import torch
from collections import namedtuple

from torch.utils.data import DataLoader as DL
from torch.utils.data.sampler import Sampler
from torch.nn.utils.rnn import pad_sequence

from stanza.models.common.bert_embedding import filter_data, needs_length_filter
from stanza.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanza.models.common.vocab import PAD_ID, VOCAB_PREFIX, CharVocab
from stanza.models.pos.vocab import WordVocab, XPOSVocab, FeatureVocab, MultiVocab
from stanza.models.pos.xpos_vocab_factory import xpos_vocab_factory
from stanza.models.common.doc import *

logger = logging.getLogger('stanza')

DataSample = namedtuple("DataSample", "word char upos xpos feats pretrain text")
DataBatch = namedtuple("DataBatch", "words words_mask wordchars wordchars_mask upos xpos ufeats pretrained orig_idx word_orig_idx lens word_lens text idx")

class Dataset:
    def __init__(self, doc, args, pretrain, vocab=None, evaluation=False, sort_during_eval=False, bert_tokenizer=None, **kwargs):
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.sort_during_eval = sort_during_eval
        self.doc = doc

        if vocab is None:
            self.vocab = Dataset.init_vocab([doc], args)
        else:
            self.vocab = vocab

        self.has_upos = not all(x is None or x == '_' for x in doc.get(UPOS, as_sentences=False))
        self.has_xpos = not all(x is None or x == '_' for x in doc.get(XPOS, as_sentences=False))
        self.has_feats = not all(x is None or x == '_' for x in doc.get(FEATS, as_sentences=False))

        data = self.load_doc(self.doc)
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

        data = self.preprocess(data, self.vocab, self.pretrain_vocab, args)

        self.data = data

        self.num_examples = len(data)
        self.__punct_tags = self.vocab["upos"].map(["PUNCT"])
        self.augment_nopunct = self.args.get("augment_nopunct", 0.0)

    @staticmethod
    def init_vocab(docs, args):
        data = [x for doc in docs for x in Dataset.load_doc(doc)]
        charvocab = CharVocab(data, args['shorthand'])
        wordvocab = WordVocab(data, args['shorthand'], cutoff=args['word_cutoff'], lower=True)
        uposvocab = WordVocab(data, args['shorthand'], idx=1)
        xposvocab = xpos_vocab_factory(data, args['shorthand'])
        try:
            featsvocab = FeatureVocab(data, args['shorthand'], idx=3)
        except ValueError as e:
            raise ValueError("Unable to build features vocab.  Please check the Features column of your data for an error which may match the following description.") from e
        vocab = MultiVocab({'char': charvocab,
                            'word': wordvocab,
                            'upos': uposvocab,
                            'xpos': xposvocab,
                            'feats': featsvocab})
        return vocab

    def preprocess(self, data, vocab, pretrain_vocab, args):
        processed = []
        for sent in data:
            processed_sent = DataSample(
                word = [vocab['word'].map([w[0] for w in sent])],
                char = [[vocab['char'].map([x for x in w[0]]) for w in sent]],
                upos = [vocab['upos'].map([w[1] for w in sent])],
                xpos = [vocab['xpos'].map([w[2] for w in sent])],
                feats = [vocab['feats'].map([w[3] for w in sent])],
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
        restoration. It recieves a DataSample object from the storage,
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
        # convert the rest to tensors
        upos = torch.tensor(sample.upos[0]) if self.has_upos else None
        xpos = torch.tensor(sample.xpos[0]) if self.has_xpos else None
        ufeats = torch.tensor(sample.feats[0]) if self.has_feats else None
        pretrained = torch.tensor(sample.pretrain[0])

        # and deal with char & raw_text
        char = sample.char[0]
        raw_text = sample.text

        # some data augmentation requires constructing a mask based on
        # which upos. For instance, sometimes we'd like to mask out ending
        # sentence punctuation. The mask is True if we want to remove the element
        if self.has_upos and upos is not None and not self.eval:
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
                if upos is not None:
                    upos[mask] = PAD_ID
                if xpos is not None:
                    # TODO: test the multi-dimension xpos
                    xpos[mask, ...] = PAD_ID
                if ufeats is not None:
                    ufeats[mask, ...] = PAD_ID
                pretrained[mask] = PAD_ID
                char = char[:mask] + char[mask+1:]
                raw_text = raw_text[:mask] + raw_text[mask+1:]

        # get each character from the input sentnece
        # chars = [w for sent in char for w in sent]

        return DataSample(words, char, upos, xpos, ufeats, pretrained, raw_text), key

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def to_loader(self, **kwargs):
        """Converts self to a DataLoader """

        return DL(self,
                  collate_fn=Dataset.__collate_fn,
                  **kwargs)

    def to_length_limited_loader(self, batch_size, maximum_tokens):
        sampler = LengthLimitedBatchSampler(self, batch_size, maximum_tokens)
        return DL(self,
                  collate_fn=Dataset.__collate_fn,
                  batch_sampler = sampler)

    @staticmethod
    def __collate_fn(data):
        """Function used by DataLoader to pack data"""
        (data, idx) = zip(*data)
        (words, wordchars, upos, xpos, ufeats, pretrained, text) = zip(*data)

        # collate_fn is given a list of length batch size
        batch_size = len(data)

        # sort sentences by lens for easy RNN operations
        lens = [torch.sum(x != PAD_ID) for x in words]
        (words, wordchars, upos, xpos,
         ufeats, pretrained, text), orig_idx = sort_all((words, wordchars, upos, xpos,
                                                         ufeats, pretrained, text), lens)
        lens = [torch.sum(x != PAD_ID) for x in words] # we need to reinterpret lengths for the RNN

        # combine all words into one large list, and sort for easy charRNN ops
        wordchars = [w for sent in wordchars for w in sent]
        word_lens = [len(x) for x in wordchars]
        (wordchars,), word_orig_idx = sort_all([wordchars], word_lens)
        word_lens = [len(x) for x in wordchars] # we need to reinterpret lengths for the RNN

        # We now pad everything
        words = pad_sequence(words, True, PAD_ID)
        if None not in upos:
            upos = pad_sequence(upos, True, PAD_ID)
        else:
            upos = None
        if None not in xpos:
            xpos = pad_sequence(xpos, True, PAD_ID)
        else:
            xpos = None
        if None not in ufeats:
            ufeats = pad_sequence(ufeats, True, PAD_ID)
        else:
            ufeats = None
        pretrained = pad_sequence(pretrained, True, PAD_ID)
        wordchars = get_long_tensor(wordchars, len(word_lens))

        # and finally create masks for the padding indices
        words_mask = torch.eq(words, PAD_ID)
        wordchars_mask = torch.eq(wordchars, PAD_ID)

        return DataBatch(words, words_mask, wordchars, wordchars_mask, upos, xpos, ufeats,
                         pretrained, orig_idx, word_orig_idx, lens, word_lens, text, idx)

    @staticmethod
    def load_doc(doc):
        data = doc.get([TEXT, UPOS, XPOS, FEATS], as_sentences=True)
        data = Dataset.resolve_none(data)
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
    def __init__(self, datasets, batch_size):
        self.batch_size = batch_size
        self.datasets = datasets
        self.loaders = [x.to_loader(batch_size=self.batch_size, shuffle=True) for x in self.datasets]

    def __iter__(self):
        iterators = [iter(x) for x in self.loaders]
        lengths = [len(x) for x in self.loaders]
        indices = [[x] * y for x, y in enumerate(lengths)]
        indices = [idx for inner in indices for idx in inner]
        random.shuffle(indices)

        for idx in indices:
            yield(next(iterators[idx]))

    def __len__(self):
        return sum(len(x) for x in self.datasets)
