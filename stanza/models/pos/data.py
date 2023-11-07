import random
import logging
import copy
import torch
from collections import namedtuple

from torch.utils.data import DataLoader as DL
from torch.utils.data import Dataset as DS
from torch.nn.utils.rnn import pad_sequence

from stanza.models.common.bert_embedding import filter_data
from stanza.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanza.models.common.vocab import PAD_ID, VOCAB_PREFIX, CharVocab
from stanza.models.pos.vocab import WordVocab, XPOSVocab, FeatureVocab, MultiVocab
from stanza.models.pos.xpos_vocab_factory import xpos_vocab_factory
from stanza.models.common.doc import *

logger = logging.getLogger('stanza')

DataSample = namedtuple("DataSample", "word char upos xpos feats pretrain has_upos has_xpos has_feats text")
DataBatch = namedtuple("DataBatch", "words words_mask wordchars wordchars_mask upos xpos ufeats pretrained orig_idx word_orig_idx lens word_lens text idx")

def merge_datasets(datasets):
    """Merge multiple datasets"""

    return _ShadowDataset(*datasets)

class _ShadowDataset(DS):
    def __init__(self, *datasets):
        self.datasets = datasets

        # precache the lengths of the datasets, cumulated
        self.__cumulate_lens = []
        self.__len = 0
        for i in self.datasets:
            self.__cumulate_lens.append(self.__len)
            self.__len += len(i)


    def to_loader(self, **kwargs):
        """Converts self to a DataLoader """

        return DL(self, collate_fn=self.__collate_fn, **kwargs)

    def __indx2loader(self, index):
        """Search through the loader lengths to get the id to the right dataset"""

        # we iterate through cumulative lengths in *REVERSE* bec
        for indx, i in reversed(list(enumerate(self.__cumulate_lens))):
            if index >= i:
                return indx, index-i

    def __getitem__(self, key):
        """Get a single key for whether or not upos/xpos etc. is avaliable"""

        dataset_num, indx = self.__indx2loader(key)
        return self.datasets[dataset_num][indx], key

    def __len__(self):
        return self.__len

    @staticmethod
    def __collate_fn(data):
        """Function used by DataLoader to pack data"""
        (data, idx) = zip(*data)
        (words, wordchars, upos, xpos, ufeats, pretrained,
         has_upos, has_xpos, has_feats, text) = zip(*data)

        # collate_fn is given a list of length batch size
        batch_size = len(data)

        # sort sentences by lens for easy RNN operations
        lens = [torch.sum(x != PAD_ID) for x in words]
        (words, wordchars, upos, xpos, ufeats, pretrained,
         has_upos, has_xpos, has_feats, text), orig_idx = sort_all((words, wordchars, upos, xpos, ufeats, pretrained,
                                                                    has_upos, has_xpos, has_feats, text), lens)
        lens = [torch.sum(x != PAD_ID) for x in words] # we need to reinterpret lengths for the RNN

        # combine all words into one large list, and sort for easy charRNN ops
        wordchars = [w for sent in wordchars for w in sent]
        word_lens = [len(x) for x in wordchars]
        (wordchars,), word_orig_idx = sort_all([wordchars], word_lens)
        word_lens = [len(x) for x in wordchars] # we need to reinterpret lengths for the RNN

        # We now pad everything
        words = pad_sequence(words, True, PAD_ID)
        upos = pad_sequence(upos, True, PAD_ID)
        xpos = pad_sequence(xpos, True, PAD_ID)
        ufeats = pad_sequence(ufeats, True, PAD_ID)
        pretrained = pad_sequence(pretrained, True, PAD_ID)
        wordchars = get_long_tensor(wordchars, len(word_lens))

        # and get boolean mask tensors for upos, xpos, feats
        upos_mask = torch.tensor(has_upos)
        xpos_mask = torch.tensor(has_xpos)
        feats_mask = torch.tensor(has_feats)

        # mask out the elements for which upos/xpos/feats isn't available
        upos[~upos_mask] = PAD_ID
        xpos[~xpos_mask] = PAD_ID
        ufeats[~feats_mask] = PAD_ID

        # and finally create masks for the padding indices
        words_mask = torch.eq(words, PAD_ID)
        wordchars_mask = torch.eq(wordchars, PAD_ID)

        return DataBatch(words, words_mask, wordchars, wordchars_mask, upos, xpos, ufeats,
                         pretrained, orig_idx, word_orig_idx, lens, word_lens, text, idx)

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
        if self.args.get('bert_model', None):
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
        featsvocab = FeatureVocab(data, args['shorthand'], idx=3)
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
                has_upos = self.has_upos,
                has_xpos = self.has_xpos,
                has_feats = self.has_feats,
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
        upos = torch.tensor(sample.upos[0])
        xpos = torch.tensor(sample.xpos[0])
        ufeats = torch.tensor(sample.feats[0])
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

        return DataSample(words, char, upos, xpos, ufeats, pretrained,
                          self.has_upos, self.has_xpos, self.has_feats, raw_text)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def to_loader(self, **kwargs):
        """Converts self to a DataLoader """

        return _ShadowDataset(self).to_loader(**kwargs)

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


