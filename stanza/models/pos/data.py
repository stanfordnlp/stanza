import random
import logging
import torch
from collections import namedtuple

from torch.utils.data import DataLoader as DL

from stanza.models.common.bert_embedding import filter_data
from stanza.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanza.models.common.vocab import PAD_ID, VOCAB_PREFIX, CharVocab
from stanza.models.pos.vocab import WordVocab, XPOSVocab, FeatureVocab, MultiVocab
from stanza.models.pos.xpos_vocab_factory import xpos_vocab_factory
from stanza.models.common.doc import *

logger = logging.getLogger('stanza')

DataSample = namedtuple("DataSample", "word char upos xpos feats pretrain text")

class Dataset:
    def __init__(self, doc, args, pretrain, vocab=None, evaluation=False, sort_during_eval=False, bert_tokenizer=None, **kwargs):
        # we takes extra kwargs
        # self.batch_size = batch_size
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

        # shuffle for training
        if self.shuffled:
            random.shuffle(data)
        self.num_examples = len(data)
        self.__punct_tags = self.vocab["upos"].map(["PUNCT"])

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
                text = [w[0] for w in sent]
            )
            processed.append(processed_sent)

        return processed

    def __len__(self):
        return len(self.data)

    def __mask(self, upos):
        """Returns a torch boolean about which elements should be masked out"""

        # creates temporary tensor to operate on
        upos = torch.tensor(upos)

        # creates all false mask
        mask = ~upos.bool()

        ### augmentation 1: punctuation augmentation ###
        # tags that needs to be checked, currently only PUNCT
        augment_nopunct = self.args.get("augment_nopunct")

        if augment_nopunct is None:
            # default augment to 30%
            augment_nopunct = 0.3

        if random.uniform(0,1) < augment_nopunct:
            for i in self.__punct_tags: 
                # generate a mask for the last element
                last_element = torch.zeros_like(upos, dtype=torch.bool)
                last_element[..., -1] = True
                # we or the bitmask against the existing mask
                # if it satisfies, we remove the word by masking it
                # to true
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

        # convert to tensors
        words = sample.word

        # some data augmentation requires constructing a mask based on
        # which upos. For instance, sometimes we'd like to mask out ending
        # sentence punctuation. The mask is True if we want to remove the element
        upos = sample.upos if self.has_upos else None
        if not self.has_upos and upos is not None and (not self.eval):
            # perform actual masking
            mask = self.__mask(upos)
        else:
            # dummy mask that's all false 
            mask = torch.zeros_like(get_long_tensor(words, max([len(i)
                                                                for i in words])),
                                    dtype=torch.bool)
        mask_index = mask.nonzero()
        # convert rest to tensors
        xpos =  sample.xpos if self.has_xpos else None
        ufeats = sample.feats if self.has_feats else None
        pretrained = sample.pretrain

        # and deal with char
        char = sample.char.copy()

        # mask out the elements that we need to mask out
        for mask in mask_index:
            words[mask[0]][mask[1]] = PAD_ID
            upos[mask[0]][mask[1]] = PAD_ID
            xpos[mask[0]][mask[1]] = PAD_ID
            ufeats[mask[0]][mask[1]] = PAD_ID
            pretrained[mask[0]][mask[1]] = PAD_ID
            del char[mask[0]][mask[1]] # to avoid screwing up word len
            
        # get each character from the input sentnece
        chars = [w for sent in char for w in sent]

        return DataSample(words, chars, upos, xpos, ufeats, pretrained, sample[6]), key

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def to_loader(self, **kwargs):
        """Converts self to a DataLoader """

        return DL(self,
                  collate_fn=Dataset.__collate_fn,
                  **kwargs)

    @staticmethod
    def __collate_fn(data):
        """Function used by DataLoader to pack data"""
        (data, idx) = zip(*data)
        (words, wordchars, upos, xpos, ufeats, pretrained, text) = zip(*data)

        # collate_fn is given a list of length batch size
        batch_size = len(data)

        # flatten everything else (because they seem to contain
        # only one sentence each. TODO are we sure about this @John)
        words = [i[0] for i in words]
        if None not in upos:
            upos = [i[0] for i in upos]
        if None not in xpos:
            xpos = [i[0] for i in xpos]
        if None not in ufeats:
            ufeats = [i[0] for i in ufeats]
        pretrained = [i[0] for i in pretrained]

        # sort sentences by lens for easy RNN operations
        lens = [len(x) for x in words]
        (words, upos, xpos,
         ufeats, pretrained, text), orig_idx = sort_all((words, upos, xpos,
                                                         ufeats, pretrained, text), lens)
        lens = [len(x) for x in words] # we need to reinterpret lengths for the RNN

        # combine all words into one large list, and sort for easy charRNN ops
        wordchars = sum(wordchars, [])
        word_lens = [len(x) for x in wordchars]
        (wordchars,), word_orig_idx = sort_all([wordchars], word_lens)
        word_lens = [len(x) for x in wordchars] # we need to reinterpret lengths for the RNN

        # We now pad everything
        words = get_long_tensor(words, batch_size)
        if None not in upos:
            upos = get_long_tensor(upos, batch_size)
        else:
            upos = None
        if None not in xpos:
            xpos = get_long_tensor(xpos, batch_size)
        else:
            xpos = None
        if None not in ufeats:
            ufeats = get_long_tensor(ufeats, batch_size)
        else:
            ufeats = None
        pretrained = get_long_tensor(pretrained, batch_size)
        wordchars = get_long_tensor(wordchars, len(word_lens))

        # and finally create masks for the padding indicies 
        words_mask = torch.eq(words, PAD_ID)
        wordchars_mask = torch.eq(wordchars, PAD_ID)

        return (words, words_mask, wordchars, wordchars_mask, upos, xpos, ufeats,
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


