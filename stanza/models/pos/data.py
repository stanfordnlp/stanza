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

# # sort sentences by lens for easy RNN operations
# lens = [len(x) for x in batch[0]]
# batch, orig_idx = sort_all(batch, lens)

# # sort words by lens for easy char-RNN operations
# batch_words = [w for sent in batch[1] for w in sent]
# word_lens = [len(x) for x in batch_words]
# batch_words, word_orig_idx = sort_all([batch_words], word_lens)
# batch_words = batch_words[0]
# word_lens = [len(x) for x in batch_words]


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

        self.data = self.preprocess(data, self.vocab, self.pretrain_vocab, args)
        # shuffle for training
        if self.shuffled:
            random.shuffle(data)
        self.num_examples = len(data)

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

        # creates all false mask
        mask = ~upos.bool()

        ### augmentation 1: punctuation augmentation ###
        # tags that needs to be checked, currently only PUNCT
        punct_tags = self.vocab["upos"].map(["PUNCT"])

        augment_nopunct = self.args.get("augment_nopunct")

        if not augment_nopunct:
            # default augment to 30%
            augment_nopunct = 0.3

        if random.uniform(0,1) < augment_nopunct:
            for i in punct_tags: 
                # generate a mask for the last element
                last_element = torch.zeros_like(upos, dtype=torch.bool)
                last_element[..., -1] = True
                # we or the bitmask against the existing mask
                # if it satisfies, we remove the word by masking it
                # to true
                mask |= ((upos == i) & (last_element))

        return mask

    def __getitem__(self, key):
        """ Get a batch with index. """

        # get a sample of the input data
        sample = self.data[key]

        # convert to tensors
        words = sample.word
        words = torch.tensor(words)

        # some data augmentation requires constructing a mask based on
        # which upos. For instance, sometimes we'd like to mask out ending
        # sentence punctuation. The mask is True if we want to remove the element
        upos = torch.tensor(sample.upos) if self.has_upos else None
        if self.has_upos: # and if not eval?
            # perform actual masking
            mask = self.__mask(upos)
        else:
            # dummy mask that's all false 
            mask = torch.zeros_like(words, dtype=torch.bool)
        mask_index = mask.nonzero()

        # convert rest to tensors
        xpos =  torch.tensor(sample.xpos)if self.has_xpos else None
        ufeats = torch.tensor(sample.feats) if self.has_feats else None
        pretrained = torch.tensor(sample.pretrain)

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
            
        # calculate sentence lengths
        sentlens = [len(x) for x in sample.word]

        # flatten and pad characters
        batch_words = [w for sent in char for w in sent]
        word_lens = [len(x) for x in batch_words]
        batch_words = batch_words[0]
        wordchars = get_long_tensor(batch_words, len(word_lens))

        # calculate output masks
        wordchars_mask = torch.eq(char, PAD_ID)
        words_mask = torch.eq(words, PAD_ID)

        # return words, words_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, orig_idx, word_orig_idx, sentlens, word_lens, text
        return words, words_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, sentlens, word_lens, text

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

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


