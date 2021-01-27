import random
import logging
import torch

from stanza.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
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


class DataLoader:

    def __init__(self, doc, batch_size, args, pretrain, vocab=None, evaluation=False, sort_during_eval=False, min_length_to_batch_separately=None):
        self.batch_size = batch_size
        self.min_length_to_batch_separately=min_length_to_batch_separately
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.sort_during_eval = sort_during_eval
        self.doc = doc
        data = self.load_doc(doc)

        # handle vocab
        if vocab is None:
            self.vocab = self.init_vocab(data)
        else:
            self.vocab = vocab
        
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
        # shuffle for training
        if self.shuffled:
            random.shuffle(data)
        self.num_examples = len(data)

        # chunk into batches
        self.data = self.chunk_batches(data)
        logger.debug("{} batches created.".format(len(self.data)))

    def init_vocab(self, data):
        assert self.eval == False # for eval vocab must exist
        charvocab = CharVocab(data, self.args['shorthand'])
        wordvocab = WordVocab(data, self.args['shorthand'], cutoff=7, lower=True)
        uposvocab = WordVocab(data, self.args['shorthand'], idx=1)
        xposvocab = xpos_vocab_factory(data, self.args['shorthand'])
        featsvocab = FeatureVocab(data, self.args['shorthand'], idx=3)
        lemmavocab = WordVocab(data, self.args['shorthand'], cutoff=7, idx=4, lower=True)
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
            processed.append(processed_sent)
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 9

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
        return words, words_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, orig_idx, word_orig_idx, sentlens, word_lens

    def load_doc(self, doc):
        data = doc.get([TEXT, UPOS, XPOS, FEATS, LEMMA, HEAD, DEPREL], as_sentences=True)
        data = self.resolve_none(data)
        return data

    def resolve_none(self, data):
        # replace None to '_'
        for sent_idx in range(len(data)):
            for tok_idx in range(len(data[sent_idx])):
                for feat_idx in range(len(data[sent_idx][tok_idx])):
                    if data[sent_idx][tok_idx][feat_idx] is None:
                        data[sent_idx][tok_idx][feat_idx] = '_'
        return data

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def reshuffle(self):
        data = [y for x in self.data for y in x]
        self.data = self.chunk_batches(data)
        random.shuffle(self.data)

    def chunk_batches(self, data):
        batches, data_orig_idx = data_to_batches(data=data, batch_size=self.batch_size,
                                                 eval_mode=self.eval, sort_during_eval=self.sort_during_eval,
                                                 min_length_to_batch_separately=self.min_length_to_batch_separately)
        # data_orig_idx might be None at train time, since we don't anticipate unsorting
        self.data_orig_idx = data_orig_idx
        return batches

def to_int(string, ignore_error=False):
    try:
        res = int(string)
    except ValueError as err:
        if ignore_error:
            return 0
        else:
            raise err
    return res

