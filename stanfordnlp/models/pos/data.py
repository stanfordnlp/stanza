import random
import torch

from stanfordnlp.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanfordnlp.models.common import conll
from stanfordnlp.models.common.vocab import PAD_ID, VOCAB_PREFIX
from stanfordnlp.models.pos.vocab import CharVocab, WordVocab, XPOSVocab, FeatureVocab, MultiVocab
from stanfordnlp.models.pos.xpos_vocab_factory import xpos_vocab_factory
from stanfordnlp.pipeline.doc import Document

class DataLoader:
    def __init__(self, input_src, batch_size, args, pretrain, vocab=None, evaluation=False, sort_during_eval=False):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.sort_during_eval = sort_during_eval

        # check if input source is a file or a Document object
        if isinstance(input_src, str):
            filename = input_src
            assert filename.endswith('conllu'), "Loaded file must be conllu file."
            self.conll, data = self.load_file(filename, evaluation=self.eval)
        elif isinstance(input_src, Document):
            filename = None
            doc = input_src
            self.conll, data = self.load_doc(doc)

        # handle vocab
        if vocab is None:
            self.vocab = self.init_vocab(data)
        else:
            self.vocab = vocab
        self.pretrain_vocab = pretrain.vocab

        # filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)
            print("Subsample training set with rate {:g}".format(args['sample_train']))

        data = self.preprocess(data, self.vocab, self.pretrain_vocab, args)
        # shuffle for training
        if self.shuffled:
            random.shuffle(data)
        self.num_examples = len(data)

        # chunk into batches
        self.data = self.chunk_batches(data)
        if filename is not None:
            print("{} batches created for {}.".format(len(self.data), filename))

    def init_vocab(self, data):
        assert self.eval == False # for eval vocab must exist
        charvocab = CharVocab(data, self.args['shorthand'])
        wordvocab = WordVocab(data, self.args['shorthand'], cutoff=7, lower=True)
        uposvocab = WordVocab(data, self.args['shorthand'], idx=1)
        xposvocab = xpos_vocab_factory(data, self.args['shorthand'])
        featsvocab = FeatureVocab(data, self.args['shorthand'], idx=3)
        vocab = MultiVocab({'char': charvocab,
                            'word': wordvocab,
                            'upos': uposvocab,
                            'xpos': xposvocab,
                            'feats': featsvocab})
        return vocab

    def preprocess(self, data, vocab, pretrain_vocab, args):
        processed = []
        for sent in data:
            processed_sent = [vocab['word'].map([w[0] for w in sent])]
            processed_sent += [[vocab['char'].map([x for x in w[0]]) for w in sent]]
            processed_sent += [vocab['upos'].map([w[1] for w in sent])]
            processed_sent += [vocab['xpos'].map([w[2] for w in sent])]
            processed_sent += [vocab['feats'].map([w[3] for w in sent])]
            processed_sent += [pretrain_vocab.map([w[0] for w in sent])]
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
        assert len(batch) == 6

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
        return words, words_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, orig_idx, word_orig_idx, sentlens, word_lens

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def load_file(self, filename, evaluation=False):
        conll_file = conll.CoNLLFile(filename)
        data = conll_file.get(['word', 'upos', 'xpos', 'feats'], as_sentences=True)
        return conll_file, data

    def load_doc(self, doc):
        data = doc.conll_file.get(['word', 'upos', 'xpos', 'feats'], as_sentences=True)
        return doc.conll_file, data

    def reshuffle(self):
        data = [y for x in self.data for y in x]
        self.data = self.chunk_batches(data)
        random.shuffle(self.data)

    def chunk_batches(self, data):
        res = []

        if not self.eval:
            # sort sentences (roughly) by length for better memory utilization
            data = sorted(data, key = lambda x: len(x[0]), reverse=random.random() > .5)
        elif self.sort_during_eval:
            (data, ), self.data_orig_idx = sort_all([data], [len(x[0]) for x in data])

        current = []
        currentlen = 0
        for x in data:
            if len(x[0]) + currentlen > self.batch_size:
                res.append(current)
                current = []
                currentlen = 0
            current.append(x)
            currentlen += len(x[0])

        if currentlen > 0:
            res.append(current)

        return res
