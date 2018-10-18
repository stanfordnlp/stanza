import random
import numpy as np
import lzma
import os
import pickle
from collections import Counter
import torch

from models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from models.common import conll
from models.common.constant import lcode2lang
from models.common.vocab import PAD_ID, VOCAB_PREFIX
from models.pos.vocab import CharVocab, WordVocab, XPOSVocab, FeatureVocab, PretrainedWordVocab
from models.pos.xpos_vocab_factory import xpos_vocab_factory

class DataLoader:
    def __init__(self, filename, batch_size, args, evaluation=False):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval

        assert filename.endswith('conllu'), "Loaded file must be conllu file."
        self.conll, data = self.load_file(filename, evaluation=self.eval)

        # handle vocab
        vocab_pattern = "{}/{}.{{}}.vocab".format(args['data_dir'], args['shorthand'])
        self.vocab = self.init_vocab(vocab_pattern, data)

	# filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)
            print("Subsample training set with rate {:g}".format(args['sample_train']))

        data = self.preprocess(data, self.vocab, args)
        # shuffle for training
        if self.shuffled:
            random.shuffle(data)
        self.num_examples = len(data)

	# chunk into batches
        self.data = self.chunk_batches(data)
        print("{} batches created for {}.".format(len(self.data), filename))

    def init_vocab(self, vocab_pattern, data):
        types = ['char', 'word', 'upos', 'xpos', 'feats']
        if not all([os.path.exists(vocab_pattern.format(type_)) for type_ in types]):
            assert self.eval == False # for eval vocab file must exist
        charvocab = CharVocab(vocab_pattern.format('char'), data, self.args['shorthand'])
        wordvocab = WordVocab(vocab_pattern.format('word'), data, self.args['shorthand'], cutoff=7, lower=True)
        self.pretrained_emb, pretrainedvocab = self.read_emb_matrix(self.args['wordvec_dir'], self.args['shorthand'], vocab_pattern.format('pretrained'))
        uposvocab = WordVocab(vocab_pattern.format('upos'), data, self.args['shorthand'], idx=1)
        xposvocab = xpos_vocab_factory(vocab_pattern.format('xpos'), data, self.args['shorthand'])
        featsvocab = FeatureVocab(vocab_pattern.format('feats'), data, self.args['shorthand'], idx=3)
        vocab = {'char': charvocab,
                'word': wordvocab,
                'pretrained': pretrainedvocab,
                'upos': uposvocab,
                'xpos': xposvocab,
                'feats': featsvocab}
        return vocab

    def read_emb_matrix(self, wordvec_dir, shorthand, vocab_file):
        vec_file = vocab_file + '.vec'
        if not os.path.exists(vocab_file) or not os.path.exists(vec_file):
            lcode, tcode = shorthand.split('_')

            lang = lcode2lang[lcode] if lcode != 'no' else lcode2lang[shorthand]
            if lcode == 'zh':
                lang = 'ChineseT'
            wordvec_file = os.path.join(wordvec_dir, lang, '{}.vectors.xz'.format(lcode if lcode != 'no' else (shorthand if shorthand != 'no_nynorsklia' else 'no_nynorsk')))

            first = True
            words = []
            failed = 0
            with lzma.open(wordvec_file, 'rb') as f:
                for i, line in enumerate(f):
                    try:
                        line = line.decode()
                    except UnicodeDecodeError:
                        failed += 1
                        continue
                    if first:
                        # the first line contains the number of word vectors and the
                        # dimensionality of them
                        first = False
                        line = line.strip().split(' ')
                        rows, cols = [int(x) for x in line]
                        res = np.zeros((rows + len(VOCAB_PREFIX), cols), dtype=np.float32) # save embeddings for special tokens
                        continue

                    line = line.rstrip().split(' ')
                    res[i+len(VOCAB_PREFIX)-1-failed, :] = [float(x) for x in line[-cols:]]
                    words.append(' '.join(line[:-cols]))

            pretrained_vocab = PretrainedWordVocab(vocab_file, words, shorthand)

            if failed > 0:
                res = res[:-failed]

            with open(vec_file, 'wb') as f:
                pickle.dump(res, f)

        else:
            pretrained_vocab = PretrainedWordVocab(vocab_file, [], shorthand)
            with open(vec_file, 'rb') as f:
                res = pickle.load(f)

        return res, pretrained_vocab

    def preprocess(self, data, vocab, args):
        processed = []
        for sent in data:
            processed_sent = [vocab['word'].map([w[0] for w in sent])]
            processed_sent += [[vocab['char'].map([x for x in w[0]]) for w in sent]]
            processed_sent += [vocab['upos'].map([w[1] for w in sent])]
            processed_sent += [vocab['xpos'].map([w[2] for w in sent])]
            processed_sent += [vocab['feats'].map([w[3] for w in sent])]
            processed_sent += [vocab['pretrained'].map([w[0] for w in sent])]
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

    def reshuffle(self):
        data = [y for x in self.data for y in x]
        random.shuffle(data)
        self.data = self.chunk_batches(data)

    def chunk_batches(self, data):
        res = []

        if not self.eval:
            # sort sentences (roughly) by length for better memory utilization
            data = sorted(data, key = lambda x: len(x[0]) + random.random() * 5)

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
