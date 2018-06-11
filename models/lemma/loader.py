import random
import numpy as np
import os
from collections import Counter
import torch

import models.common.seq2seq_constant as constant
from models.lemma import conll
from models.lemma.vocab import Vocab

class DataLoader:
    def __init__(self, filename, batch_size, args, evaluation=False):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval

        assert filename.endswith('conllu'), "Loaded file must be conllu file."
        self.conll, data = self.load_file(filename)
        
        # handle vocab
        vocab_file = filename.split('.')[0] + '.vocab'
        self.vocab = self.init_vocab(vocab_file, data)

	# filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)
            print("Subsample training set with rate {:g}".format(args['sample_train']))

        data = self.preprocess(data, self.vocab, args)
        # shuffle for training
        if self.shuffled:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.num_examples = len(data)

	# chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}.".format(len(data), filename))

    def init_vocab(self, vocab_file, data):
        if os.path.exists(vocab_file):
            vocab = Vocab(vocab_file, load=True)
        else:
            assert self.eval == False # for eval vocab file must exist
            chars = "".join([d[0] for d in data])
            char_counter = Counter(chars)
            vocab = Vocab(vocab_file, load=False, unit_counter=char_counter)
        return vocab

    def preprocess(self, data, vocab, args):
        processed = []
        for d in data:
            src = list(d[0])
            src = [constant.SOS] + src + [constant.EOS]
            src = map_to_ids(src, vocab.unit2id)
            if self.eval:
                tgt = src # as a placeholder
            else:
                tgt = list(d[1])
            tgt_in = map_to_ids([constant.SOS] + tgt, vocab.unit2id)
            tgt_out = map_to_ids(tgt + [constant.EOS], vocab.unit2id)
            processed += [[src, tgt_in, tgt_out]]
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
        assert len(batch) == 3
        
        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # convert to tensors
        src = batch[0]
        src = get_long_tensor(src, batch_size)
        src_mask = torch.eq(src, constant.PAD_ID)
        tgt_in = get_long_tensor(batch[1], batch_size)
        tgt_out = get_long_tensor(batch[2], batch_size)
        assert tgt_in.size(1) == tgt_out.size(1), \
                "Target input and output sequence sizes do not match."
        return (src, src_mask, tgt_in, tgt_out, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def load_file(self, filename, evaluation=False):
        conll_file = conll.CoNLLFile(filename)
        if evaluation:
            data = conll_file.get_words()
            data = [[d] for d in data]
        else:
            data = conll_file.get_words_and_lemmas()
        return conll_file, data

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_long_tensor(tokens_list, batch_size, pad_id=constant.PAD_ID):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(pad_id)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def get_float_tensor(features_list, batch_size):
    if features_list is None or features_list[0] is None:
        return None
    seq_len = max(len(x) for x in features_list)
    feature_len = len(features_list[0][0])
    features = torch.FloatTensor(batch_size, seq_len, feature_len).zero_()
    for i,f in enumerate(features_list):
        features[i,:len(f),:] = torch.FloatTensor(f)
    return features

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]
