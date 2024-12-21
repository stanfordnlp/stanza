import random
import numpy as np
import os
from collections import Counter
import logging
import torch

import stanza.models.common.seq2seq_constant as constant
from stanza.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanza.models.common.vocab import DeltaVocab
from stanza.models.lemma.vocab import Vocab, MultiVocab
from stanza.models.lemma import edit
from stanza.models.common.doc import *

logger = logging.getLogger('stanza')

class DataLoader:
    def __init__(self, doc, batch_size, args, vocab=None, evaluation=False, conll_only=False, skip=None, expand_unk_vocab=False):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.doc = doc

        data = self.raw_data()

        if conll_only: # only load conll file
            return

        if skip is not None:
            assert len(data) == len(skip)
            data = [x for x, y in zip(data, skip) if not y]

        # handle vocab
        if vocab is not None:
            if expand_unk_vocab:
                pos_vocab = vocab['pos']
                char_vocab = DeltaVocab(data, vocab['char'])
                self.vocab = MultiVocab({'char': char_vocab, 'pos': pos_vocab})
            else:
                self.vocab = vocab
        else:
            self.vocab = dict()
            char_vocab, pos_vocab = self.init_vocab(data)
            self.vocab = MultiVocab({'char': char_vocab, 'pos': pos_vocab})

        # filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)
            logger.debug("Subsample training set with rate {:g}".format(args['sample_train']))

        data = self.preprocess(data, self.vocab['char'], self.vocab['pos'], args)
        # shuffle for training
        if self.shuffled:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        logger.debug("{} batches created.".format(len(data)))

    def init_vocab(self, data):
        assert self.eval is False, "Vocab file must exist for evaluation"
        char_data = "".join(d[0] + d[2] for d in data)
        char_vocab = Vocab(char_data, self.args['lang'])
        pos_data = [d[1] for d in data]
        pos_vocab = Vocab(pos_data, self.args['lang'])
        return char_vocab, pos_vocab

    def preprocess(self, data, char_vocab, pos_vocab, args):
        processed = []
        for d in data:
            edit_type = edit.EDIT_TO_ID[edit.get_edit_type(d[0], d[2])]
            src = list(d[0])
            src = [constant.SOS] + src + [constant.EOS]
            src = char_vocab.map(src)
            pos = d[1]
            pos = pos_vocab.unit2id(pos)
            tgt = list(d[2])
            tgt_in = char_vocab.map([constant.SOS] + tgt)
            tgt_out = char_vocab.map(tgt + [constant.EOS])
            processed += [[src, tgt_in, tgt_out, pos, edit_type, d[0]]]
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

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # convert to tensors
        src = batch[0]
        src = get_long_tensor(src, batch_size)
        src_mask = torch.eq(src, constant.PAD_ID)
        tgt_in = get_long_tensor(batch[1], batch_size)
        tgt_out = get_long_tensor(batch[2], batch_size)
        pos = torch.LongTensor(batch[3])
        edits = torch.LongTensor(batch[4])
        text = batch[5]
        assert tgt_in.size(1) == tgt_out.size(1), "Target input and output sequence sizes do not match."
        return src, src_mask, tgt_in, tgt_out, pos, edits, orig_idx, text

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def raw_data(self):
        return self.load_doc(self.doc, self.args.get('caseless', False), self.eval)

    @staticmethod
    def load_doc(doc, caseless, evaluation):
        if evaluation:
            data = doc.get([TEXT, UPOS, LEMMA])
        else:
            data = doc.get([TEXT, UPOS, LEMMA, HEAD, DEPREL, MISC], as_sentences=True)
            data = DataLoader.remove_goeswith(data)
            data = DataLoader.extract_correct_forms(data)
        data = DataLoader.resolve_none(data)
        if caseless:
            data = DataLoader.lowercase_data(data)
        return data

    @staticmethod
    def extract_correct_forms(data):
        """
        Here we go through the raw data and use the CorrectForm of words tagged with CorrectForm

        In addition, if the incorrect form of the word is not present in the training data,
        we keep the incorrect form for the lemmatizer to learn from.
        This way, it can occasionally get things right in misspelled input text.

        We do check for and eliminate words where the incorrect form is already known as the
        lemma for a different word.  For example, in the English datasets, there is a "busy"
        which was meant to be "buys", and we don't want the model to learn to lemmatize "busy" to "buy"
        """
        new_data = []
        incorrect_forms = []
        for word in data:
            misc = word[-1]
            if not misc:
                new_data.append(word[:3])
                continue
            misc = misc.split("|")
            for piece in misc:
                if piece.startswith("CorrectForm="):
                    cf = piece.split("=", maxsplit=1)[1]
                    # treat the CorrectForm as the desired word
                    new_data.append((cf, word[1], word[2]))
                    # and save the broken one for later in case it wasn't used anywhere else
                    incorrect_forms.append((cf, word))
                    break
            else:
                # if no CorrectForm, just keep the word as normal
                new_data.append(word[:3])
        known_words = {x[0] for x in new_data}
        for correct_form, word in incorrect_forms:
            if word[0] not in known_words:
                new_data.append(word[:3])
        return new_data

    @staticmethod
    def remove_goeswith(data):
        """
        This method specifically removes words that goeswith something else, along with the something else

        The purpose is to eliminate text such as

1	Ken	kenrice@enroncommunications	X	GW	Typo=Yes	0	root	0:root	_
2	Rice@ENRON	_	X	GW	_	1	goeswith	1:goeswith	_
3	COMMUNICATIONS	_	X	ADD	_	1	goeswith	1:goeswith	_
        """
        filtered_data = []
        remove_indices = set()
        for sentence in data:
            remove_indices.clear()
            for word_idx, word in enumerate(sentence):
                if word[4] == 'goeswith':
                    remove_indices.add(word_idx)
                    remove_indices.add(word[3]-1)
            filtered_data.extend([x for idx, x in enumerate(sentence) if idx not in remove_indices])
        return filtered_data

    @staticmethod
    def lowercase_data(data):
        for token in data:
            token[0] = token[0].lower()
        return data

    @staticmethod
    def resolve_none(data):
        # replace None to '_'
        for tok_idx in range(len(data)):
            for feat_idx in range(len(data[tok_idx])):
                if data[tok_idx][feat_idx] is None:
                    data[tok_idx][feat_idx] = '_'
        return data
