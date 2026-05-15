"""
Data loading and training augmentation for the Stanza neural tokenizer.

The tokenizer treats tokenization and sentence segmentation as a character-level
tagging problem.  Each character in the input is assigned one of the following
labels:

  0  continuation  – character is inside a token (not its last character)
  1  word end      – last character of a token / word
  2  sentence end  – last character of the final token in a sentence
  3  MWT end       – last character of a multi-word token (e.g. "ocultándolo")
  4  MWT+sent end  – last character of an MWT that also ends a sentence

Note that languages with no MWT layer only use 0,1,2.

The central classes are:

TokenizationDataset
    Base class.  Reads raw text (and optionally a parallel label file) from
    disk or a string, normalises whitespace (including C1 control characters
    such as U+0097 which are invisible but would otherwise attach to tokens),
    splits on blank lines into paragraphs, and stores the result as a list of
    (character, label) pairs per paragraph.  Also provides character-level
    feature extraction (space_before, capitalized, numeric, dictionary look-up)
    used as auxiliary input to the neural model.

DataLoader(TokenizationDataset)
    Training-time subclass.  Builds a vocabulary, wraps the data into fixed-
    length windows suitable for batched GPU training, and applies a suite of
    data augmentation techniques (described below) to improve generalisation.

SortedDataset(Dataset)
    Thin wrapper around TokenizationDataset that sorts paragraphs by length and
    exposes a collate() method so that a torch DataLoader can efficiently batch
    them with minimal padding during evaluation.

--------------------------------------------------------------------------------
Training augmentations
--------------------------------------------------------------------------------

All augmentations are applied stochastically at batch-construction time inside
DataLoader.next() / strings_starting(), so the model sees a different
perturbation of each sentence on each training pass.  Each is gated by a
probability argument (default 0.0, i.e. off unless explicitly enabled).

move_last_char  (last_char_move_prob)
    Moves the sentence-final punctuation character one position to the right,
    inserting a space before it.  Teaches the tokenizer that a space-separated
    sentence-final punct still ends the sentence, which is useful for languages
    where the training data always has the punct attached.

move_punct_back  (punct_move_back_prob)
    The inverse of move_last_char: removes the space before a punctuation character
    that appears space-separated from the preceding word anywhere in the sentence.
    Teaches the tokenizer that punctuation can appear attached to the preceding
    word, which is important for languages such as Vietnamese where the dataset
    may always space-separate them.  The eligible punctuation set is determined
    automatically from the training data via build_move_punct_set().

split_mwt  (split_mwt_prob)
    Randomly selects an MWT in a sentence and replaces it with the two
    space-separated words it expands to (both labelled as ordinary word ends).
    Teaches the tokenizer not to over-generate MWT labels when the constituent
    words appear separately.  The eligible MWT set is determined from the
    training data and the MWT expansion dictionary via build_known_mwt().

augment_final_punct  (augment_final_punct_prob)
    Replaces the sentence-final punctuation character with a typographic
    variant (e.g. ASCII '?' <-> fullwidth '？').  Useful for datasets that use
    only one form, so that the model generalises to the other.  Eligible pairs
    are checked against the vocabulary via augment_vocab() to ensure the source
    exists and the target is absent before activating the substitution.

augment_mid_sent_punct  (augment_mid_punct_prob)
    Replaces a mid-sentence punctuation character (currently comma) with a
    typographic alternative (en dash U+2013 or em dash U+2014).  Intended for
    languages whose training data contains no dashes, so that the model learns
    to tokenize them correctly without retraining from scratch.  Unlike
    augment_final_punct, the substitution also randomly varies the surrounding
    whitespace, producing all four spacing styles (spaced both sides, attached
    left, attached right, attached both sides) with equal probability, since
    dashes appear in real text with all of these conventions.  Eligible pairs
    are checked via augment_vocab(..., final=False).

drop_last_char  (last_char_drop_prob)
    Drops the final character of a training window with some probability,
    relabelling the new final character as a sentence end.  Teaches the model
    to handle documents that end without sentence-final punctuation.

sent_drop  (sent_drop_prob)
    Drops a suffix of the concatenated sentences in a training window.  Also
    teaches the model to handle incomplete input.
"""

from bisect import bisect_right
from collections import defaultdict
from copy import copy
import numpy as np
import random
import logging
import re
import torch
from torch.utils.data import Dataset

from stanza.models.common.utils import sort_with_indices, unsort
from stanza.models.tokenization.vocab import Vocab

logger = logging.getLogger('stanza')

def filter_consecutive_whitespaces(para):
    filtered = []
    for i, (char, label) in enumerate(para):
        if i > 0:
            if char == ' ' and para[i-1][0] == ' ':
                continue

        filtered.append((char, label))

    return filtered

# control characters not covered by \s, but still not part of normal text
# for example, U+0097 was reported as being stuck on a token in this issue:
# https://github.com/stanfordnlp/stanza/issues/1257
NEWLINE_WHITESPACE_RE = re.compile(r'\n[\s\u0080-\u009f]*\n')
# this was (r'^([\d]+[,\.]*)+$')
# but the runtime on that can explode exponentially
# for example, on 111111111111111111111111a
NUMERIC_RE = re.compile(r'^[\d]+([,\.]+[\d]+)*[,\.]*$')
WHITESPACE_RE = re.compile(r'[\s\u0080-\u009f]')

class TokenizationDataset:
    def __init__(self, tokenizer_args, input_files={'txt': None, 'label': None}, input_text=None, vocab=None, evaluation=False, dictionary=None, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        self.args = tokenizer_args
        self.eval = evaluation
        self.dictionary = dictionary
        self.vocab = vocab

        # get input files
        txt_file = input_files['txt']
        label_file = input_files['label']

        # Load data and process it
        # set up text from file or input string
        assert txt_file is not None or input_text is not None
        if input_text is None:
            with open(txt_file, encoding="utf-8") as f:
                text = ''.join(f.readlines()).rstrip()
        else:
            text = input_text

        text_chunks = NEWLINE_WHITESPACE_RE.split(text)
        text_chunks = [pt.rstrip() for pt in text_chunks]
        text_chunks = [pt for pt in text_chunks if pt]
        if label_file is not None:
            with open(label_file, encoding="utf-8") as f:
                labels = ''.join(f.readlines()).rstrip()
                labels = NEWLINE_WHITESPACE_RE.split(labels)
                labels = [pt.rstrip() for pt in labels]
                labels = [map(int, pt) for pt in labels if pt]
        else:
            labels = [[0 for _ in pt] for pt in text_chunks]

        skip_newline = self.args.get('skip_newline', False)
        self.data = [[(WHITESPACE_RE.sub(' ', char), label) # substitute special whitespaces
                      for char, label in zip(pt, pc) if not (skip_newline and char == '\n')] # check if newline needs to be eaten
                     for pt, pc in zip(text_chunks, labels)]

        # remove consecutive whitespaces
        self.data = [filter_consecutive_whitespaces(x) for x in self.data]

    def labels(self):
        """
        Returns a list of the labels for all of the sentences in this DataLoader

        Used at eval time to compare to the results, for example
        """
        return [np.array(list(x[1] for x in sent)) for sent in self.data]

    def extract_dict_feat(self, para, idx):
        """
        This function is to extract dictionary features for each character
        """
        length = len(para)

        dict_forward_feats = [0 for i in range(self.args['num_dict_feat'])]
        dict_backward_feats = [0 for i in range(self.args['num_dict_feat'])]
        forward_word = para[idx][0]
        backward_word = para[idx][0]
        prefix = True
        suffix = True
        for window in range(1,self.args['num_dict_feat']+1):
            # concatenate each character and check if words found in dict not, stop if prefix not found
            #check if idx+t is out of bound and if the prefix is already not found
            if (idx + window) <= length-1 and prefix:
                forward_word += para[idx+window][0].lower()
                #check in json file if the word is present as prefix or word or None.
                feat = 1 if forward_word in self.dictionary["words"] else 0
                #if the return value is not 2 or 3 then the checking word is not a valid word in dict.
                dict_forward_feats[window-1] = feat
                #if the dict return 0 means no prefixes found, thus, stop looking for forward.
                if forward_word not in self.dictionary["prefixes"]:
                    prefix = False
            #backward check: similar to forward
            if (idx - window) >= 0 and suffix:
                backward_word = para[idx-window][0].lower() + backward_word
                feat = 1 if backward_word in self.dictionary["words"] else 0
                dict_backward_feats[window-1] = feat
                if backward_word not in self.dictionary["suffixes"]:
                    suffix = False
            #if cannot find both prefix and suffix, then exit the loop
            if not prefix and not suffix:
                break

        return dict_forward_feats + dict_backward_feats

    def para_to_sentences(self, para):
        """ Convert a paragraph to a list of processed sentences. """
        res = []
        funcs = []
        for feat_func in self.args['feat_funcs']:
            if feat_func == 'end_of_para' or feat_func == 'start_of_para':
                # skip for position-dependent features
                continue
            if feat_func == 'space_before':
                func = lambda x: 1 if x.startswith(' ') else 0
            elif feat_func == 'capitalized':
                func = lambda x: 1 if x[0].isupper() else 0
            elif feat_func == 'numeric':
                func = lambda x: 1 if (NUMERIC_RE.match(x) is not None) else 0
            else:
                raise ValueError('Feature function "{}" is undefined.'.format(feat_func))

            funcs.append(func)

        # stacking all featurize functions
        composite_func = lambda x: [f(x) for f in funcs]

        def process_sentence(sent_units, sent_labels, sent_feats):
            return (np.array([self.vocab.unit2id(y) for y in sent_units]),
                    np.array(sent_labels),
                    np.array(sent_feats),
                    list(sent_units))

        use_end_of_para = 'end_of_para' in self.args['feat_funcs']
        use_start_of_para = 'start_of_para' in self.args['feat_funcs']
        use_dictionary = self.args['use_dictionary']
        current_units = []
        current_labels = []
        current_feats = []
        for i, (unit, label) in enumerate(para):
            feats = composite_func(unit)
            # position-dependent features
            if use_end_of_para:
                f = 1 if i == len(para)-1 else 0
                feats.append(f)
            if use_start_of_para:
                f = 1 if i == 0 else 0
                feats.append(f)

            #if dictionary feature is selected
            if use_dictionary:
                dict_feats = self.extract_dict_feat(para, i)
                feats = feats + dict_feats

            current_units.append(unit)
            current_labels.append(label)
            current_feats.append(feats)
            if not self.eval and (label == 2 or label == 4): # end of sentence
                if len(current_units) <= self.args['max_seqlen']:
                    # get rid of sentences that are too long during training of the tokenizer
                    res.append(process_sentence(current_units, current_labels, current_feats))
                current_units.clear()
                current_labels.clear()
                current_feats.clear()

        if len(current_units) > 0:
            if self.eval or len(current_units) <= self.args['max_seqlen']:
                res.append(process_sentence(current_units, current_labels, current_feats))

        return res

    def advance_old_batch(self, eval_offsets, old_batch):
        """
        Advance to a new position in a batch where we have partially processed the batch

        If we have previously built a batch of data and made predictions on them, then when we are trying to make
        prediction on later characters in those paragraphs, we can avoid rebuilding the converted data from scratch
        and just (essentially) advance the indices/offsets from where we read converted data in this old batch.
        In this case, eval_offsets index within the old_batch to advance the strings to process.
        """
        unkid = self.vocab.unit2id('<UNK>')
        padid = self.vocab.unit2id('<PAD>')

        ounits, olabels, ofeatures, oraw = old_batch
        feat_size = ofeatures.shape[-1]
        lens = (ounits != padid).sum(1).tolist()
        pad_len = max(l-i for i, l in zip(eval_offsets, lens))

        units = torch.full((len(ounits), pad_len), padid, dtype=torch.int64)
        labels = torch.full((len(ounits), pad_len), -1, dtype=torch.int32)
        features = torch.zeros((len(ounits), pad_len, feat_size), dtype=torch.float32)
        raw_units = []

        for i in range(len(ounits)):
            eval_offsets[i] = min(eval_offsets[i], lens[i])
            units[i, :(lens[i] - eval_offsets[i])] = ounits[i, eval_offsets[i]:lens[i]]
            labels[i, :(lens[i] - eval_offsets[i])] = olabels[i, eval_offsets[i]:lens[i]]
            features[i, :(lens[i] - eval_offsets[i])] = ofeatures[i, eval_offsets[i]:lens[i]]
            raw_units.append(oraw[i][eval_offsets[i]:lens[i]] + ['<PAD>'] * (pad_len - lens[i] + eval_offsets[i]))

        return units, labels, features, raw_units

def build_move_punct_set(data, move_back_prob):
    move_punct = {',', ':', '!', '.', '?', '"', '(', ')'}
    for chunk in data:
        # ignore positions at the start and end of a chunk
        for idx in range(1, len(chunk)-1):
            if chunk[idx][0] not in move_punct:
                continue
            if chunk[idx][1] == 0:
                if chunk[idx+1][0].isspace() and not chunk[idx-1][0].isdigit():
                    # this check removes punct which isn't ending a word...
                    # honestly that's a rather unusual situation
                    # VI has |3, 5| as a complete token
                    # so we also eliminate isdigit()
                    move_punct.remove(chunk[idx][0])
                continue
            # we skip isdigit() because we will intentionally not
            # create things that look like decimal numbers
            if not chunk[idx-1][0].isspace() and chunk[idx-1][0] not in move_punct and not chunk[idx-1][0].isdigit():
                # this check eliminates things like '.' after 'Mr.'
                move_punct.remove(chunk[idx][0])
                continue
    return move_punct

def build_known_mwt(data, mwt_expansions):
    known_mwts = set()
    for chunk in data:
        for idx, unit in enumerate(chunk):
            if unit[1] != 3:
                continue
            # found an MWT
            prev_idx = idx - 1
            while prev_idx >= 0 and chunk[prev_idx][1] == 0:
                prev_idx -= 1
            prev_idx += 1
            while chunk[prev_idx][0].isspace():
                prev_idx += 1
            if prev_idx == idx:
                continue
            mwt = "".join(x[0] for x in chunk[prev_idx:idx+1])
            if mwt not in mwt_expansions:
                continue
            if len(mwt_expansions[mwt]) > 2:
                # TODO: could split 3 word tokens as well
                continue
            known_mwts.add(mwt)
    return known_mwts


# Pairs of (existing, replacement) for mid-sentence punctuation augmentation.
# The existing character must appear in the training data and the replacement
# must not, otherwise the pair is skipped (checked in augment_vocab).
MID_SENT_AUGMENT_PAIRS = [
    (",", "\u2013"),   # comma -> en dash
    (",", "\u2014"),   # comma -> em dash
]


class DataLoader(TokenizationDataset):
    """
    This is the training version of the dataset.
    """
    def __init__(self, args, input_files={'txt': None, 'label': None}, input_text=None, vocab=None, evaluation=False, dictionary=None, mwt_expansions=None):
        super().__init__(args, input_files, input_text, vocab, evaluation, dictionary)

        self.vocab = vocab if vocab is not None else self.init_vocab()

        # data comes in a list of paragraphs, where each paragraph is a list of units with unit-level labels.
        # At evaluation time, each paragraph is treated as single "sentence" as we don't know a priori where
        # sentence breaks occur. We make prediction from left to right for each paragraph and move forward to
        # the last predicted sentence break to start afresh.
        self.sentences = [self.para_to_sentences(para) for para in self.data]

        self.init_sent_ids()
        logger.debug(f"{len(self.sentence_ids)} sentences loaded.")

        punct_move_back_prob = args.get('punct_move_back_prob', 0.0)
        if punct_move_back_prob > 0.0:
            self.move_punct = build_move_punct_set(self.data, punct_move_back_prob)
            if len(self.move_punct) > 0:
                logger.debug('Based on the training data, will augment space/punct combinations {}'.format(self.move_punct))
            else:
                logger.debug('Based on the training data, no punct are eligible to be rearranged with extra whitespace')

        split_mwt_prob = args.get('split_mwt_prob', 0.0)
        if split_mwt_prob > 0.0 and not evaluation:
            self.mwt_expansions = mwt_expansions
            self.known_mwt = build_known_mwt(self.data, mwt_expansions)
            if len(self.known_mwt) > 0:
                logger.debug('Based on the training data, there are %d MWT which might be split at training time', len(self.known_mwt))
            else:
                logger.debug('Based on the training data, there are NO MWT to split at training time')

        augment_final_punct_prob = 0.0 if evaluation else args.get('augment_final_punct_prob', 0.0)
        if augment_final_punct_prob > 0:
            self.augmentations = defaultdict(list)
            AUGMENT_PAIRS = [("?", "？"),
                             ("?", "︖"),
                             ("?", "﹖"),
                             ("?", "⁇"),
                             ("!", "！"),
                             ("!", "︕"),
                             ("!", "﹗"),
                             ("!", "‼"),]
            for orig, target in AUGMENT_PAIRS:
                if self.augment_vocab(self.vocab, self.data, orig, target):
                    logger.debug('Based on the training data, augmenting final |%s| to |%s|' % (orig, target))
                    self.augmentations[orig].append(target)
                if self.augment_vocab(self.vocab, self.data, target, orig):
                    logger.debug('Based on the training data, augmenting final |%s| to |%s|' % (target, orig))
                    self.augmentations[target].append(orig)

        augment_mid_punct_prob = 0.0 if evaluation else args.get('augment_mid_punct_prob', 0.0)
        if augment_mid_punct_prob > 0.0:
            self.mid_sent_augmentations = self.build_mid_sent_augmentations(self.vocab, self.data, MID_SENT_AUGMENT_PAIRS)

    def __len__(self):
        return len(self.sentence_ids)

    def init_vocab(self):
        vocab = Vocab(self.data, self.args['lang'])
        return vocab

    @staticmethod
    def augment_vocab(vocab, data, existing_unit, new_unit, final=True):
        if existing_unit not in vocab:
            return False
        new_unit_count = 0
        existing_unit_count = 0
        for sentence in data:
            if final:
                units = [sentence[-1][0]]
            else:
                units = [x[0] for x in sentence]
            for unit in units:
                if unit == new_unit:
                    new_unit_count += 1
                elif unit == existing_unit:
                    existing_unit_count += 1
        if existing_unit_count == 0:
            return False
        if new_unit_count > 0:
            return False
        if new_unit not in vocab:
            vocab.append(new_unit)
        logger.debug("Found %d |%s| and %d |%s|", new_unit_count, new_unit, existing_unit_count, existing_unit)
        return True

    @staticmethod
    def build_mid_sent_augmentations(vocab, data, pairs):
        """
        For each (existing, replacement) pair, check whether the substitution
        is appropriate for this dataset (existing present, replacement absent)
        using the same augment_vocab logic as augment_final_punct.  Returns a
        dict mapping each source character to a list of valid replacement
        characters.
        """
        augmentations = defaultdict(list)
        for orig, target in pairs:
            if DataLoader.augment_vocab(vocab, data, orig, target, final=False):
                logger.debug('Mid-sentence augmentation: will substitute |%s| with |%s|', orig, target)
                augmentations[orig].append(target)
        return augmentations

    def init_sent_ids(self):
        self.sentence_ids = []
        self.cumlen = [0]
        for i, para in enumerate(self.sentences):
            for j in range(len(para)):
                self.sentence_ids += [(i, j)]
                self.cumlen += [self.cumlen[-1] + len(self.sentences[i][j][0])]

    def has_mwt(self):
        # presumably this only needs to be called either 0 or 1 times,
        # 1 when training and 0 any other time, so no effort is put
        # into caching the result
        for sentence in self.data:
            for word in sentence:
                if word[1] > 2:
                    return True
        return False

    def shuffle(self):
        for para in self.sentences:
            random.shuffle(para)
        self.init_sent_ids()

    def move_last_char(self, sentence):
        if len(sentence[3]) > 1 and len(sentence[3]) < self.args['max_seqlen'] and sentence[1][-1] == 2 and sentence[1][-2] != 0:
            new_units = [(x, int(y)) for x, y in zip(sentence[3][:-1], sentence[1][:-1])]
            new_units.extend([(' ', 0), (sentence[3][-1], int(sentence[1][-1]))])
            encoded = self.para_to_sentences(new_units)
            return encoded
        return None

    def split_mwt(self, sentence):
        if len(sentence[3]) <= 1 or len(sentence[3]) >= self.args['max_seqlen']:
            return None

        # if we find a token in the sentence which ends with label 3,
        # eg it is an MWT,
        # with some probability we split it into two tokens
        # and treat the split tokens as both label 1 instead of 3
        # in this manner, we teach the tokenizer not to treat the
        # entire sequence of characters with added spaces as an MWT,
        # which weirdly can happen in some corner cases

        mwt_ends = [idx for idx, label in enumerate(sentence[1]) if label == 3]
        if len(mwt_ends) == 0:
            return None
        random_end = random.randint(0, len(mwt_ends)-1)
        mwt_end = mwt_ends[random_end]
        mwt_start = mwt_end - 1
        while mwt_start >= 0 and sentence[1][mwt_start] == 0:
            mwt_start -= 1
        mwt_start += 1
        while sentence[3][mwt_start].isspace():
            mwt_start += 1
        if mwt_start == mwt_end:
            return None
        mwt = "".join(x for x in sentence[3][mwt_start:mwt_end+1])
        if mwt not in self.mwt_expansions:
            return None

        all_units = [(x, int(y)) for x, y in zip(sentence[3], sentence[1])]
        w0_units = [(x, 0) for x in self.mwt_expansions[mwt][0]]
        w0_units[-1] = (w0_units[-1][0], 1)
        w1_units = [(x, 0) for x in self.mwt_expansions[mwt][1]]
        w1_units[-1] = (w1_units[-1][0], 1)
        split_units = w0_units + [(' ', 0)] + w1_units
        new_units = all_units[:mwt_start] + split_units + all_units[mwt_end+1:]
        encoded = self.para_to_sentences(new_units)
        if len(encoded) == 0:
            # this can happen if the MWT split to be too long for max_seqlen
            return None
        return encoded

    def move_punct_back(self, sentence):
        if len(sentence[3]) <= 1 or len(sentence[3]) >= self.args['max_seqlen']:
            return None

        # check that we are not accidentally creating decimal numbers
        #   idx == 1 or not sentence[3][idx-2].isdigit()
        # one disadvantage of checking for sentence[1][idx] == 0
        #   would be that tokens of all punct, such as '...',
        #   should move but would not move if this is eliminated
        commas = [idx for idx, c in enumerate(sentence[3])
                  if c in self.move_punct and idx > 0 and sentence[3][idx-1].isspace() and (idx == 1 or not sentence[3][idx-2].isdigit())]
        if len(commas) == 0:
            return None

        all_units = [(x, int(y)) for x, y in zip(sentence[3], sentence[1])]
        new_units = []

        span_start = 0
        for span_end in commas:
            new_units.extend(all_units[span_start:span_end-1])
            span_start = span_end
        if span_end < len(sentence[3]):
            new_units.extend(all_units[span_end:])

        encoded = self.para_to_sentences(new_units)
        return encoded

    def augment_mid_sent_punct(self, sentence):
        """
        With some probability (controlled externally by the caller), replace
        one mid-sentence punctuation character with an augmented alternative
        drawn from self.mid_sent_augmentations.

        Because en/em dashes appear in text with varying spacing conventions,
        the substitution also randomly adjusts the spaces immediately
        surrounding the replacement character:

          original (comma style):  "A, B"   -> chars: A , ' ' B
          possible outputs:
            spaced both sides:     "A – B"  -> A ' ' – ' ' B
            attached left:         "A– B"   -> A     – ' ' B
            attached right:        "A –B"   -> A ' ' –     B
            attached both sides:   "A–B"    -> A     –     B

        Returns a re-encoded sentence list, or None if no eligible position
        was found.
        """
        if not hasattr(self, 'mid_sent_augmentations') or not self.mid_sent_augmentations:
            return None

        if len(sentence[3]) <= 2 or len(sentence[3]) >= self.args['max_seqlen'] - 2:
            return None

        eligible = [
            idx for idx, (char, label) in enumerate(zip(sentence[3], sentence[1]))
            if char in self.mid_sent_augmentations
            and 0 < idx < len(sentence[3]) - 1
            and label != 0
            and sentence[1][idx - 1] != 0
        ]

        if not eligible:
            return None

        idx = random.choice(eligible)
        orig_char = sentence[3][idx]
        new_char = random.choice(self.mid_sent_augmentations[orig_char])

        all_units = [(x, int(y)) for x, y in zip(sentence[3], sentence[1])]

        # Replace the character itself.
        new_units = list(all_units)
        new_units[idx] = (new_char, all_units[idx][1])

        # Determine current spacing around the punctuation character.
        has_space_before = idx > 0 and all_units[idx - 1][0] == ' '
        has_space_after  = idx < len(all_units) - 1 and all_units[idx + 1][0] == ' '

        # Four spacing styles, chosen uniformly at random:
        #   0: spaced both sides  "A – B"
        #   1: attached left      "A– B"   (drop space before)
        #   2: attached right     "A –B"   (drop space after)
        #   3: attached both      "A–B"
        spacing_style = random.randint(0, 3)

        want_space_before = spacing_style in (0, 2)
        want_space_after  = spacing_style in (0, 1)

        # Adjust space before the dash.
        if has_space_before and not want_space_before:
            del new_units[idx - 1]
            idx -= 1   # keep idx pointing at the dash after deletion
        elif not has_space_before and want_space_before:
            new_units.insert(idx, (' ', 0))
            idx += 1

        # Adjust space after the dash (idx still points at the dash).
        if has_space_after and not want_space_after:
            del new_units[idx + 1]
        elif not has_space_after and want_space_after:
            new_units.insert(idx + 1, (' ', 0))

        encoded = self.para_to_sentences(new_units)
        if not encoded:
            return None
        return encoded

    def augment_final_punct(self, sentence):
        if len(sentence[3]) > 1 and len(sentence[3]) < self.args['max_seqlen']:
            if sentence[3][-1] in self.augmentations:
                augmented = random.choice(self.augmentations[sentence[3][-1]])
                new_units = [(x, int(y)) for x, y in zip(sentence[3][:-1], sentence[1][:-1])]
                new_units.append((augmented, sentence[1][-1]))
            else:
                return None
            encoded = self.para_to_sentences(new_units)
            return encoded
        return None


    def next(self, eval_offsets=None, unit_dropout=0.0, feat_unit_dropout=0.0):
        ''' Get a batch of converted and padded PyTorch data from preprocessed raw text for training/prediction. '''
        feat_size = len(self.sentences[0][0][2][0])
        unkid = self.vocab.unit2id('<UNK>')
        padid = self.vocab.unit2id('<PAD>')

        def strings_starting(id_pair, offset=0, pad_len=self.args['max_seqlen']):
            # At eval time, this combines sentences in paragraph (indexed by id_pair[0]) starting sentence (indexed 
            # by id_pair[1]) into a long string for evaluation. At training time, we just select random sentences
            # from the entire dataset until we reach max_seqlen.
            drop_sents = False if self.eval or (self.args.get('sent_drop_prob', 0) == 0) else (random.random() < self.args.get('sent_drop_prob', 0))
            drop_last_char = False if self.eval or (self.args.get('last_char_drop_prob', 0) == 0) else (random.random() < self.args.get('last_char_drop_prob', 0))
            move_last_char_prob = 0.0 if self.eval else self.args.get('last_char_move_prob', 0.0)
            move_punct_back_prob = 0.0 if self.eval else self.args.get('punct_move_back_prob', 0.0)
            split_mwt_prob = 0.0 if self.eval else self.args.get('split_mwt_prob', 0.0)
            augment_mid_punct_prob = 0.0 if self.eval else self.args.get('augment_mid_punct_prob', 0.0)
            augment_final_punct_prob = 0.0 if self.eval else self.args.get('augment_final_punct_prob', 0.0)

            pid, sid = id_pair if self.eval else random.choice(self.sentence_ids)
            sentences = [copy([x[offset:] for x in self.sentences[pid][sid]])]
            total_len = len(sentences[0][0])

            assert self.eval or total_len <= self.args['max_seqlen'], 'The maximum sequence length {} is less than that of the longest sentence length ({}) in the data, consider increasing it! {}'.format(self.args['max_seqlen'], total_len, ' '.join(["{}/{}".format(*x) for x in zip(self.sentences[pid][sid])]))
            if self.eval:
                for sid1 in range(sid+1, len(self.sentences[pid])):
                    total_len += len(self.sentences[pid][sid1][0])
                    sentences.append(self.sentences[pid][sid1])

                    if total_len >= self.args['max_seqlen']:
                        break
            else:
                while True:
                    pid1, sid1 = random.choice(self.sentence_ids)
                    total_len += len(self.sentences[pid1][sid1][0])
                    sentences.append(self.sentences[pid1][sid1])

                    if total_len >= self.args['max_seqlen']:
                        break

            if move_last_char_prob > 0.0:
                for sentence_idx, sentence in enumerate(sentences):
                    if random.random() < move_last_char_prob:
                        # the sentence might not be eligible, such as
                        # already having a space or not having a sentence final punct,
                        # so we need to do a two step checking process here
                        new_sentence = self.move_last_char(sentence)
                        if new_sentence is not None:
                            sentences[sentence_idx] = new_sentence[0]
                            total_len += 1

            if move_punct_back_prob > 0.0:
                for sentence_idx, sentence in enumerate(sentences):
                    if random.random() < move_punct_back_prob:
                        # the sentence might not be eligible, such as
                        # not having a space separated punct,
                        # so we need to do a two step checking process here
                        new_sentence = self.move_punct_back(sentence)
                        if new_sentence is not None:
                            total_len = total_len + len(new_sentence[0][3]) - len(sentences[sentence_idx][3])
                            sentences[sentence_idx] = new_sentence[0]

            if split_mwt_prob > 0.0:
                for sentence_idx, sentence in enumerate(sentences):
                    if random.random() < split_mwt_prob:
                        new_sentence = self.split_mwt(sentence)
                        if new_sentence is not None:
                            total_len = total_len + len(new_sentence[0][3]) - len(sentences[sentence_idx][3])
                            sentences[sentence_idx] = new_sentence[0]

            if augment_final_punct_prob > 0.0:
                for sentence_idx, sentence in enumerate(sentences):
                    if random.random() < augment_final_punct_prob:
                        new_sentence = self.augment_final_punct(sentence)
                        if new_sentence is not None:
                            sentences[sentence_idx] = new_sentence[0]

            if augment_mid_punct_prob > 0.0:
                for sentence_idx, sentence in enumerate(sentences):
                    if random.random() < augment_mid_punct_prob:
                        new_sentence = self.augment_mid_sent_punct(sentence)
                        if new_sentence is not None:
                            sentences[sentence_idx] = new_sentence[0]

            if drop_sents and len(sentences) > 1:
                if total_len > self.args['max_seqlen']:
                    sentences = sentences[:-1]
                if len(sentences) > 1:
                    p = [.5 ** i for i in range(1, len(sentences) + 1)] # drop a large number of sentences with smaller probability
                    cutoff = random.choices(list(range(len(sentences))), weights=list(reversed(p)))[0]
                    sentences = sentences[:cutoff+1]

            units = np.concatenate([s[0] for s in sentences])
            labels = np.concatenate([s[1] for s in sentences])
            feats = np.concatenate([s[2] for s in sentences])
            raw_units = [x for s in sentences for x in s[3]]

            if not self.eval:
                cutoff = self.args['max_seqlen']
                units, labels, feats, raw_units = units[:cutoff], labels[:cutoff], feats[:cutoff], raw_units[:cutoff]

            if drop_last_char:  # can only happen in non-eval mode
                if len(labels) > 1 and labels[-1] == 2 and labels[-2] in (1, 3):
                    # training text ended with a sentence end position
                    # and that word was a single character
                    # and the previous character ended the word
                    units, labels, feats, raw_units = units[:-1], labels[:-1], feats[:-1], raw_units[:-1]
                    # word end -> sentence end, mwt end -> sentence mwt end
                    labels[-1] = labels[-1] + 1

            return units, labels, feats, raw_units

        if eval_offsets is not None:
            # find max padding length
            pad_len = 0
            for eval_offset in eval_offsets:
                if eval_offset < self.cumlen[-1]:
                    pair_id = bisect_right(self.cumlen, eval_offset) - 1
                    pair = self.sentence_ids[pair_id]
                    pad_len = max(pad_len, len(strings_starting(pair, offset=eval_offset-self.cumlen[pair_id])[0]))

            pad_len += 1
            id_pairs = [bisect_right(self.cumlen, eval_offset) - 1 for eval_offset in eval_offsets]
            pairs = [self.sentence_ids[pair_id] for pair_id in id_pairs]
            offsets = [eval_offset - self.cumlen[pair_id] for eval_offset, pair_id in zip(eval_offsets, id_pairs)]

            offsets_pairs = list(zip(offsets, pairs))
        else:
            id_pairs = random.sample(self.sentence_ids, min(len(self.sentence_ids), self.args['batch_size']))
            offsets_pairs = [(0, x) for x in id_pairs]
            pad_len = self.args['max_seqlen']

        # put everything into padded and nicely shaped NumPy arrays and eventually convert to PyTorch tensors
        units = np.full((len(id_pairs), pad_len), padid, dtype=np.int64)
        labels = np.full((len(id_pairs), pad_len), -1, dtype=np.int64)
        features = np.zeros((len(id_pairs), pad_len, feat_size), dtype=np.float32)
        raw_units = []
        for i, (offset, pair) in enumerate(offsets_pairs):
            u_, l_, f_, r_ = strings_starting(pair, offset=offset, pad_len=pad_len)
            units[i, :len(u_)] = u_
            labels[i, :len(l_)] = l_
            features[i, :len(f_), :] = f_
            raw_units.append(r_ + ['<PAD>'] * (pad_len - len(r_)))

        if unit_dropout > 0 and not self.eval:
            # dropout characters/units at training time and replace them with UNKs
            mask = np.random.random_sample(units.shape) < unit_dropout
            mask[units == padid] = 0
            units[mask] = unkid
            for i in range(len(raw_units)):
                for j in range(len(raw_units[i])):
                    if mask[i, j]:
                        raw_units[i][j] = '<UNK>'

        # dropout unit feature vector in addition to only torch.dropout in the model.
        # experiments showed that only torch.dropout hurts the model
        # we believe it is because the dict feature vector is mostly scarse so it makes
        # more sense to drop out the whole vector instead of only single element.
        if self.args['use_dictionary'] and feat_unit_dropout > 0 and not self.eval:
            mask_feat = np.random.random_sample(units.shape) < feat_unit_dropout
            mask_feat[units == padid] = 0
            for i in range(len(raw_units)):
                for j in range(len(raw_units[i])):
                    if mask_feat[i,j]:
                        features[i,j,:] = 0
                        
        units = torch.from_numpy(units)
        labels = torch.from_numpy(labels)
        features = torch.from_numpy(features)

        return units, labels, features, raw_units

class SortedDataset(Dataset):
    """
    Holds a TokenizationDataset for use in a torch DataLoader

    The torch DataLoader is different from the DataLoader defined here
    and allows for cpu & gpu parallelism.  Updating output_predictions
    to use this class as a wrapper to a TokenizationDataset means the
    calculation of features can happen in parallel, saving quite a
    bit of time.
    """
    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset
        self.data, self.indices = sort_with_indices(self.dataset.data, key=len, reverse=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # This will return a single sample
        #   np: index in character map
        #   np: tokenization label
        #   np: features
        #   list: original text as one length strings
        return self.dataset.para_to_sentences(self.data[index])

    def unsort(self, arr):
        return unsort(arr, self.indices)

    def collate(self, samples):
        if any(len(x) > 1 for x in samples):
            raise ValueError("Expected all paragraphs to have no preset sentence splits!")
        feat_size = samples[0][0][2].shape[-1]
        padid = self.dataset.vocab.unit2id('<PAD>')

        # +1 so that all samples end with at least one pad
        pad_len = max(len(x[0][3]) for x in samples) + 1

        units = torch.full((len(samples), pad_len), padid, dtype=torch.int64)
        labels = torch.full((len(samples), pad_len), -1, dtype=torch.int32)
        features = torch.zeros((len(samples), pad_len, feat_size), dtype=torch.float32)
        raw_units = []
        for i, sample in enumerate(samples):
            u_, l_, f_, r_ = sample[0]
            units[i, :len(u_)] = torch.from_numpy(u_)
            labels[i, :len(l_)] = torch.from_numpy(l_)
            features[i, :len(f_), :] = torch.from_numpy(f_)
            raw_units.append(r_ + ['<PAD>'])

        return units, labels, features, raw_units

