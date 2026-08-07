"""
Utility functions for data transformations.
"""

import logging
import random

import torch

import stanza.models.common.seq2seq_constant as constant
from stanza.models.common.doc import HEAD, ID, UPOS

logger = logging.getLogger('stanza')

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_long_tensor(tokens_list, batch_size, pad_id=constant.PAD_ID):
    """ Convert (list of )+ tokens to a padded LongTensor. """
    sizes = []
    x = tokens_list
    while isinstance(x[0], list):
        sizes.append(max(len(y) for y in x))
        x = [z for y in x for z in y]
    # TODO: pass in a device parameter and put it directly on the relevant device?
    # that might be faster than creating it and then moving it
    tokens = torch.LongTensor(batch_size, *sizes).fill_(pad_id)
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
    if batch == [[]]:
        return [[]], []
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def get_augment_ratio(train_data, should_augment_predicate, can_augment_predicate, desired_ratio=0.1, max_ratio=0.5):
    """
    Returns X so that if you randomly select X * N sentences, you get 10%

    The ratio will be chosen in the assumption that the final dataset
    is of size N rather than N + X * N.

    should_augment_predicate: returns True if the sentence has some
      feature which we may want to change occasionally.  for example,
      depparse sentences which end in punct
    can_augment_predicate: in the depparse sentences example, it is
      technically possible for the punct at the end to be the parent
      of some other word in the sentence.  in that case, the sentence
      should not be chosen.  should be at least as restrictive as
      should_augment_predicate
    """
    n_data = len(train_data)
    n_should_augment = sum(should_augment_predicate(sentence) for sentence in train_data)
    n_can_augment = sum(can_augment_predicate(sentence) for sentence in train_data)
    n_error = sum(can_augment_predicate(sentence) and not should_augment_predicate(sentence)
                  for sentence in train_data)
    if n_error > 0:
        raise AssertionError("can_augment_predicate allowed sentences not allowed by should_augment_predicate")

    if n_can_augment == 0:
        logger.warning("Found no sentences which matched can_augment_predicate {}".format(can_augment_predicate))
        return 0.0
    n_needed = n_data * desired_ratio - (n_data - n_should_augment)
    # if we want 10%, for example, and more than 10% already matches, we can skip
    if n_needed < 0:
        return 0.0
    ratio = n_needed / n_can_augment
    if ratio > max_ratio:
        return max_ratio
    return ratio


# Spanish/Catalan-style inverted question and exclamation marks. Some UD
# treebanks have every training sentence begin with one of these, which
# teaches a model to always expect the mark and misparse/mistag/mistokenize
# a sentence that's missing it. augment_initial_punct in
# prepare_tokenizer_treebank.py handles this at dataset-preparation time
# (currently for ¿ only); the POS tagger and dependency parser instead
# apply the equivalent augmentation dynamically, per sentence, inside their
# own Dataset.__getitem__ (see starts_with_initial_mark below). The
# tokenizer needs its own character-level version of this same check
# (see drop_initial_punct in stanza.models.tokenization.data), since it
# operates on individual characters before word boundaries exist, but
# imports this same tuple so the mark set itself is defined in one place.
INITIAL_INVERTED_PUNCT_MARKS = ('¿', '¡')

def starts_with_initial_mark(words, marks=INITIAL_INVERTED_PUNCT_MARKS):
    """
    True if the given list of word/token strings starts with one of the
    given marks, and no mark from the set (of any kind, not just the
    leading one) appears anywhere else in the list.

    The restriction mirrors augment_initial_punct in
    prepare_tokenizer_treebank.py, and exists to avoid ambiguity with
    nested or quoted questions/exclamations -- e.g. a sentence like
    '¿Dijo "¡hola!"?' has two candidate marks (one ¿ and one ¡) and isn't
    a case this augmentation should touch, even though neither mark is
    individually repeated.

    Shared by the POS tagger and dependency parser, whose sentences are
    both, at this point, plain lists of word strings -- the two models
    diverge only in how they physically drop the first word afterward
    (the parser also has to renumber head positions, which the tagger
    does not need to do).
    """
    if len(words) <= 1:
        return False
    first = words[0]
    if first not in marks:
        return False
    total_marks = sum(1 for w in words if w in marks)
    return total_marks == 1
