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


def should_augment_nopunct_predicate(sentence):
    last_word = sentence[-1]
    return last_word[UPOS] == 'PUNCT'

def can_augment_nopunct_predicate(sentence):
    """
    Check that the sentence ends with PUNCT and also doesn't have any words which depend on the last word
    """
    last_word = sentence[-1]
    if last_word[UPOS] != 'PUNCT':
        return False
    # don't cut off MWT
    if len(last_word[ID]) > 1:
        return False
    if any(len(word[ID]) == 1 and word[HEAD] == last_word[ID][0] for word in sentence):
        return False
    return True

def augment_punct(train_data, augment_ratio,
                  should_augment_predicate=should_augment_nopunct_predicate,
                  can_augment_predicate=can_augment_nopunct_predicate,
                  keep_original_sentences=True):

    """
    Adds extra training data to compensate for some models having all sentences end with PUNCT

    Some of the models (for example, UD_Hebrew-HTB) have the flaw that
    all of the training sentences end with PUNCT.  The model therefore
    learns to finish every sentence with punctuation, even if it is
    given a sentence with non-punct at the end.

    One simple way to fix this is to train on some fraction of training data with punct.

    Params:
    train_data: list of list of dicts, eg a conll doc
    augment_ratio: the fraction to augment.  if None, a best guess is made to get to 10%

    should_augment_predicate: a function which returns T/F if a sentence already ends with not PUNCT
    can_augment_predicate: a function which returns T/F if it makes sense to remove the last PUNCT

    TODO: do this dynamically, as part of the DataLoader or elsewhere?
    One complication is the data comes back from the DataLoader as
    tensors & indices, so it is much more complicated to manipulate
    """
    if len(train_data) == 0:
        return []

    if augment_ratio is None:
        augment_ratio = get_augment_ratio(train_data, should_augment_predicate, can_augment_predicate)

    if augment_ratio <= 0:
        if keep_original_sentences:
            return list(train_data)
        else:
            return []

    new_data = []
    for sentence in train_data:
        if can_augment_predicate(sentence):
            if random.random() < augment_ratio and len(sentence) > 1:
                # todo: could deep copy the words
                #       or not deep copy any of this
                new_sentence = list(sentence[:-1])
                new_data.append(new_sentence)
            elif keep_original_sentences:
                new_data.append(new_sentence)

    return new_data
