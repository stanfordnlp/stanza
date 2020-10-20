"""
Utility functions for data transformations.
"""

import logging
import random

import torch

import stanza.models.common.seq2seq_constant as constant

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


def augment_punct(train_data, augment_nopunct, sentence_nopunct_predicate, can_augment_nopunct_predicate):
    """
    Adds extra training data to compensate for some models having all sentences end with PUNCT

    Some of the models (for example, UD_Hebrew-HTB) have the flaw that
    all of the training sentences end with PUNCT.  The model therefore
    learns to finish every sentence with punctuation, even if it is
    given a sentence with non-punct at the end.

    One simple way to fix this is to train on some fraction of training data with punct.

    Params:
    train_data: list of list of dicts, eg a conll doc
    augment_nopunct: the fraction to augment.  if None, a best guess is made to get to 10%

    sentence_nopunct_predicate: a function which returns T/F if a sentence already ends with not PUNCT
    can_augment_nopunct_predicate: a function which returns T/F if it makes sense to remove the last PUNCT
    """
    if len(train_data) == 0:
        return train_data

    aug = augment_nopunct

    n_nopunct = sum(sentence_nopunct_predicate(sentence) for sentence in train_data)

    if aug is None:
        # x = # of sentences with punct
        #   = len(train_data) - n_nopunct
        # y = n_nopunct
        # aug x + y = 0.1 (x + y)
        # aug = (0.1 (x + y) - y) / x
        aug = (0.1 * len(train_data) - n_nopunct) / (len(train_data) - n_nopunct)
        logger.info("No-punct augmentation not specified.  Using %.4f to get 10%% training data with no punct at end" % aug)

    if aug <= 0:
        return train_data

    new_data = list(train_data)

    for sentence in train_data:
        if can_augment_nopunct_predicate(sentence):
            if random.random() < aug:
                # todo: could deep copy the words
                #       or not deep copy any of this
                new_sentence = list(sentence[:-1])
                new_data.append(new_sentence)

    logger.info("Augmenting dataset with non-punct-ending sentences.  Original length %d, with %d no-punct" % (len(train_data), n_nopunct))
    logger.info("Added %d additional sentences.  New total length %d" % (len(new_data) - len(train_data), len(new_data)))
    return new_data
