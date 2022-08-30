"""Stanza models classifier data functions."""

import collections
import logging
import json
import random
import re
from typing import List

import stanza.models.classifiers.classifier_args as classifier_args
from stanza.models.common.vocab import PAD, PAD_ID, UNK, UNK_ID

logger = logging.getLogger('stanza')


def update_text(sentence: List[str], wordvec_type: classifier_args.WVType) -> List[str]:
    """
    Process a line of text (with tokenization provided as whitespace)
    into a list of strings.
    """
    # stanford sentiment dataset has a lot of random - and /
    # remove those characters and flatten the newly created sublists into one list each time
    sentence = [y for x in sentence for y in x.split("-") if y]
    sentence = [y for x in sentence for y in x.split("/") if y]
    sentence = [x.strip() for x in sentence]
    sentence = [x for x in sentence if x]
    if sentence == []:
        # removed too much
        sentence = ["-"]
    # our current word vectors are all entirely lowercased
    sentence = [word.lower() for word in sentence]
    if wordvec_type == classifier_args.WVType.WORD2VEC:
        return sentence
    elif wordvec_type == classifier_args.WVType.GOOGLE:
        new_sentence = []
        for word in sentence:
            if word != '0' and word != '1':
                word = re.sub('[0-9]', '#', word)
            new_sentence.append(word)
        return new_sentence
    elif wordvec_type == classifier_args.WVType.FASTTEXT:
        return sentence
    elif wordvec_type == classifier_args.WVType.OTHER:
        return sentence
    else:
        raise ValueError("Unknown wordvec_type {}".format(wordvec_type))


def read_dataset(dataset, wordvec_type: classifier_args.WVType, min_len: int) -> List[tuple]:
    """
    returns a list where the values of the list are
      label, [token...]
    """
    lines = []
    for filename in dataset.split(","):
        with open(filename, encoding="utf-8") as fin:
            new_lines = json.load(fin)
        new_lines = [(str(x['sentiment']), x['text']) for x in new_lines]
        lines.extend(new_lines)
    # TODO: maybe do this processing later, once the model is built.
    # then move the processing into the model so we can use
    # overloading to potentially make future model types
    lines = [(x[0], update_text(x[1], wordvec_type)) for x in lines]
    if min_len:
        lines = [x for x in lines if len(x[1]) >= min_len]
    return lines

def dataset_labels(dataset):
    """
    Returns a sorted list of label name
    """
    labels = set([x[0] for x in dataset])
    if all(re.match("^[0-9]+$", label) for label in labels):
        # if all of the labels are integers, sort numerically
        # maybe not super important, but it would be nicer than having
        # 10 before 2
        labels = [str(x) for x in sorted(map(int, list(labels)))]
    else:
        labels = sorted(list(labels))
    return labels

def dataset_vocab(dataset):
    vocab = set()
    for line in dataset:
        for word in line[1]:
            vocab.add(word)
    vocab = [PAD, UNK] + list(vocab)
    if vocab[PAD_ID] != PAD or vocab[UNK_ID] != UNK:
        raise ValueError("Unexpected values for PAD and UNK!")
    return vocab

def sort_dataset_by_len(dataset):
    """
    returns a dict mapping length -> list of items of that length
    an OrderedDict is used to that the mapping is sorted from smallest to largest
    """
    sorted_dataset = collections.OrderedDict()
    lengths = sorted(list(set(len(x[1]) for x in dataset)))
    for l in lengths:
        sorted_dataset[l] = []
    for item in dataset:
        sorted_dataset[len(item[1])].append(item)
    return sorted_dataset

def shuffle_dataset(sorted_dataset):
    """
    Given a dataset sorted by len, sorts within each length to make
    chunks of roughly the same size.  Returns all items as a single list.
    """
    dataset = []
    for l in sorted_dataset.keys():
        items = list(sorted_dataset[l])
        random.shuffle(items)
        dataset.extend(items)
    return dataset


def check_labels(labels, dataset):
    """
    Check that all of the labels in the dataset are in the known labels.

    Actually, unknown labels could be acceptable if we just treat the model as always wrong.
    However, this is a good sanity check to make sure the datasets match
    """
    new_labels = dataset_labels(dataset)
    not_found = [i for i in new_labels if i not in labels]
    if not_found:
        raise RuntimeError('Dataset contains labels which the model does not know about:' + str(not_found))

