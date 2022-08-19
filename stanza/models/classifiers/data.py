"""Stanza models classifier data functions."""

import logging
import json
import re
from typing import List

import stanza.models.classifiers.classifier_args as classifier_args

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
