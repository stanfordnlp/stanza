"""
Convert a ArmTDP-NER dataset to BIO format

The dataset is here:

https://github.com/myavrum/ArmTDP-NER.git
"""

import os
import json
import re
import stanza
import random
random.seed(1234)
nlp = stanza.Pipeline(lang='hy', processors='tokenize')


def read_data(path: str) -> list:
    """
    Reads Armenian data file

    Returns list of dictionaries, where each dictionary represents
    a paragraph's information (text, labels, etc.)
    """
    with open(path, 'r') as file:
        paragraphs = [json.loads(line) for line in file]
    return paragraphs


def filter_unicode_broken_characters(paragraphs: list) -> list:
    """
    Removes all '\u202c' unicode characters in texts
    TODO: why?
    """
    for paragraph in paragraphs:
        paragraph['text'] = re.sub('\u202c', '', paragraph['text'])


def format_sentence_as_beios(sentence, labels) -> list:
    sentence_toc = ''
    current_label = []
    for token in sentence.tokens:
        if current_label:
            tag = current_label[2]
            if token.end_char == current_label[1]:
                sentence_toc += token.text + '\tE-' + tag + '\n'
                current_label = []
            else:
                sentence_toc += token.text + '\tI-' + tag + '\n'
        else:
            current_label = get_label(token.start_char, labels)
            if current_label:
                tag = current_label[2]
                if token.start_char == current_label[0] and token.end_char == current_label[1]:
                    sentence_toc += token.text + '\tS-' + tag + '\n'
                    current_label = []
                elif token.start_char == current_label[0]:
                    sentence_toc += token.text + '\tB-' + tag + '\n'
            else:
                sentence_toc += token.text + '\tO' + '\n'
                current_label = []
    return sentence_toc[:-1]


def get_label(tok_start_char: int, labels: list) -> list:
    for label in labels:
        if label[0] == tok_start_char:
            return label
    return []


def convert_to_bioes(paragraphs):
    beios_sents = []
    for i, paragraph in enumerate(paragraphs):
        print(i)
        doc = nlp(paragraph['text'])
        for sentence in doc.sentences:
            beios_sents.append(format_sentence_as_beios(sentence, paragraph['labels']))
    return beios_sents


def write_sentences_to_file_(sents, filename):
    print(f"Writing {len(sents)} sentences to {filename}")
    with open(filename, 'w') as outfile:
        for sent in sents:
            outfile.write(sent + '\n\n')


def train_test_dev_split(sents, base_output_path, short_name, train_fraction=0.7, dev_fraction=0.15):
    num = len(sents)
    train_num = int(num * train_fraction)
    dev_num = int(num * dev_fraction)
    if train_fraction + dev_fraction > 1.0:
        raise ValueError(
            "Train and dev fractions added up to more than 1: {} {} {}".format(train_fraction, dev_fraction))

    random.shuffle(sents)
    train_sents = sents[:train_num]
    dev_sents = sents[train_num:train_num + dev_num]
    test_sents = sents[train_num + dev_num:]
    batches = [train_sents, dev_sents, test_sents]
    filenames = [f'{short_name}.train.tsv', f'{short_name}.dev.tsv', f'{short_name}.test.tsv']
    for batch, filename in zip(batches, filenames):
        write_sentences_to_file_(batch, os.path.join(base_output_path, filename))


def convert_hy_armtdp(base_input_path, base_output_path, short_name):
    paragraphs = read_data(os.path.join(base_input_path, 'ArmNER-HY.json1'))
    filter_unicode_broken_characters(paragraphs)
    beios_sentences = convert_to_bioes(paragraphs)
    train_test_dev_split(beios_sentences, base_output_path, short_name)


