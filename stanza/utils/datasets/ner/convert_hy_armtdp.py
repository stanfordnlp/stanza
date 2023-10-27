"""
Convert a ArmTDP-NER dataset to BIO format

The dataset is here:

https://github.com/myavrum/ArmTDP-NER.git
"""

import argparse
import os
import json
import re
import stanza
import random
from tqdm import tqdm

from stanza import DownloadMethod, Pipeline
import stanza.utils.default_paths as default_paths

def read_data(path: str) -> list:
    """
    Reads the Armenian named entity recognition dataset

    Returns a list of dictionaries.
    Each dictionary contains information
    about a paragraph (text, labels, etc.)
    """
    with open(path, 'r') as file:
        paragraphs = [json.loads(line) for line in file]
    return paragraphs


def filter_unicode_broken_characters(text: str) -> str:
    """
    Removes all unicode characters in text
    """
    return re.sub(r'\\u[A-Za-z0-9]{4}', '', text)


def get_label(tok_start_char: int, tok_end_char: int, labels: list) -> list:
    """
    Returns the label that corresponds to the given token
    """
    for label in labels:
        if label[0] <= tok_start_char and label[1] >= tok_end_char:
            return label
    return []


def format_sentences(paragraphs: list, nlp_hy: Pipeline) -> list:
    """
    Takes a list of paragraphs and returns a list of sentences,
    where each sentence is a list of tokens along with their respective entity tags.
    """
    sentences = []
    for paragraph in tqdm(paragraphs):
        doc = nlp_hy(filter_unicode_broken_characters(paragraph['text']))
        for sentence in doc.sentences:
            sentence_ents = []
            entity = []
            for token in sentence.tokens:
                label = get_label(token.start_char, token.end_char, paragraph['labels'])
                if label:
                    entity.append(token.text)
                    if token.end_char == label[1]:
                        sentence_ents.append({'tokens': entity,
                                              'tag': label[2]})
                        entity = []
                else:
                    sentence_ents.append({'tokens': [token.text],
                                          'tag': 'O'})
            sentences.append(sentence_ents)
    return sentences


def convert_to_bioes(sentences: list) -> list:
    """
    Returns a list of strings where each string represents a sentence in BIOES format
    """
    beios_sents = []
    for sentence in tqdm(sentences):
        sentence_toc = ''
        for ent in sentence:
            if ent['tag'] == 'O':
                sentence_toc += ent['tokens'][0] + '\tO' + '\n'
            else:
                if len(ent['tokens']) == 1:
                    sentence_toc += ent['tokens'][0] + '\tS-' + ent['tag'] + '\n'
                else:
                    sentence_toc += ent['tokens'][0] + '\tB-' + ent['tag'] + '\n'
                    for token in ent['tokens'][1:-1]:
                        sentence_toc += token + '\tI-' + ent['tag'] + '\n'
                    sentence_toc += ent['tokens'][-1] + '\tE-' + ent['tag'] + '\n'
        beios_sents.append(sentence_toc)
    return beios_sents


def write_sentences_to_file(sents, filename):
    print(f"Writing {len(sents)} sentences to {filename}")
    with open(filename, 'w') as outfile:
        for sent in sents:
            outfile.write(sent + '\n\n')


def train_test_dev_split(sents, base_output_path, short_name, train_fraction=0.7, dev_fraction=0.15):
    """
    Splits a list of sentences into training, dev, and test sets,
    and writes each set to a separate file with write_sentences_to_file
    """
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
        write_sentences_to_file(batch, os.path.join(base_output_path, filename))


def convert_dataset(base_input_path, base_output_path, short_name, download_method=DownloadMethod.DOWNLOAD_RESOURCES):
    nlp_hy = stanza.Pipeline(lang='hy', processors='tokenize', download_method=download_method)
    paragraphs = read_data(os.path.join(base_input_path, 'ArmNER-HY.json1'))
    tagged_sentences = format_sentences(paragraphs, nlp_hy)
    beios_sentences = convert_to_bioes(tagged_sentences)
    train_test_dev_split(beios_sentences, base_output_path, short_name)


if __name__ == '__main__':
    paths = default_paths.get_default_paths()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=os.path.join(paths["NERBASE"], "armenian", "ArmTDP-NER"), help="Path to input file")
    parser.add_argument('--output_path', type=str, default=paths["NER_DATA_DIR"], help="Path to the output directory")
    parser.add_argument('--short_name', type=str, default="hy_armtdp", help="Name to identify the dataset and the model")
    parser.add_argument('--download_method', type=str, default=DownloadMethod.DOWNLOAD_RESOURCES, help="Download method for initializing the Pipeline.  Default downloads the Armenian pipeline, --download_method NONE does not.  Options: %s" % DownloadMethod._member_names_)
    args = parser.parse_args()

    convert_dataset(args.input_path, args.output_path, args.short_name, args.download_method)
