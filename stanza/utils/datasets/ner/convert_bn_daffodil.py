"""
Convert a Bengali NER dataset to our internal .json format

The dataset is here:

https://github.com/Rifat1493/Bengali-NER/tree/master/Input
"""

import argparse
import os
import random
import tempfile

from stanza.utils.datasets.ner.utils import read_tsv, write_dataset

def redo_time_tags(sentences):
    """
    Replace all TIM, TIM with B-TIM, I-TIM

    A brief use of Google Translate suggests the time phrases are
    generally one phrase, so we don't want to turn this into B-TIM, B-TIM
    """
    new_sentences = []

    for sentence in sentences:
        new_sentence = []
        prev_time = False
        for word, tag in sentence:
            if tag == 'TIM':
                if prev_time:
                    new_sentence.append((word, "I-TIM"))
                else:
                    prev_time = True
                    new_sentence.append((word, "B-TIM"))
            else:
                prev_time = False
                new_sentence.append((word, tag))
        new_sentences.append(new_sentence)

    return new_sentences

def strip_words(dataset):
    return [[(x[0].strip().replace('\ufeff', ''), x[1]) for x in sentence] for sentence in dataset]

def filter_blank_words(train_file, train_filtered_file):
    """
    As of July 2022, this dataset has blank words with O labels, which is not ideal

    This method removes those lines
    """
    with open(train_file, encoding="utf-8") as fin:
        with open(train_filtered_file, "w", encoding="utf-8") as fout:
            for line in fin:
                if line.strip() == 'O':
                    continue
                fout.write(line)

def filter_broken_tags(train_sentences):
    """
    Eliminate any sentences where any of the tags were empty
    """
    return [x for x in train_sentences if not any(y[1] is None for y in x)]

def filter_bad_words(train_sentences):
    """
    Not bad words like poop, but characters that don't exist

    These characters look like n and l in emacs, but they are really
    0xF06C and 0xF06E
    """
    return [[x for x in sentence if not x[0] in ("", "")] for sentence in train_sentences]

def read_datasets(in_directory):
    """
    Reads & splits the train data, reads the test data

    There is no validation data, so we split the training data into
    two pieces and use the smaller piece as the dev set

    Also performeed is a conversion of TIM -> B-TIM, I-TIM
    """
    # make sure we always get the same shuffle & split
    random.seed(1234)

    train_file = os.path.join(in_directory, "Input", "train_data.txt")
    with tempfile.TemporaryDirectory() as tempdir:
        train_filtered_file = os.path.join(tempdir, "train.txt")
        filter_blank_words(train_file, train_filtered_file)
        train_sentences = read_tsv(train_filtered_file, text_column=0, annotation_column=1, keep_broken_tags=True)
    train_sentences = filter_broken_tags(train_sentences)
    train_sentences = filter_bad_words(train_sentences)
    train_sentences = redo_time_tags(train_sentences)
    train_sentences = strip_words(train_sentences)

    test_file = os.path.join(in_directory, "Input", "test_data.txt")
    test_sentences = read_tsv(test_file, text_column=0, annotation_column=1, keep_broken_tags=True)
    test_sentences = filter_broken_tags(test_sentences)
    test_sentences = filter_bad_words(test_sentences)
    test_sentences = redo_time_tags(test_sentences)
    test_sentences = strip_words(test_sentences)

    random.shuffle(train_sentences)
    split_len = len(train_sentences) * 9 // 10
    dev_sentences = train_sentences[split_len:]
    train_sentences = train_sentences[:split_len]

    datasets = (train_sentences, dev_sentences, test_sentences)
    return datasets

def convert_dataset(in_directory, out_directory):
    """
    Reads the datasets using read_datasets, then write them back out
    """
    datasets = read_datasets(in_directory)
    write_dataset(datasets, out_directory, "bn_daffodil")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="/home/john/extern_data/ner/bangla/Bengali-NER", help="Where to find the files")
    parser.add_argument('--output_path', type=str, default="/home/john/stanza/data/ner", help="Where to output the results")
    args = parser.parse_args()

    convert_dataset(args.input_path, args.output_path)
