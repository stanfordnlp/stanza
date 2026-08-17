"""
A script for handling the specific format used in the Shahmukhi NER dataset

Train and Test files are both written as tokenized sentences, one
sentence per line, with separate files for the tokens and the NER tags

This script combines those lines into the tagged sentences used by the
utilities elsewhere in this project

Furthermore, as there is no dev set, the train set is split 90/10
"""

import os
import random
import stanza.utils.datasets.ner.utils as utils

def combine_files(txt_filename, ner_filename):
    sentences = []
    with (open(txt_filename, encoding="utf-8") as text_fin,
          open(ner_filename, encoding="utf-8") as ner_fin):
        for line_idx, (text, ner) in enumerate(zip(text_fin, ner_fin)):
            text_pieces = text.strip().split()
            ner_pieces = ner.strip().split()

            # already verified - all lines are the same length
            assert len(text_pieces) == len(ner_pieces), "Error at line %d" % line_idx
            sentence = [(x, y) for x, y in zip(text_pieces, ner_pieces)]
            sentences.append(sentence)
    return sentences

def random_split(train_section, split_size=0.1):
    random.seed(1234)
    train_split = []
    dev_split = []
    for sentence in train_section:
        if random.random() < split_size:
            dev_split.append(sentence)
        else:
            train_split.append(sentence)
    return train_split, dev_split

def convert(short_name, base_input_path, base_output_path):
    train_section = combine_files(os.path.join(base_input_path, "Train.txt"),
                                  os.path.join(base_input_path, "Train.ner"))
    test_section =  combine_files(os.path.join(base_input_path, "Test.txt"),
                                  os.path.join(base_input_path, "Test.ner"))

    print("Read %d sentences from the train section" % len(train_section))
    print("Read %d sentences from the test section" % len(test_section))

    train_section, dev_section = random_split(train_section)

    utils.write_dataset([train_section, dev_section, test_section], base_output_path, short_name)
    utils.convert_bio_to_json(base_output_path, base_output_path, short_name)
