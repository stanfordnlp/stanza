"""
Converts the FIRE 2013 dataset to TSV

http://au-kbc.org/nlp/NER-FIRE2013/index.html

The dataset is in six tab separated columns.  The columns are

word tag chunk ner1 ner2 ner3

This script keeps just the word and the ner1.  It is quite possible that using the tag would help
"""

import argparse
import glob
import os
import random

def normalize(e1, e2, e3):
    if e1 == 'o':
        return "O"

    if e2 != 'o' and e1[:2] != e2[:2]:
        raise ValueError("Found a token with conflicting position tags %s,%s" % (e1, e2))
    if e3 != 'o' and e2 == 'o':
        raise ValueError("Found a token with tertiary label but no secondary label %s,%s,%s" % (e1, e2, e3))
    if e3 != 'o' and (e1[:2] != e2[:2] or e1[:2] != e3[:2]):
        raise ValueError("Found a token with conflicting position tags %s,%s,%s" % (e1, e2, e3))

    if e1[2:] in ('ORGANIZATION', 'FACILITIES'):
        return e1
    if e1[2:] == 'ENTERTAINMENT' and e2[2:] != 'SPORTS' and e2[2:] != 'CINEMA':
        return e1
    if e1[2:] == 'DISEASE' and e2 == 'o':
        return e1
    if e1[2:] == 'PLANTS' and e2[2:] != 'PARTS':
        return e1
    if e1[2:] == 'PERSON' and e2[2:] == 'INDIVIDUAL':
        return e1
    if e1[2:] == 'LOCATION' and e2[2:] == 'PLACE':
        return e1
    if e1[2:] in ('DATE', 'TIME', 'YEAR'):
        string = e1[:2] + 'DATETIME'
        return string

    return "O"

def read_fileset(filenames):
    # first, read the sentences from each data file
    sentences = []
    for filename in filenames:
        with open(filename) as fin:
            next_sentence = []
            for line in fin:
                line = line.strip()
                if not line:
                    # lots of single line "sentences" in the dataset
                    if next_sentence:
                        if len(next_sentence) > 1:
                            sentences.append(next_sentence)
                        next_sentence = []
                else:
                    next_sentence.append(line)
            if next_sentence and len(next_sentence) > 1:
                sentences.append(next_sentence)
    return sentences

def write_fileset(output_csv_file, sentences):
    with open(output_csv_file, "w") as fout:
        for sentence in sentences:
            for line in sentence:
                pieces = line.split("\t")
                if len(pieces) != 6:
                    raise ValueError("Found %d pieces instead of the expected 6" % len(pieces))
                if pieces[3] == 'o' and (pieces[4] != 'o' or pieces[5] != 'o'):
                    raise ValueError("Inner NER labeled but the top layer was O")
                fout.write("%s\t%s\n" % (pieces[0], normalize(pieces[3], pieces[4], pieces[5])))
            fout.write("\n")

def convert_fire_2013(input_path, train_csv_file, dev_csv_file, test_csv_file):
    random.seed(1234)

    filenames = glob.glob(os.path.join(input_path, "*"))

    # won't be numerically sorted... shouldn't matter
    filenames = sorted(filenames)
    random.shuffle(filenames)

    sentences = read_fileset(filenames)
    random.shuffle(sentences)

    train_cutoff = int(0.8 * len(sentences))
    dev_cutoff   = int(0.9 * len(sentences))

    train_sentences = sentences[:train_cutoff]
    dev_sentences   = sentences[train_cutoff:dev_cutoff]
    test_sentences  = sentences[dev_cutoff:]

    random.shuffle(train_sentences)
    random.shuffle(dev_sentences)
    random.shuffle(test_sentences)

    assert len(train_sentences) > 0
    assert len(dev_sentences) > 0
    assert len(test_sentences) > 0

    write_fileset(train_csv_file, train_sentences)
    write_fileset(dev_csv_file,   dev_sentences)
    write_fileset(test_csv_file,  test_sentences)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="/home/john/extern_data/ner/FIRE2013/hindi_train",  help="Directory with raw files to read")
    parser.add_argument('--train_file', type=str, default="/home/john/stanza/data/ner/hi_fire2013.train.csv", help="Where to put the train file")
    parser.add_argument('--dev_file',   type=str, default="/home/john/stanza/data/ner/hi_fire2013.dev.csv",   help="Where to put the dev file")
    parser.add_argument('--test_file',  type=str, default="/home/john/stanza/data/ner/hi_fire2013.test.csv",  help="Where to put the test file")
    args = parser.parse_args()

    convert_fire_2013(args.input_path, args.train_file, args.dev_file, args.test_file)
