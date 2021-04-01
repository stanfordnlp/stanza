"""
Preprocess the WikiNER dataset, by
1) normalizing tags;
2) split into train (70%), dev (15%), test (15%) datasets.
"""

import os
import random
from collections import Counter
random.seed(1234)

def read_sentences(filename):
    sents = []
    cache = []
    skipped = 0
    skip = False
    with open(filename) as infile:
        for i, line in enumerate(infile):
            line = line.rstrip()
            if len(line) == 0:
                if len(cache) > 0:
                    if not skip:
                        sents.append(cache)
                    else:
                        skipped += 1
                        skip = False
                    cache = []
                continue
            array = line.split()
            if len(array) != 2:
                skip = True
                continue
            #assert len(array) == 2, "Format error at line {}: {}".format(i+1, line)
            w, t = array
            cache.append([w, t])
        if len(cache) > 0:
            if not skip:
                sents.append(cache)
            else:
                skipped += 1
            cache = []
    print("Skipped {} examples due to formatting issues.".format(skipped))
    return sents

def write_sentences_to_file(sents, filename):
    print(f"Writing {len(sents)} sentences to {filename}")
    with open(filename, 'w') as outfile:
        for sent in sents:
            for pair in sent:
                print(f"{pair[0]}\t{pair[1]}", file=outfile)
            print("", file=outfile)

def split_wikiner(in_filename, directory):
    sents = read_sentences(in_filename)
    print(f"{len(sents)} sentences read from file.")

    # split
    num = len(sents)
    train_num = int(num*0.7)
    dev_num = int(num*0.15)

    random.shuffle(sents)
    train_sents = sents[:train_num]
    dev_sents = sents[train_num:train_num+dev_num]
    test_sents = sents[train_num+dev_num:]

    write_sentences_to_file(train_sents, os.path.join(directory, 'train.bio'))
    write_sentences_to_file(dev_sents, os.path.join(directory, 'dev.bio'))
    write_sentences_to_file(test_sents, os.path.join(directory, 'test.bio'))

if __name__ == "__main__":
    in_filename = 'raw/wp2.txt'
    directory = "."
    split_wikiner(in_filename, directory)
