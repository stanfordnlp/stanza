"""
Report the fraction of NER entities in one file which are present in another.

Purpose: show the coverage of one file on another, such as reporting
the number of entities in one dataset on another
"""


import argparse

from stanza.utils.datasets.ner.utils import read_bioes_entities

def parse_args():
    parser = argparse.ArgumentParser(description="Report the coverage of one NER file on another.")
    parser.add_argument('--train', type=str, required=True, help='File to use to collect the known entities (not necessarily train).')
    parser.add_argument('--test', type=str, required=True, help='File for which we want to know the ratio of known entities')
    args = parser.parse_args()
    return args

def report_known_entities(train_file, test_file):
    train_entities = read_bioes_entities(train_file)
    test_entities = read_bioes_entities(test_file)

    train_entities = set(x[0] for x in train_entities)
    total_score = sum(1 for x in test_entities if x[0] in train_entities)
    print(total_score / len(test_entities))

def main():
    args = parse_args()

    report_known_entities(args.train, args.test)

if __name__ == '__main__':
    main()
