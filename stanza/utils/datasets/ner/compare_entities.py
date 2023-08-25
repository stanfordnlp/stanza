"""
Report the fraction of NER entities in one file which are present in another.

Purpose: show the coverage of one file on another, such as reporting
the number of entities in one dataset on another
"""


import argparse
import json

from stanza.models.common.doc import Document

def parse_args():
    parser = argparse.ArgumentParser(description="Report the coverage of one NER file on another.")
    parser.add_argument('--train', type=str, required=True, help='File to use to collect the known entities (not necessarily train).')
    parser.add_argument('--test', type=str, required=True, help='File for which we want to know the ratio of known entities')
    args = parser.parse_args()
    return args

def read_entities(filename):
    with open(filename) as fin:
        doc = Document(json.load(fin))

    entities = []
    for sentence in doc.sentences:
        current_entity = []
        previous_label = None
        for token in sentence.tokens:
            if token.ner == 'O' or token.ner.startswith("E-"):
                if token.ner.startswith("E-"):
                    current_entity.append(token.text)
                if current_entity:
                    entities.append(current_entity)
                    current_entity = []
                    previous_label = None
            elif token.ner.startswith("I-"):
                if previous_label is not None and previous_label != 'O' and previous_label[2:] != token.ner[2:]:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = []
                        previous_label = None
                current_entity.append(token.text)
            elif token.ner.startswith("B-") or token.ner.startswith("S-"):
                if current_entity:
                    entities.append(current_entity)
                    current_entity = []
                    previous_label = None
                entities.append(token.text)
                if token.ner.startswith("S-"):
                    entities.append(current_entity)
                    current_entity = []
                    previous_label = None
            previous_label = token.ner
        if current_entity:
            entities.append(current_entity)
    entities = [tuple(x) for x in entities]
    return entities

def report_known_entities(train_file, test_file):
    train_entities = read_entities(train_file)
    test_entities = read_entities(test_file)

    train_entities = set(train_entities)
    total_score = sum(1 for x in test_entities if x in train_entities)
    print(total_score / len(test_entities))

def main():
    args = parse_args()

    report_known_entities(args.train, args.test)

if __name__ == '__main__':
    main()
