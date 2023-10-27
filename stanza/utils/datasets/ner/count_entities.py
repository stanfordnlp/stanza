
import argparse
from collections import defaultdict
import json

from stanza.models.common.doc import Document
from stanza.utils.datasets.ner.utils import list_doc_entities

def parse_args():
    parser = argparse.ArgumentParser(description="Report the coverage of one NER file on another.")
    parser.add_argument('filename', type=str, nargs='+', help='File(s) to count')
    args = parser.parse_args()
    return args


def count_entities(*filenames):
    entity_collection = defaultdict(list)

    for filename in filenames:
        with open(filename) as fin:
            doc = Document(json.load(fin))
            num_tokens = sum(1 for sentence in doc.sentences for token in sentence.tokens)
            print("Number of tokens in %s: %d" % (filename, num_tokens))
            entities = list_doc_entities(doc)

        for ent in entities:
            entity_collection[ent[1]].append(ent[0])

    keys = sorted(entity_collection.keys())
    for k in keys:
        print(k, len(entity_collection[k]))

def main():
    args = parse_args()

    count_entities(*args.filename)

if __name__ == '__main__':
    main()
