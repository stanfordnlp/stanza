"""
This script converts NER data from the CoNLL03 format to the latest CoNLL-U format. The script assumes that in the 
input column format data, the token is always in the first column, while the NER tag is always in the last column.
"""

import argparse
import json

MIN_NUM_FIELD = 2
MAX_NUM_FIELD = 5

DOC_START_TOKEN = '-DOCSTART-'

def parse_args():
    parser = argparse.ArgumentParser(description="Convert the conll03 format data into conllu format.")
    parser.add_argument('input', help='Input conll03 format data filename.')
    parser.add_argument('output', help='Output json filename.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    process_dataset(args.input, args.output)

def process_dataset(input_filename, output_filename):
    sentences = load_conll03(input_filename)
    print("{} examples loaded from {}".format(len(sentences), input_filename))
    
    document = []
    for (words, tags) in sentences:
        sent = []
        for w, t in zip(words, tags):
            sent += [{'text': w, 'ner': t}]
        document += [sent]

    with open(output_filename, 'w') as outfile:
        json.dump(document, outfile)
    print("Generated json file {}.".format(output_filename))

# TODO: make skip_doc_start an argument
def load_conll03(filename, skip_doc_start=True):
    cached_lines = []
    examples = []
    with open(filename) as infile:
        for line in infile:
            line = line.strip()
            if skip_doc_start and DOC_START_TOKEN in line:
                continue
            if len(line) > 0:
                array = line.split()
                if len(array) < MIN_NUM_FIELD:
                    continue
                else:
                    cached_lines.append(line)
            elif len(cached_lines) > 0:
                example = process_cache(cached_lines)
                examples.append(example)
                cached_lines = []
        if len(cached_lines) > 0:
            examples.append(process_cache(cached_lines))
    return examples

def process_cache(cached_lines):
    tokens = []
    ner_tags = []
    for line in cached_lines:
        array = line.split()
        assert len(array) >= MIN_NUM_FIELD and len(array) <= MAX_NUM_FIELD
        tokens.append(array[0])
        ner_tags.append(array[-1])
    return (tokens, ner_tags)

if __name__ == '__main__':
    main()
