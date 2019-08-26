"""
This script converts NER data from the CoNLL03 format to the latest CoNLL-U format.
"""

import argparse

MIN_NUM_FIELD = 4
MAX_NUM_FIELD = 5

DOC_START_TOKEN = '-DOCSTART-'

def parse_args():
    parser = argparse.ArgumentParser(description="Convert the conll03 data into conllu format.")
    parser.add_argument('input', help='Input filename.')
    parser.add_argument('output', help='Output filename.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    sentences = load_conll03(args.input)
    print("{} examples loaded from {}".format(len(sentences), args.input))
    
    filler = "_"
    with open(args.output, 'w') as outfile:
        for (words, tags) in sentences:
            for i, (w, t) in enumerate(zip(words, tags)):
                print("{}\t{}\t{}\tner={}".format(i+1, w, "\t".join([filler]*7), t), file=outfile)
            print("", file=outfile)
    print("Generated conllu file {}.".format(args.output))

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
