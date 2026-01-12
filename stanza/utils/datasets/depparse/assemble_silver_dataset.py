"""
A script that reads back in the matched conllu trees and splits them into files for each match length

(eg, a file for trees that matched in 0 subparsers, 1 subparser, 2, 3, etc)

python3 stanza/utils/datasets/depparse/assemble_silver_dataset.py en_silver.aa.p[12].conllu --output_prefix en_silver.part
"""

import argparse
import random

from collections import defaultdict

from stanza.utils.conll import CoNLL

random.seed(1234)

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Script that puts together the silver training data")
    parser.add_argument('input_file', type=str, nargs='+', help='Files to process')
    parser.add_argument('--output_prefix', type=str, default=None, help='Where to put the output files - a prefix')
    parser.add_argument('--max_silver_size', type=int, default=200000, help='Maximum number of sentences to put in the final silver')
    args = vars(parser.parse_args())
    return args

def check_matches(comments, match_type, input_file, sentence_idx):
    comments = [x for x in comments if x.startswith("# %s" % match_type)]
    assert len(comments) > 0, "Sentence with no %s in %s line %d" % (match_type, input_file, sentence_idx)
    assert len(comments) == 1, "Got more than one %s in %s line %d" % (match_type, input_file, sentence_idx)
    count = comments[0].split("=")[1].split("/")
    matched = int(count[0].strip())
    pipelines = int(count[1].strip())
    #print(sentence_idx, match_type, matched, pipelines, matched < pipelines)
    return matched < pipelines

def is_partial_match(sentence, input_file, sentence_idx):
    comments = sentence.comments
    if not check_matches(comments, "e1_matches", input_file, sentence_idx):
        return False
    if not check_matches(comments, "e2_matches", input_file, sentence_idx):
        return False
    return True

def main():
    args = parse_args()

    organized_sentences = defaultdict(list)
    partial_matches = []
    for input_file in args['input_file']:
        print("Reading %s" % input_file)
        try:
            doc = CoNLL.conll2doc(input_file)
        except Exception as e:
            raise RuntimeError("Could not read file %s" % input_file) from e

        for sentence_idx, sentence in enumerate(doc.sentences):
            comments = sentence.comments
            comments = [x for x in comments if x.startswith("# total_matches")]
            assert len(comments) > 0, "Sentence with no total_matches in %s line %d" % (input_file, sentence_idx)
            assert len(comments) == 1, "Got more than one total_matches in %s line %d" % (input_file, sentence_idx)
            matches = int(comments[0].split("=")[1].split("/")[0].strip())
            organized_sentences[matches].append(sentence)

            if is_partial_match(sentence, input_file, sentence_idx):
                partial_matches.append(sentence)

    for length in sorted(organized_sentences.keys()):
        print("Got %d sentences of length %d" % (len(organized_sentences[length]), length))
        if args['output_prefix']:
            output_filename = "%s.%02d.conllu" % (args['output_prefix'], length)
            doc.sentences = organized_sentences[length]
            CoNLL.write_doc2conll(doc, output_filename)

    if args['output_prefix']:
        print("Got %d sentences with partial matches in both ensembles" % len(partial_matches))
        output_filename = "%s.mixed.conllu" % (args['output_prefix'])
        if args['max_silver_size'] > 0 and len(partial_matches) > args['max_silver_size']:
            random.shuffle(partial_matches)
            partial_matches = partial_matches[:args['max_silver_size']]

        doc.sentences = partial_matches
        CoNLL.write_doc2conll(doc, output_filename)

if __name__ == '__main__':
    main()
