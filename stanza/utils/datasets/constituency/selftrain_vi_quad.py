"""
Processes the train section of VI QuAD into trees suitable for use in the conparser lm
"""

import argparse
import json
import logging

import stanza
from stanza.utils.datasets.constituency import selftrain

logger = logging.getLogger('stanza')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script that converts vi quad to silver standard trees"
    )
    selftrain.common_args(parser)
    selftrain.add_length_args(parser)
    parser.add_argument(
        '--input_file',
        default="extern_data/vietnamese/ViQuAD/train_ViQuAD.json",
        help='Path to the ViQuAD train file'
    )
    parser.add_argument(
        '--tokenize_only',
        default=False,
        action='store_true',
        help='Tokenize instead of writing trees'
    )

    args = parser.parse_args()
    return args

def parse_quad(text):
    """
    Read in a file from the VI quad dataset

    The train file has a specific format:
    the doc has a 'data' section
    each block in the data is a separate document (138 in the train file, for example)
    each block has a 'paragraphs' section
    each paragrah has 'qas' and 'context'.  we care about the qas
    each piece of qas has 'question', which is what we actually want
    """
    doc = json.loads(text)

    questions = []

    for block in doc['data']:
        paragraphs = block['paragraphs']
        for paragraph in paragraphs:
            qas = paragraph['qas']
            for question in qas:
                questions.append(question['question'])

    return questions


def read_quad(train_file):
    with open(train_file) as fin:
        text = fin.read()

    return parse_quad(text)

def main():
    """
    Turn the train section of VI quad into a list of trees
    """
    args = parse_args()

    docs = read_quad(args.input_file)
    logger.info("Read %d lines from %s", len(docs), args.input_file)
    if args.tokenize_only:
        pipe = stanza.Pipeline(args.lang, processors="tokenize")
        text = selftrain.tokenize_docs(docs, pipe, args.min_len, args.max_len)
        with open(args.output_file, "w", encoding="utf-8") as fout:
            for line in text:
                fout.write(line)
                fout.write("\n")
    else:
        tag_pipe = selftrain.build_tag_pipe(ssplit=False, lang=args.lang)
        parser_pipes = selftrain.build_parser_pipes(args.lang, args.models)

        # create a blank file.  we will append to this file so that partial results can be used
        with open(args.output_file, "w") as fout:
            pass

        accepted_trees = set()
        new_trees = selftrain.find_matching_trees(docs, args.num_sentences, accepted_trees, tag_pipe, parser_pipes, shuffle=False, chunk_size=100)
        new_trees = [tree for tree in new_trees if tree.find("(_SQ") >= 0]
        with open(args.output_file, "a") as fout:
            for tree in sorted(new_trees):
                fout.write(tree)
                fout.write("\n")

if __name__ == '__main__':
    main()
