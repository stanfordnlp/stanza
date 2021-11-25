"""
Processes the train section of VI QuAD into trees suitable for use in the conparser lm
"""

import argparse
import json
import logging

from stanza.utils.datasets.constituency import selftrain

logger = logging.getLogger('stanza')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script that converts vi quad to silver standard trees"
    )
    selftrain.common_args(parser)
    parser.add_argument(
        '--input_file',
        default="extern_data/vietnamese/ViQuAD/train_ViQuAD.json",
        help='Path to the ViQuAD train file'
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

    tag_pipe = selftrain.build_tag_pipe(ssplit=False, lang=args.lang)
    parser_pipes = selftrain.build_parser_pipes(args.lang, args.models)

    # create a blank file.  we will append to this file so that partial results can be used
    with open(args.output_file, "w") as fout:
        pass

    accepted_trees = set()
    docs = read_quad(args.input_file)
    logger.info("Read %d lines from %s", len(docs), args.input_file)
    new_trees = selftrain.find_matching_trees(docs, args.num_sentences, accepted_trees, tag_pipe, parser_pipes, shuffle=False, chunk_size=100)
    new_trees = [tree for tree in new_trees if tree.find("(_SQ") >= 0]
    with open(args.output_file, "a") as fout:
        for tree in sorted(new_trees):
            fout.write(tree)
            fout.write("\n")

if __name__ == '__main__':
    main()
