"""Builds a self-training dataset from an Italian data source and two models

The idea is that the top down and the inorder parsers should make
somewhat different errors, so hopefully the sum of an 86 f1 parser and
an 85.5 f1 parser will produce some half-decent silver trees which can
be used as self-training so that a new model can do better than either.

The dataset used is PaCCSS, which has 63000 pairs of sentences:

http://www.italianlp.it/resources/paccss-it-parallel-corpus-of-complex-simple-sentences-for-italian/
"""

import argparse
from collections import deque
import glob
import os
import random

from stanza.models.common import utils
from stanza.utils.datasets.constituency import selftrain

tqdm = utils.get_tqdm()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script that converts part of a wikipedia dump to silver standard trees"
    )
    selftrain.common_args(parser)
    parser.add_argument(
        '--input_dir',
        default='extern_data/vietnamese/wikipedia/text',
        help='Path to the wikipedia dump after processing by wikiextractor'
    )
    parser.add_argument(
        '--no_shuffle',
        dest='shuffle',
        action='store_false',
        help="Don't shuffle files when processing the directory"
    )

    parser.set_defaults(num_sentences=10000)

    args = parser.parse_args()
    return args

def list_wikipedia_files(input_dir):
    """
    Get a list of wiki files under the input_dir

    Recursively traverse the directory, then sort
    """
    wiki_files = []

    recursive_files = deque()
    recursive_files.extend(glob.glob(os.path.join(input_dir, "*")))
    while len(recursive_files) > 0:
        next_file = recursive_files.pop()
        if os.path.isdir(next_file):
            recursive_files.extend(glob.glob(os.path.join(next_file, "*")))
        elif os.path.split(next_file)[1].startswith("wiki_"):
            wiki_files.append(next_file)

    wiki_files.sort()
    return wiki_files

def read_wiki_file(filename):
    """
    Read the text from a wiki file as a list of paragraphs.

    Each <doc> </doc> is its own item in the list.
    Lines are separated by \n\n to give hints to the stanza tokenizer.
    The first line after <doc> is skipped as it is usually the document title.
    """
    with open(filename) as fin:
        lines = fin.readlines()
    docs = []
    current_doc = []
    line_iterator = iter(lines)
    line = next(line_iterator, None)
    while line is not None:
        if line.startswith("<doc"):
            # skip the next line, as it is usually the title
            line = next(line_iterator, None)
        elif line.startswith("</doc"):
            if current_doc:
                docs.append("\n\n".join(current_doc))
                current_doc = []
        else:
            # not the start or end of a doc
            # hopefully this is valid text
            line = line.replace("()", " ")
            line = line.replace("( )", " ")
            line = line.strip()
            if line.find("&lt;") > 0 or line.find("&gt;") > 0:
                line = ""
            if line:
                current_doc.append(line)
        line = next(line_iterator, None)
            
    if current_doc:
        docs.append("\n\n".join(current_doc))
    return docs

def main():
    args = parse_args()

    random.seed(1234)

    wiki_files = list_wikipedia_files(args.input_dir)
    if args.shuffle:
        random.shuffle(wiki_files)

    tag_pipe = selftrain.build_tag_pipe(ssplit=True, lang=args.lang)
    parser_pipes = selftrain.build_parser_pipes(args.lang, args.models)

    # create a blank file.  we will append to this file so that partial results can be used
    with open(args.output_file, "w") as fout:
        pass

    accepted_trees = set()
    for filename in tqdm(wiki_files, disable=False):
        docs = read_wiki_file(filename)
        new_trees = selftrain.find_matching_trees(docs, args.num_sentences, accepted_trees, tag_pipe, parser_pipes, shuffle=args.shuffle)
        accepted_trees.update(new_trees)

        with open(args.output_file, "a") as fout:
            for tree in sorted(new_trees):
                fout.write(tree)
                fout.write("\n")

if __name__ == '__main__':
    main()
