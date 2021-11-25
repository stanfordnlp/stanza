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

import stanza
from stanza.models.common import utils

tqdm = utils.get_tqdm()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script that converts part of a wikipedia dump to silver standard trees"
    )
    parser.add_argument(
        '--input_dir',
        default='extern_data/vietnamese/wikipedia/text',
        help='Path to the wikipedia dump after processing by wikiextractor'
    )
    parser.add_argument(
        '--output_file',
        default='data/constituency/vi_silver.mrg',
        help='Where to write the silver trees'
    )
    parser.add_argument(
        '--lang',
        default='vi',
        help='Which language tools to use for tokenization and POS'
    )
    parser.add_argument(
        '--num_sentences',
        type=int,
        default=1000,
        help='How many sentences to get per file (max)'
    )
    parser.add_argument(
        '--models',
        default='saved_models/constituency/vi_vlsp21_inorder.pt',
        help='What models to use for parsing.  comma-separated'
    )
    parser.add_argument(
        '--no_shuffle',
        dest='shuffle',
        action='store_false',
        help="Don't shuffle files when processing the directory"
    )

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

def find_matching_trees(docs, num_sentences, accepted_trees, tag_pipe, parser_pipes):
    """
    Find trees where all the parsers in parser_pipes agree
    """
    random.shuffle(docs)
    chunk_size = 10
    new_trees = set()
    with tqdm(total=num_sentences, leave=False) as pbar:
        for chunk_start in range(0, len(docs), chunk_size):
            chunk = docs[chunk_start:chunk_start+chunk_size]
            chunk = [stanza.Document([], text=t) for t in chunk]
            tag_pipe(chunk)

            # for now, we don't have a good way to deal with sentences longer than the bert maxlen
            chunk = [d for d in chunk if max(len(s.words) for s in d.sentences) < 145]

            parses = []
            for pipe in parser_pipes:
                pipe(chunk)
                trees = ["{:L}".format(sent.constituency) for doc in chunk for sent in doc.sentences]
                parses.append(trees)

            for tree in zip(*parses):
                if len(set(tree)) != 1:
                    continue
                tree = tree[0]
                if tree in accepted_trees:
                    continue
                if tree not in new_trees:
                    new_trees.add(tree)
                    pbar.update(1)
                if len(new_trees) >= num_sentences:
                    return new_trees

    return new_trees

def build_tag_pipe(ssplit, lang):
    if ssplit:
        return stanza.Pipeline(lang, processors="tokenize,pos")
    else:
        return stanza.Pipeline(lang, processors="tokenize,pos", tokenize_no_ssplit=True)

def build_parser_pipes(lang, models):
    """
    Build separate pipelines for each parser model we want to use
    """
    parser_pipes = []
    for model_name in models.split(","):
        if os.path.exists(model_name):
            # if the model name exists as a file, treat it as the path to the model
            pipe = stanza.Pipeline(lang, processors="constituency", constituency_model_path=model_name, constituency_pretagged=True)
        else:
            # otherwise, assume it is a package name?
            pipe = stanza.Pipeline(lang, processors={"constituency": model_name}, constituency_pretagged=True, package=None)
        parser_pipes.append(pipe)
    return parser_pipes

def main():
    args = parse_args()

    random.seed(1234)

    wiki_files = list_wikipedia_files(args.input_dir)
    if args.shuffle:
        random.shuffle(wiki_files)

    tag_pipe = build_tag_pipe(ssplit=True, lang=args.lang)
    parser_pipes = build_parser_pipes(args.lang, args.models)

    # create a blank file.  we will append to this file so that partial results can be used
    with open(args.output_file, "w") as fout:
        pass

    accepted_trees = set()
    for filename in tqdm(wiki_files, disable=False):
        docs = read_wiki_file(filename)
        new_trees = find_matching_trees(docs, args.num_sentences, accepted_trees, tag_pipe, parser_pipes)
        accepted_trees.update(new_trees)

        with open(args.output_file, "a") as fout:
            for tree in sorted(new_trees):
                fout.write(tree)
                fout.write("\n")

if __name__ == '__main__':
    main()
