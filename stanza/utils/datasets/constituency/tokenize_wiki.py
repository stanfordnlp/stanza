"""
A short script to use a Stanza tokenizer to extract tokenized sentences from Wikipedia

The first step is to convert a Wikipedia dataset using Prof. Attardi's wikiextractor:
https://github.com/attardi/wikiextractor

This script then writes out sentences, one per line, whitespace separated
Some common issues with the tokenizer are accounted for by discarding those lines.

Also, to account for languages such as VI where whitespace occurs within words,
spaces are replaced with _  This should not cause any confusion, as any line with
a natural _ in has already been discarded.
"""

import argparse
import logging

import stanza
from stanza.models.common.bert_embedding import load_tokenizer, filter_data
from stanza.utils.datasets.constituency import selftrain_wiki
from stanza.utils.datasets.constituency.selftrain import add_length_args, tokenize_docs
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script that converts part of a wikipedia dump to silver standard trees"
    )
    parser.add_argument(
        '--output_file',
        default='vi_wiki_tokenized.txt',
        help='Where to write the tokenized lines'
    )
    parser.add_argument(
        '--lang',
        default='vi',
        help='Which language tools to use for tokenization and POS'
    )
    parser.add_argument(
        '--input_dir',
        default='extern_data/vietnamese/wikipedia/text/AA',
        help='Path to the wikipedia dump after processing by wikiextractor'
    )
    parser.add_argument(
        '--bert_tokenizer',
        default=None,
        help='Which bert tokenizer (if any) to use to filter long sentences'
    )
    add_length_args(parser)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    files = selftrain_wiki.list_wikipedia_files(args.input_dir)

    if args.bert_tokenizer:
        tokenizer = load_tokenizer(args.bert_tokenizer)
        print("Max model length: %d" % tokenizer.model_max_length)
    pipe = stanza.Pipeline(args.lang, processors="tokenize")

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for filename in tqdm(files):
            docs = selftrain_wiki.read_wiki_file(filename)
            text = tokenize_docs(docs, pipe, args.min_len, args.max_len)
            if args.bert_tokenizer:
                filtered = filter_data(args.bert_tokenizer, [x.split() for x in text], tokenizer, logging.DEBUG)
                text = [" ".join(x) for x in filtered]
            for line in text:
                fout.write(line)
                fout.write("\n")

if __name__ == '__main__':
    main()
