"""
A short script to use a Stanza tokenizer to extract tokenized sentences from Wikipedia

The first step is to convert a Wikipedia dataset using Prof. Attardi's wikiextractor:
https://github.com/attardi/wikiextractor

This script then writes out sentences, one per line, whitespace separated
Some common issues with the tokenizer are accounted for by discarding those lines.

Also, to account for languages such as VI where whitespace occurs within words,
spaces are replaced with _  This should not cause any confusion, as any line with
a natural _ in has already been discarded.

for i in `echo A B C D E F G H I J K`; do nlprun "python3 stanza/utils/datasets/constituency/tokenize_wiki.py --output_file /u/nlp/data/constituency-parser/italian/2024_wiki_tokenization/it_wiki_tokenized_B$i.txt --lang it --max_len 120 --input_dir /u/nlp/data/Wikipedia/itwiki/B$i --tokenizer_model saved_models/tokenize/it_combined_tokenizer.pt --download_method None" -o /u/nlp/data/constituency-parser/italian/2024_wiki_tokenization/it_wiki_tokenized_B$i.out; done
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

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input_dir',
        default=None,
        help='Path to the wikipedia dump after processing by wikiextractor'
    )
    input_group.add_argument(
        '--input_file',
        default=None,
        help='Path to a single file of the wikipedia dump after processing by wikiextractor'
    )
    parser.add_argument(
        '--bert_tokenizer',
        default=None,
        help='Which bert tokenizer (if any) to use to filter long sentences'
    )
    parser.add_argument(
        '--tokenizer_model',
        default=None,
        help='Use this model instead of the current Stanza tokenizer for this language'
    )
    parser.add_argument(
        '--download_method',
        default=None,
        help='Download pipeline models using this method (defaults to downloading updates from HF)'
    )
    add_length_args(parser)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.input_dir is not None:
        files = selftrain_wiki.list_wikipedia_files(args.input_dir)
    elif args.input_file is not None:
        files = [args.input_file]
    else:
        raise ValueError("Need to specify at least one file or directory!")

    if args.bert_tokenizer:
        tokenizer = load_tokenizer(args.bert_tokenizer)
        print("Max model length: %d" % tokenizer.model_max_length)
    pipeline_args = {}
    if args.tokenizer_model:
        pipeline_args["tokenize_model_path"] = args.tokenizer_model
    if args.download_method:
        pipeline_args["download_method"] = args.download_method
    pipe = stanza.Pipeline(args.lang, processors="tokenize", **pipeline_args)

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
