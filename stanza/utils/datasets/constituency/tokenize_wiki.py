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

import stanza
from stanza.models.common.utils import get_tqdm
from stanza.utils.datasets.constituency import selftrain_wiki

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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    files = selftrain_wiki.list_wikipedia_files(args.input_dir)

    pipe = stanza.Pipeline(args.lang, processors="tokenize")

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for filename in tqdm(files):
            docs = selftrain_wiki.read_wiki_file(filename)
            docs = [stanza.Document([], text=t) for t in docs]
            pipe(docs)

            for doc in docs:
                for sentence in doc.sentences:
                    text = sentence.text
                    if (text.find("|") >= 0 or text.find("_") >= 0 or
                        text.find("<") >= 0 or text.find(">") >= 0 or
                        text.find("[") >= 0 or text.find("]") >= 0):
                        continue
                    # the VI tokenizer in particular doesn't split these well
                    if any(any(w.text.find(c) >= 0 and len(w.text) > 1 for w in sentence.words)
                           for c in '"()'):
                        continue
                    text = [w.text.replace(" ", "_") for w in sentence.words]
                    text = " ".join(text)
                    fout.write(text)
                    fout.write("\n")

if __name__ == '__main__':
    main()
