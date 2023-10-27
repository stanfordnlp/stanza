"""Use a Stanza tokenizer to turn a text file into one tokenized paragraph per line

For example, the output of this script is suitable for Glove

Currently this *only* supports tokenization, no MWT splitting.
It also would be beneficial to have an option to convert spaces into
NBSP, underscore, or some other marker to make it easier to process
languages such as VI which have spaces in them
"""


import argparse
import io
import os
import time
import re
import zipfile

import torch

import stanza
from stanza.models.common.utils import open_read_text, default_device
from stanza.models.tokenization.data import TokenizationDataset
from stanza.models.tokenization.utils import output_predictions
from stanza.pipeline.tokenize_processor import TokenizeProcessor
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

NEWLINE_SPLIT_RE = re.compile(r"\n\s*\n")

def tokenize_to_file(tokenizer, fin, fout, chunk_size=500):
    raw_text = fin.read()
    documents = NEWLINE_SPLIT_RE.split(raw_text)
    for chunk_start in tqdm(range(0, len(documents), chunk_size), leave=False):
        chunk_end = min(chunk_start + chunk_size, len(documents))
        chunk = documents[chunk_start:chunk_end]
        in_docs = [stanza.Document([], text=d) for d in chunk]
        out_docs = tokenizer.bulk_process(in_docs)
        for document in out_docs:
            for sent_idx, sentence in enumerate(document.sentences):
                if sent_idx > 0:
                    fout.write(" ")
                fout.write(" ".join(x.text for x in sentence.tokens))
            fout.write("\n")

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="sd", help="Which language to use for tokenization")
    parser.add_argument("--tokenize_model_path", type=str, default=None, help="Specific tokenizer model to use")
    parser.add_argument("input_files", type=str, nargs="+", help="Which input files to tokenize")
    parser.add_argument("--output_file", type=str, default="glove.txt", help="Where to write the tokenized output")
    parser.add_argument("--model_dir", type=str, default=None, help="Where to get models for a Pipeline (None => default models dir)")
    parser.add_argument("--chunk_size", type=int, default=500, help="How many 'documents' to use in a chunk when tokenizing.  This is separate from the tokenizer batching - this limits how much memory gets used at once, since we don't need to store an entire file in memory at once")
    args = parser.parse_args(args=args)

    if os.path.exists(args.output_file):
        print("Cowardly refusing to overwrite existing output file %s" % args.output_file)
        return

    if args.tokenize_model_path:
        config = { "model_path": args.tokenize_model_path,
                   "check_requirements": False }
        tokenizer = TokenizeProcessor(config, pipeline=None, device=default_device())
    else:
        pipe = stanza.Pipeline(lang=args.lang, processors="tokenize", model_dir=args.model_dir)
        tokenizer = pipe.processors["tokenize"]

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for filename in tqdm(args.input_files):
            if filename.endswith(".zip"):
                with zipfile.ZipFile(filename) as zin:
                    input_names = zin.namelist()
                    for input_name in tqdm(input_names, leave=False):
                        with zin.open(input_names[0]) as fin:
                            fin = io.TextIOWrapper(fin, encoding='utf-8')
                            tokenize_to_file(tokenizer, fin, fout)
            else:
                with open_read_text(filename, encoding="utf-8") as fin:
                    tokenize_to_file(tokenizer, fin, fout)

if __name__ == '__main__':
    main()
