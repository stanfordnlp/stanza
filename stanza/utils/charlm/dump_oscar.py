"""
This script downloads and extracts the text from an Oscar crawl on HuggingFace

To use, just run

dump_oscar.py <lang>

It will download the dataset and output all of the text to the --output directory.
Files will be broken into pieces to avoid having one giant file.
By default, files will also be compressed with xz (although this can be turned off)
"""

import argparse
import lzma
import math
import os

from tqdm import tqdm

from datasets import get_dataset_split_names
from datasets import load_dataset

from stanza.models.common.constant import lang_to_langcode

def parse_args():
    """
    A few specific arguments for the dump program

    Uses lang_to_langcode to process args.language, hopefully converting
    a variety of possible formats to the short code used by HuggingFace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("language", help="Language to download")
    parser.add_argument("--output", default="oscar_dump", help="Path for saving files")
    parser.add_argument("--no_xz", dest="xz", default=True, action='store_false', help="Don't xz the files - default is to compress while writing")
    parser.add_argument("--prefix", default="oscar_dump", help="Prefix to use for the pieces of the dataset")
    parser.add_argument("--version", choices=["2019", "2023"], default="2023", help="Which version of the Oscar dataset to download")

    args = parser.parse_args()
    args.language = lang_to_langcode(args.language)
    return args

def download_2023(args):
    dataset = load_dataset('oscar-corpus/OSCAR-2301', 'sd')
    split_names = list(dataset.keys())


def main():
    args = parse_args()

    # this is the 2019 version.  for 2023, you can do
    # dataset = load_dataset('oscar-corpus/OSCAR-2301', 'sd')
    language = args.language
    if args.version == "2019":
        dataset_name = "unshuffled_deduplicated_%s" % language
        try:
            split_names = get_dataset_split_names("oscar", dataset_name)
        except ValueError as e:
            raise ValueError("Language %s not available in HuggingFace Oscar" % language) from e

        if len(split_names) > 1:
            raise ValueError("Unexpected split_names: {}".format(split_names))

        dataset = load_dataset("oscar", dataset_name)
        dataset = dataset[split_names[0]]
        size_in_bytes = dataset.info.size_in_bytes
        process_item = lambda x: x['text']
    elif args.version == "2023":
        dataset = load_dataset("oscar-corpus/OSCAR-2301", language)
        split_names = list(dataset.keys())
        if len(split_names) > 1:
            raise ValueError("Unexpected split_names: {}".format(split_names))
        # it's not clear if some languages don't support size_in_bytes,
        # or if there was an update to datasets which now allows that
        #
        # previously we did:
        #  dataset = dataset[split_names[0]]['text']
        #  size_in_bytes = sum(len(x) for x in dataset)
        #  process_item = lambda x: x
        dataset = dataset[split_names[0]]
        size_in_bytes = dataset.info.size_in_bytes
        process_item = lambda x: x['text']
    else:
        raise AssertionError("Unknown version: %s" % args.version)

    chunks = max(1.0, size_in_bytes // 1e8) # an overestimate
    id_len = max(3, math.floor(math.log10(chunks)) + 1)

    if args.xz:
        format_str = "%s_%%0%dd.txt.xz" % (args.prefix, id_len)
        fopen = lambda file_idx: lzma.open(os.path.join(args.output, format_str % file_idx), "wt")
    else:
        format_str = "%s_%%0%dd.txt" % (args.prefix, id_len)
        fopen = lambda file_idx: open(os.path.join(args.output, format_str % file_idx), "w")

    print("Writing dataset to %s" % args.output)
    print("Dataset length: {}".format(size_in_bytes))
    os.makedirs(args.output, exist_ok=True)

    file_idx = 0
    file_len = 0
    total_len = 0
    fout = fopen(file_idx)

    for item in tqdm(dataset):
        text = process_item(item)
        fout.write(text)
        fout.write("\n")
        file_len += len(text)
        file_len += 1
        if file_len > 1e8:
            file_len = 0
            fout.close()
            file_idx = file_idx + 1
            fout = fopen(file_idx)

    fout.close()

if __name__ == '__main__':
    main()
