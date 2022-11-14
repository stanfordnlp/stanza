"""
Turns an Oscar 2022 jsonl file to text

YOU DO NOT NEED THIS if you use the oscar extractor which reads from
HuggingFace, dump_oscar.py

to run:
python3 -m stanza.utils.charlm.oscar_to_text <path> ...

each path can be a file or a directory with multiple .jsonl files in it
"""

import argparse
import glob
import json
import lzma
import os
import sys
from stanza.models.common.utils import open_read_text

def extract_file(output_directory, input_filename, use_xz):
    print("Extracting %s" % input_filename)
    if output_directory is None:
        output_directory, output_filename = os.path.split(input_filename)
    else:
        _, output_filename = os.path.split(input_filename)

    json_idx = output_filename.rfind(".jsonl")
    if json_idx < 0:
        output_filename = output_filename + ".txt"
    else:
        output_filename = output_filename[:json_idx] + ".txt"
    if use_xz:
        output_filename += ".xz"
        open_file = lambda x: lzma.open(x, "wt", encoding="utf-8")
    else:
        open_file = lambda x: open(x, "w", encoding="utf-8")

    output_filename = os.path.join(output_directory, output_filename)
    print("Writing content to %s" % output_filename)
    with open_read_text(input_filename) as fin:
        with open_file(output_filename) as fout:
            for line in fin:
                content = json.loads(line)
                content = content['content']

                fout.write(content)
                fout.write("\n\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="Output directory for saving files.  If None, will write to the original directory")
    parser.add_argument("--no_xz", default=True, dest="xz", action="store_false", help="Don't use xz to compress the output files")
    parser.add_argument("filenames", nargs="+", help="Filenames or directories to process")
    args = parser.parse_args()
    return args

def main():
    """
    Go through each of the given filenames or directories, convert json to .txt.xz
    """
    args = parse_args()
    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)
    for filename in args.filenames:
        if os.path.isfile(filename):
            extract_file(args.output, filename, args.xz)
        elif os.path.isdir(filename):
            files = glob.glob(os.path.join(filename, "*jsonl*"))
            files = sorted([x for x in files if os.path.isfile(x)])
            print("Found %d files:" % len(files))
            if len(files) > 0:
                print("  %s" % "\n  ".join(files))
            for json_filename in files:
                extract_file(args.output, json_filename, args.xz)

if __name__ == "__main__":
    main()
