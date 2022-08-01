"""
Turns an Oscar 2022 jsonl file to text

YOU DO NOT NEED THIS if you use the oscar extractor which reads from
HuggingFace, dump_oscar.py

to run:
python3 -m stanza.utils.charlm.oscar_to_text <path> ...

each path can be a file or a directory with multiple .jsonl files in it
"""

import glob
import json
import lzma
import os
import sys
from stanza.models.common.utils import open_read_text

def extract_file(input_filename):
    print("Extracting %s" % input_filename)
    json_idx = input_filename.rfind(".jsonl")
    if json_idx < 0:
        output_filename = input_filename + ".txt.xz"
    else:
        output_filename = input_filename[:json_idx] + ".txt.xz"
    print("Writing content to %s" % output_filename)
    with open_read_text(input_filename) as fin:
        with lzma.open(output_filename, "wt") as fout:
            for line in fin:
                content = json.loads(line)
                content = content['content']

                fout.write(content)
                fout.write("\n\n")

def main():
    """
    Go through each of the given filenames or directories, convert json to .txt.xz
    """
    for filename in sys.argv[1:]:
        if os.path.isfile(filename):
            extract_file(filename)
        elif os.path.isdir(filename):
            files = glob.glob(os.path.join(filename, "*jsonl*"))
            files = sorted([x for x in files if os.path.isfile(x)])
            print("Found %d files:" % len(files))
            if len(files) > 0:
                print("  %s" % "\n  ".join(files))
            for json_filename in files:
                extract_file(json_filename)

if __name__ == "__main__":
    main()
