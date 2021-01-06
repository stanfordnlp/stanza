"""
SCARE is a dataset of German text with sentiment annotations.

http://romanklinger.de/scare/

To run the script, pass in the directory where scare was unpacked.  It
should have subdirectories scare_v1.0.0 and scare_v1.0.0_text

You need to fill out a license agreement to not redistribute the data
in order to get the data, but the process is not onerous.

Although it sounds interesting, there are unfortunately a lot of very
short items.  Not sure the long items will be enough
"""


import csv
import glob
import os
import random
import sys

import stanza

import scripts.sentiment.process_utils as process_utils

basedir = sys.argv[1]
nlp = stanza.Pipeline('de', processors='tokenize')

text_files = glob.glob(os.path.join(basedir, "scare_v1.0.0_text/annotations/*txt"))
text_id_map = {}
for filename in text_files:
    with open(filename) as fin:
        for line in fin.readlines():
            line = line.strip()
            if not line:
                continue
            key, value = line.split(maxsplit=1)
            if key in text_id_map:
                raise ValueError("Duplicate key {}".format(key))
            text_id_map[key] = value

print(len(text_id_map))

snippets = process_utils.get_scare_snippets(nlp, os.path.join(basedir, "scare_v1.0.0/annotations"), text_id_map)

print(len(snippets))
process_utils.write_list(os.path.join(basedir, "train.txt"), snippets)

