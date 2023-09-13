"""
Turn the ETE results into markdown

Parses blocks like this from the model eval script

2022-01-14 01:23:34 INFO: End to end results for af_afribooms models on af_afribooms test data:
Metric     | Precision |    Recall |  F1 Score | AligndAcc
-----------+-----------+-----------+-----------+-----------
Tokens     |     99.93 |     99.92 |     99.93 |
Sentences  |    100.00 |    100.00 |    100.00 |
Words      |     99.93 |     99.92 |     99.93 |
UPOS       |     97.97 |     97.96 |     97.97 |     98.04
XPOS       |     93.98 |     93.97 |     93.97 |     94.04
UFeats     |     97.23 |     97.22 |     97.22 |     97.29
AllTags    |     93.89 |     93.88 |     93.88 |     93.95
Lemmas     |     97.40 |     97.39 |     97.39 |     97.46
UAS        |     87.39 |     87.38 |     87.38 |     87.45
LAS        |     83.57 |     83.56 |     83.57 |     83.63
CLAS       |     76.88 |     76.45 |     76.66 |     76.52
MLAS       |     72.28 |     71.87 |     72.07 |     71.94
BLEX       |     73.20 |     72.79 |     73.00 |     72.86


Turns them into a markdown table.

Included is an attempt to mark the default packages with a green check.
  <i class="fas fa-check" style="color:#33a02c"></i>
"""

import argparse

from stanza.models.common.constant import pretty_langcode_to_lang
from stanza.models.common.short_name_to_treebank import short_name_to_treebank
from stanza.utils.training.run_ete import RESULTS_STRING
from stanza.resources.default_packages import default_treebanks

EXPECTED_ORDER = ["Tokens", "Sentences", "Words", "UPOS", "XPOS", "UFeats", "AllTags", "Lemmas", "UAS", "LAS", "CLAS", "MLAS", "BLEX"]

parser = argparse.ArgumentParser()
parser.add_argument("filenames", type=str, nargs="+", help="Which file(s) to read")
args = parser.parse_args()

lines = []
for filename in args.filenames:
    with open(filename) as fin:
        lines.extend(fin.readlines())

blocks = []
index = 0
while index < len(lines):
    line = lines[index]
    if line.find(RESULTS_STRING) < 0:
        index = index + 1
        continue

    line = line[line.find(RESULTS_STRING) + len(RESULTS_STRING):].strip()
    short_name = line.split()[0]

    # skip the header of the expected output
    index = index + 1
    line = lines[index]
    pieces = line.split("|")
    assert pieces[0].strip() == 'Metric', "output format changed?"
    assert pieces[3].strip() == 'F1 Score', "output format changed?"

    index = index + 1
    line = lines[index]
    assert line.startswith("-----"), "output format changed?"

    index = index + 1

    block = lines[index:index+13]
    assert len(block) == 13
    index = index + 13

    block = [x.split("|") for x in block]
    assert all(x[0].strip() == y for x, y in zip(block, EXPECTED_ORDER)), "output format changed?"
    lcode, short_dataset = short_name.split("_", 1)
    language = pretty_langcode_to_lang(lcode)
    treebank = short_name_to_treebank(short_name)
    long_dataset = treebank.split("-")[-1]

    checkmark = ""
    if default_treebanks[lcode] == short_dataset:
        checkmark = '<i class="fas fa-check" style="color:#33a02c"></i>'

    block = [language, "[%s](%s)" % (long_dataset, "https://github.com/UniversalDependencies/%s" % treebank), lcode, checkmark] + [x[3].strip() for x in block]
    blocks.append(block)

PREFIX = ["Macro Avg", "", "", ""]

avg = [sum(float(x[i]) for x in blocks) / len(blocks) for i in range(len(PREFIX), len(EXPECTED_ORDER) + len(PREFIX))]
avg = PREFIX + ["%.2f" % x for x in avg]
blocks = sorted(blocks)
blocks = [avg] + blocks

chart = ["|%s|" % "  |  ".join(x) for x in blocks]
for line in chart:
    print(line)

