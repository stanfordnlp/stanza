"""
Combines OntoNotes and WW into a single dataset with OntoNotes used for dev & test

The resulting dataset has two layers saved in the multi_ner column.

WW is kept as 9 classes, with the tag put in either the first or
second layer depending on the flags.

OntoNotes is converted to one column for 18 and one column for 9 classes.
"""

import argparse
import json
import os
import shutil

from stanza.utils import default_paths
from stanza.utils.datasets.ner.utils import combine_files
from stanza.utils.datasets.ner.simplify_ontonotes_to_worldwide import simplify_ontonotes_to_worldwide

def convert_ontonotes_file(filename, simplify, bigger_first):
    assert "en_ontonotes" in filename
    if not os.path.exists(filename):
        raise FileNotFoundError("Cannot convert missing file %s" % filename)
    new_filename = filename.replace("en_ontonotes", "en_ontonotes-multi")

    with open(filename) as fin:
        doc = json.load(fin)

    for sentence in doc:
        for word in sentence:
            ner = word['ner']
            if simplify:
                simplified = simplify_ontonotes_to_worldwide(ner)
            else:
                simplified = "-"
            if bigger_first:
                word['multi_ner'] = (ner, simplified)
            else:
                word['multi_ner'] = (simplified, ner)

    with open(new_filename, "w") as fout:
        json.dump(doc, fout, indent=2)

def convert_worldwide_file(filename, bigger_first):
    assert "en_worldwide-9class" in filename
    if not os.path.exists(filename):
        raise FileNotFoundError("Cannot convert missing file %s" % filename)

    new_filename = filename.replace("en_worldwide-9class", "en_worldwide-9class-multi")

    with open(filename) as fin:
        doc = json.load(fin)

    for sentence in doc:
        for word in sentence:
            ner = word['ner']
            if bigger_first:
                word['multi_ner'] = ("-", ner)
            else:
                word['multi_ner'] = (ner, "-")

    with open(new_filename, "w") as fout:
        json.dump(doc, fout, indent=2)

def build_multitag_dataset(base_output_path, short_name, simplify, bigger_first):
    convert_ontonotes_file(os.path.join(base_output_path, "en_ontonotes.train.json"), simplify, bigger_first)
    convert_ontonotes_file(os.path.join(base_output_path, "en_ontonotes.dev.json"), simplify, bigger_first)
    convert_ontonotes_file(os.path.join(base_output_path, "en_ontonotes.test.json"), simplify, bigger_first)

    convert_worldwide_file(os.path.join(base_output_path, "en_worldwide-9class.train.json"), bigger_first)
    convert_worldwide_file(os.path.join(base_output_path, "en_worldwide-9class.dev.json"), bigger_first)
    convert_worldwide_file(os.path.join(base_output_path, "en_worldwide-9class.test.json"), bigger_first)

    combine_files(os.path.join(base_output_path, "%s.train.json" % short_name),
                  os.path.join(base_output_path, "en_ontonotes-multi.train.json"),
                  os.path.join(base_output_path, "en_worldwide-9class-multi.train.json"))
    shutil.copyfile(os.path.join(base_output_path, "en_ontonotes-multi.dev.json"),
                    os.path.join(base_output_path, "%s.dev.json" % short_name))
    shutil.copyfile(os.path.join(base_output_path, "en_ontonotes-multi.test.json"),
                    os.path.join(base_output_path, "%s.test.json" % short_name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_simplify', dest='simplify', action='store_false', help='By default, this script will simplify the OntoNotes 18 classes to the 8 WorldWide classes in a second column.  Turning that off will leave that column blank.  Initial experiments with that setting were very bad, though')
    parser.add_argument('--no_bigger_first', dest='bigger_first', action='store_false', help='By default, this script will put the 18 class tags in the first column and the 8 in the second.  This flips the order')
    args = parser.parse_args()

    paths = default_paths.get_default_paths()
    base_output_path = paths["NER_DATA_DIR"]

    build_multitag_dataset(base_output_path, "en_ontonotes-ww-multi", args.simplify, args.bigger_first)

if __name__ == '__main__':
    main()

