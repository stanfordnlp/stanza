"""
Builds a combined model out of OntoNotes, WW, and CoNLL.

This is done with three layers in the multi_ner column:

First layer is OntoNotes only.  Other datasets have that left as blank.

Second layer is the 9 class WW dataset.  OntoNotes is reduced to 9 classes for this column.

Third column is the CoNLL dataset.  OntoNotes and WW are both projected to this.
"""

import json
import os
import shutil

from stanza.utils import default_paths
from stanza.utils.datasets.ner.simplify_en_worldwide import process_label
from stanza.utils.datasets.ner.simplify_ontonotes_to_worldwide import simplify_ontonotes_to_worldwide
from stanza.utils.datasets.ner.utils import combine_files

def convert_ontonotes_file(filename, short_name):
    assert "en_ontonotes." in filename
    if not os.path.exists(filename):
        raise FileNotFoundError("Cannot convert missing file %s" % filename)
    new_filename = filename.replace("en_ontonotes.", short_name + ".ontonotes.")

    with open(filename) as fin:
        doc = json.load(fin)

    for sentence in doc:
        is_start = False
        for word in sentence:
            text = word['text']
            ner = word['ner']
            s9 = simplify_ontonotes_to_worldwide(ner)
            _, s4, is_start = process_label((text, s9), is_start)
            word['multi_ner'] = (ner, s9, s4)

    with open(new_filename, "w") as fout:
        json.dump(doc, fout, indent=2)

def convert_worldwide_file(filename, short_name):
    assert "en_worldwide-9class." in filename
    if not os.path.exists(filename):
        raise FileNotFoundError("Cannot convert missing file %s" % filename)
    new_filename = filename.replace("en_worldwide-9class.", short_name + ".worldwide-9class.")

    with open(filename) as fin:
        doc = json.load(fin)

    for sentence in doc:
        is_start = False
        for word in sentence:
            text = word['text']
            ner = word['ner']
            _, s4, is_start = process_label((text, ner), is_start)
            word['multi_ner'] = ("-", ner, s4)

    with open(new_filename, "w") as fout:
        json.dump(doc, fout, indent=2)

def convert_conll03_file(filename, short_name):
    assert "en_conll03." in filename
    if not os.path.exists(filename):
        raise FileNotFoundError("Cannot convert missing file %s" % filename)
    new_filename = filename.replace("en_conll03.", short_name + ".conll03.")

    with open(filename) as fin:
        doc = json.load(fin)

    for sentence in doc:
        for word in sentence:
            ner = word['ner']
            word['multi_ner'] = ("-", "-", ner)

    with open(new_filename, "w") as fout:
        json.dump(doc, fout, indent=2)

def build_combined_dataset(base_output_path, short_name):
    convert_ontonotes_file(os.path.join(base_output_path, "en_ontonotes.train.json"), short_name)
    convert_ontonotes_file(os.path.join(base_output_path, "en_ontonotes.dev.json"), short_name)
    convert_ontonotes_file(os.path.join(base_output_path, "en_ontonotes.test.json"), short_name)

    convert_worldwide_file(os.path.join(base_output_path, "en_worldwide-9class.train.json"), short_name)
    convert_conll03_file(os.path.join(base_output_path, "en_conll03.train.json"), short_name)

    combine_files(os.path.join(base_output_path, "%s.train.json" % short_name),
                  os.path.join(base_output_path, "en_combined.ontonotes.train.json"),
                  os.path.join(base_output_path, "en_combined.worldwide-9class.train.json"),
                  os.path.join(base_output_path, "en_combined.conll03.train.json"))
    shutil.copyfile(os.path.join(base_output_path, "en_combined.ontonotes.dev.json"),
                    os.path.join(base_output_path, "%s.dev.json" % short_name))
    shutil.copyfile(os.path.join(base_output_path, "en_combined.ontonotes.test.json"),
                    os.path.join(base_output_path, "%s.test.json" % short_name))


def main():
    paths = default_paths.get_default_paths()
    base_output_path = paths["NER_DATA_DIR"]

    build_combined_dataset(base_output_path, "en_combined")

if __name__ == '__main__':
    main()
