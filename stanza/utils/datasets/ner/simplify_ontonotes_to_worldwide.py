"""
Simplify an existing ner json with the OntoNotes 18 class scheme to the Worldwide scheme

Simplified classes used in the Worldwide dataset are:

Date
Facility
Location
Misc
Money
NORP
Organization
Person
Product

vs OntoNotes classes:

CARDINAL
DATE
EVENT
FAC
GPE
LANGUAGE
LAW
LOC
MONEY
NORP
ORDINAL
ORG
PERCENT
PERSON
PRODUCT
QUANTITY
TIME
WORK_OF_ART
"""

import argparse
import glob
import json
import os

from stanza.utils.default_paths import get_default_paths

WORLDWIDE_ENTITY_MAPPING = {
    "CARDINAL":    None,
    "ORDINAL":     None,
    "PERCENT":     None,
    "QUANTITY":    None,
    "TIME":        None,

    "DATE":        "Date",
    "EVENT":       "Misc",
    "FAC":         "Facility",
    "GPE":         "Location",
    "LANGUAGE":    "NORP",
    "LAW":         "Misc",
    "LOC":         "Location",
    "MONEY":       "Money",
    "NORP":        "NORP",
    "ORG":         "Organization",
    "PERSON":      "Person",
    "PRODUCT":     "Product",
    "WORK_OF_ART": "Misc",

    # identity map in case this is called on the Worldwide half of the tags
    "Date":        "Date",
    "Facility":    "Facility",
    "Location":    "Location",
    "Misc":        "Misc",
    "Money":       "Money",
    "Organization":"Organization",
    "Person":      "Person",
    "Product":     "Product",
}

def simplify_ontonotes_to_worldwide(entity):
    if not entity or entity == "O":
        return "O"

    ent_iob, ent_type = entity.split("-", maxsplit=1)

    if ent_type in WORLDWIDE_ENTITY_MAPPING:
        if not WORLDWIDE_ENTITY_MAPPING[ent_type]:
            return "O"
        return ent_iob + "-" + WORLDWIDE_ENTITY_MAPPING[ent_type]
    raise ValueError("Unhandled entity: %s" % ent_type)

def convert_file(in_file, out_file):
    with open(in_file) as fin:
        gold_doc = json.load(fin)

    for sentence in gold_doc:
        for word in sentence:
            if 'ner' not in word:
                continue
            word['ner'] = simplify_ontonotes_to_worldwide(word['ner'])

    with open(out_file, "w", encoding="utf-8") as fout:
        json.dump(gold_doc, fout, indent=2)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dataset', type=str, default='en_ontonotes', help='which files to convert')
    parser.add_argument('--output_dataset', type=str, default='en_ontonotes-8class', help='which files to write out')
    parser.add_argument('--ner_data_dir', type=str, default=get_default_paths()["NER_DATA_DIR"], help='which directory has the data')
    args = parser.parse_args()

    input_files = glob.glob(os.path.join(args.ner_data_dir, args.input_dataset + ".*"))
    for input_file in input_files:
        output_file = os.path.split(input_file)[1][len(args.input_dataset):]
        output_file = os.path.join(args.ner_data_dir, args.output_dataset + output_file)
        print("Converting %s to %s" % (input_file, output_file))
        convert_file(input_file, output_file)

    
if __name__ == '__main__':
    main()
