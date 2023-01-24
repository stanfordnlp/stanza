"""
If you want to convert .json back to .bio for some reason, this will do it for you
"""

import argparse
import json
import os
from stanza.models.common.doc import Document
from stanza.models.ner.utils import process_tags
from stanza.utils.default_paths import get_default_paths

def convert_json_to_bio(input_filename, output_filename):
    with open(input_filename, encoding="utf-8") as fin:
        doc = Document(json.load(fin))
    sentences = [[(word.text, word.ner) for word in sentence.tokens] for sentence in doc.sentences]
    sentences = process_tags(sentences, "bioes")
    with open(output_filename, "w", encoding="utf-8") as fout:
        for sentence in sentences:
            for word in sentence:
                fout.write("%s\t%s\n" % word)
            fout.write("\n")

def main(args=None):
    ner_data_dir = get_default_paths()['NER_DATA_DIR']
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filename', type=str, default="data/ner/en_foreign-4class.test.json", help='Convert an individual file')
    parser.add_argument('--input_dir', type=str, default=ner_data_dir, help='Which directory to find the dataset, if using --input_dataset')
    parser.add_argument('--input_dataset', type=str, help='Convert an entire dataset')
    parser.add_argument('--output_suffix', type=str, default='bioes', help='suffix for output filenames')
    args = parser.parse_args(args)

    if args.input_dataset:
        input_filenames = [os.path.join(args.input_dir, "%s.%s.json" % (args.input_dataset, shard))
                           for shard in ("train", "dev", "test")]
    else:
        input_filenames = [args.input_filename]
    for input_filename in input_filenames:
        output_filename = os.path.splitext(input_filename)[0] + "." + args.output_suffix
        print("%s -> %s" % (input_filename, output_filename))
        convert_json_to_bio(input_filename, output_filename)

if __name__ == '__main__':
    main()
