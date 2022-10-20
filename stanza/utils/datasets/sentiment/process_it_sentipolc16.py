"""
Process the SentiPolc dataset from Evalita

Can be run as a standalone script or as a module from
prepare_sentiment_dataset

An option controls how to split up the positive/negative/neutral/mixed classes
"""

import argparse
from enum import Enum
import os
import random
import sys

import stanza
from stanza.utils.datasets.sentiment import process_utils
import stanza.utils.default_paths as default_paths

class Mode(Enum):
    COMBINED = 1
    SEPARATE = 2
    POSITIVE = 3
    NEGATIVE = 4

def main(in_dir, out_dir, short_name, *args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default=Mode.COMBINED, type=lambda x: Mode[x.upper()],
                        help='How to handle mixed vs neutral.  {}'.format(", ".join(x.name for x in Mode)))
    parser.add_argument('--name', default=None, type=str,
                        help='Use a different name to save the dataset.  Useful for keeping POSITIVE & NEGATIVE separate')
    args = parser.parse_args(args=list(*args))

    if args.name is not None:
        short_name = args.name

    nlp = stanza.Pipeline("it", processors='tokenize')

    if args.mode == Mode.COMBINED:
        mapping = {
            ('0', '0'): "1", # neither negative nor positive: neutral
            ('1', '0'): "2", # positive, not negative: positive
            ('0', '1'): "0", # negative, not positive: negative
            ('1', '1'): "1", # mixed combined with neutral
        }
    elif args.mode == Mode.SEPARATE:
        mapping = {
            ('0', '0'): "1", # neither negative nor positive: neutral
            ('1', '0'): "2", # positive, not negative: positive
            ('0', '1'): "0", # negative, not positive: negative
            ('1', '1'): "3", # mixed as a different class
        }
    elif args.mode == Mode.POSITIVE:
        mapping = {
            ('0', '0'): "0", # neutral -> not positive
            ('1', '0'): "1", # positive -> positive
            ('0', '1'): "0", # negative -> not positive
            ('1', '1'): "1", # mixed -> positive
        }
    elif args.mode == Mode.NEGATIVE:
        mapping = {
            ('0', '0'): "0", # neutral -> not negative
            ('1', '0'): "0", # positive -> not negative
            ('0', '1'): "1", # negative -> negative
            ('1', '1'): "1", # mixed -> negative
        }

    print("Using {} scheme to handle the 4 values.  Mapping: {}".format(args.mode, mapping))
    print("Saving to {} using the short name {}".format(out_dir, short_name))

    test_filename = os.path.join(in_dir, "test_set_sentipolc16_gold2000.csv")
    test_snippets = process_utils.read_snippets(test_filename, (2,3), 8, "it", mapping, delimiter=",", skip_first_line=False, quotechar='"', nlp=nlp)

    train_filename = os.path.join(in_dir, "training_set_sentipolc16.csv")
    train_snippets = process_utils.read_snippets(train_filename, (2,3), 8, "it", mapping, delimiter=",", skip_first_line=True, quotechar='"', nlp=nlp)

    random.shuffle(train_snippets)
    dev_len = len(train_snippets) // 10
    dev_snippets = train_snippets[:dev_len]
    train_snippets = train_snippets[dev_len:]

    dataset = (train_snippets, dev_snippets, test_snippets)

    process_utils.write_dataset(dataset, out_dir, short_name)

if __name__ == '__main__':
    paths = default_paths.get_default_paths()
    random.seed(1234)

    in_directory = os.path.join(paths['SENTIMENT_BASE'], "italian", "sentipolc16")
    out_directory = paths['SENTIMENT_DATA_DIR']
    main(in_directory, out_directory, "it_sentipolc16", sys.argv[1:])
