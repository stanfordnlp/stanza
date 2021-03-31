"""
Converts raw data files into json files usable by the training script.

Currently it supports converting wikiner datasets, available here:

https://figshare.com/articles/dataset/Learning_multilingual_named_entity_recognition_from_Wikipedia/5462500

Also, Finnish Turku dataset, available here:

https://turkunlp.org/fin-ner.html

TODO: maybe a better name than treebank?  prepare_ner_data is already taken though
"""

import glob
import os
import sys

from stanza.models.common.constant import treebank_to_short_name
import stanza.utils.default_paths as default_paths

from stanza.utils.datasets.ner.preprocess_wikiner import preprocess_wikiner
from stanza.utils.datasets.ner.split_wikiner import split_wikiner
import stanza.utils.datasets.ner.prepare_ner_file as prepare_ner_file

def process_turku(paths):
    short_name = 'fi_turku'
    base_input_path = os.path.join(paths["NERBASE"], short_name)
    base_output_path = paths["NER_DATA_DIR"]
    for shard in ('train', 'dev', 'test'):
        input_filename = os.path.join(base_input_path, '%s.tsv' % shard)
        if not os.path.exists(input_filename):
            raise FileNotFoundError('Cannot find %s component of %s in %s' % (shard, short_name, input_filename))
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        prepare_ner_file.process_dataset(input_filename, output_filename)

def process_wikiner(paths, dataset):
    short_name = treebank_to_short_name(dataset)

    base_input_path = os.path.join(paths["NERBASE"], dataset)
    base_output_path = paths["NER_DATA_DIR"]

    raw_input_path = os.path.join(base_input_path, "raw")
    input_files = glob.glob(os.path.join(raw_input_path, "aij-wikiner*"))
    if len(input_files) == 0:
        raise FileNotFoundError("Could not find any raw wikiner files in %s" % raw_input_path)
    elif len(input_files) > 1:
        raise FileNotFoundError("Found too many raw wikiner files in %s: %s" % (raw_input_path, ", ".join(input_files)))

    csv_file = os.path.join(raw_input_path, "csv_" + short_name)
    print("Converting raw input %s to space separated file in %s" % (input_files[0], csv_file))
    preprocess_wikiner(input_files[0], csv_file)

    # this should create train.bio, dev.bio, and test.bio
    print("Splitting %s to %s" % (csv_file, base_input_path))
    split_wikiner(csv_file, base_input_path)

    for shard in ('train', 'dev', 'test'):
        input_filename = os.path.join(base_input_path, '%s.bio' % shard)
        if not os.path.exists(input_filename):
            raise FileNotFoundError('Cannot find %s component of %s in %s' % (shard, short_name, input_filename))
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        print("Converting %s to %s" % (input_filename, output_filename))
        prepare_ner_file.process_dataset(input_filename, output_filename)


def main():
    paths = default_paths.get_default_paths()

    dataset_name = sys.argv[1]

    if dataset_name == 'fi_turku':
        process_turku(paths)
    elif dataset_name.endswith('WikiNER'):
        process_wikiner(paths, dataset_name)
    else:
        raise ValueError(f"dataset {dataset_name} currently not handled")

if __name__ == '__main__':
    main()
