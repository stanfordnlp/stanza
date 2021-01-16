"""
Converts raw data files into json files usable by the training script.

TODO: maybe a better name than treebank?  prepare_ner_data is already taken though
"""

import os
import sys

import stanza.utils.default_paths as default_paths
import stanza.utils.datasets.prepare_ner_data as prepare_ner_data

def process_turku(paths):
    short_name = 'fi_turku'
    base_input_path = os.path.join(paths["NERBASE"], short_name)
    base_output_path = paths["NER_DATA_DIR"]
    for dataset in ('train', 'dev', 'test'):
        input_filename = os.path.join(base_input_path, '%s.tsv' % dataset)
        if not os.path.exists(input_filename):
            raise FileNotFoundError('Cannot find %s component of %s in %s' % (dataset, short_name, input_filename))
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, dataset))
        prepare_ner_data.process_dataset(input_filename, output_filename)

def main():
    paths = default_paths.get_default_paths()

    dataset_name = sys.argv[1]

    if dataset_name == 'fi_turku':
        process_turku(paths)
    else:
        raise ValueError(f"dataset {dataset_name} currently not handled")

if __name__ == '__main__':
    main()
