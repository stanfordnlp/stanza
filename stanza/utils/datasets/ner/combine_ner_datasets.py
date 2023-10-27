import argparse

from stanza.utils.default_paths import get_default_paths
from stanza.utils.datasets.ner.utils import combine_dataset

SHARDS = ("train", "dev", "test")

def main(args=None):
    ner_data_dir = get_default_paths()['NER_DATA_DIR']

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dataset', type=str, help='What dataset to output')
    parser.add_argument('input_datasets', type=str, nargs='+', help='Which datasets to input')

    parser.add_argument('--input_dir', type=str, default=ner_data_dir, help='Which directory to find the datasets')
    parser.add_argument('--output_dir', type=str, default=ner_data_dir, help='Which directory to write the dataset')
    args = parser.parse_args(args)

    input_dir = args.input_dir
    output_dir = args.output_dir

    combine_dataset(input_dir, output_dir, args.input_datasets, args.output_dataset)

if __name__ == '__main__':
    main()
