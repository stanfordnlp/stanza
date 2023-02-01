import argparse
import json
import os

from stanza.utils.default_paths import get_default_paths
from stanza.utils.datasets.ner.utils import write_dataset

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

    datasets = []
    for shard in SHARDS:
        full_dataset = []
        for input_dataset in args.input_datasets:
            input_filename = "%s.%s.json" % (input_dataset, shard)
            input_path = os.path.join(input_dir, input_filename)
            with open(input_path, encoding="utf-8") as fin:
                dataset = json.load(fin)
                converted = [[(word['text'], word['ner']) for word in sentence] for sentence in dataset]
                full_dataset.extend(converted)
        datasets.append(full_dataset)
    write_dataset(datasets, output_dir, args.output_dataset)

if __name__ == '__main__':
    main()
