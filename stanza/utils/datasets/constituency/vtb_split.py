"""
From a directory of files with VTB Trees, split into train/dev/test set
with a split of 70/15/15

The script requires two arguments
1. org_dir: the original directory obtainable from running vtb_convert.py
2. split_dir: the directory where the train/dev/test splits will be stored
"""

import os
import argparse
import random


def create_shuffle_list(org_dir):
    """
    This function creates the random order with which we use to loop through the files
    :param org_dir: original directory storing the files that store the trees
    :return: list of file names randomly shuffled
    """
    file_names = []
    for filename in os.listdir(org_dir):
        file_names.append(filename)
    random.shuffle(file_names)

    return file_names


def create_paths(split_dir, short_name):
    """
    This function creates the necessary paths for the train/dev/test splits
    :param split_dir: directory that stores the splits
    :return: train path, dev path, test path
    """
    if not short_name:
        short_name = ""
    elif not short_name.endswith("_"):
        short_name = short_name + "_"

    train_path = os.path.join(split_dir, '%strain.mrg' % short_name)
    dev_path = os.path.join(split_dir, '%sdev.mrg' % short_name)
    test_path = os.path.join(split_dir, '%stest.mrg' % short_name)

    return train_path, dev_path, test_path


def get_num_samples(org_dir, file_names):
    """
    Function for obtaining the number of samples
    :param org_dir: original directory storing the tree files
    :param file_names: list of file names in the directory
    :return: number of samples
    """
    count = 0
    # Loop through the files, which then loop through the trees
    for filename in file_names:
        # Skip files that are not .mrg
        if not filename.endswith('.mrg'):
            continue
        # File is .mrg. Start processing
        file_dir = os.path.join(org_dir, filename)
        with open(file_dir, 'r') as reader:
            content = reader.readlines()
            for _ in content:
                count += 1

    return count

def split_files(org_dir, split_dir, short_name=None):
    # Create a random shuffle list of the file names in the original directory
    file_names = create_shuffle_list(org_dir)

    # Create train_path, dev_path, test_path
    train_path, dev_path, test_path = create_paths(split_dir, short_name)
    output_names = (train_path, dev_path, test_path)

    # Set up the number of samples for each train/dev/test set
    num_samples = get_num_samples(org_dir, file_names)
    stop_train = int(num_samples * 0.7)
    stop_dev = int(num_samples * 0.85)
    output_limits = (stop_train, stop_dev, num_samples)

    # Count how much stuff we've written.
    # We will switch to the next output file when we're written enough
    count = 0

    filename_iter = iter(file_names)
    for write_path, count_limit in zip(output_names, output_limits):
        with open(write_path, 'w') as writer:
            # Loop through the files, which then loop through the trees and write to write_path
            while count < count_limit:
                filename = next(filename_iter, None)
                # Skip files that are not .mrg
                if not filename.endswith('.mrg'):
                    continue
                # File is .mrg. Start processing
                file_dir = os.path.join(org_dir, filename)
                with open(file_dir, 'r') as reader:
                    content = reader.readlines()
                    for line in content:
                        # Write to write_dir
                        writer.write(line)
                        count += 1

def main():
    """
    Main function for the script

    Process args, loop through each tree in each file in the directory
    and write the trees to the train/dev/test split with a split of
    70/15/15
    """
    parser = argparse.ArgumentParser(
        description="Script that splits a list of files of vtb trees into train/dev/test sets",
    )
    parser.add_argument(
        'org_dir',
        help='The location of the original directory storing correctly formatted vtb trees '
    )
    parser.add_argument(
        'split_dir',
        help='The location of new directory storing the train/dev/test set'
    )

    args = parser.parse_args()

    org_dir = args.org_dir
    split_dir = args.split_dir

    random.seed(1234)

    split_files(org_dir, split_dir)

if __name__ == '__main__':
    main()
