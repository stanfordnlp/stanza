"""
From a directory of files with VTB Trees, split into train/dev/test set
with a split of 70/15/15

The script requires two arguments
1. org_dir: the original directory obtainable from running vtb_script.py
2. split_dir: the directory where the train/dev/test splits will be stored
"""

import os
import argparse
import random


random.seed(1234)


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


def create_dirs(split_dir):
    """
    This function creates the necessary directories for the train/dev/test splits
    :param split_dir: directory that stores the splits
    :return: train directory, dev directory, test directory
    """
    train_dir = os.path.join(split_dir, 'train.mrg')
    dev_dir = os.path.join(split_dir, 'dev.mrg')
    test_dir = os.path.join(split_dir, 'test.mrg')

    with open(train_dir, mode='a') as tr, open(dev_dir, mode='a') as de, open(test_dir, mode='a') as te:
        pass

    return train_dir, dev_dir, test_dir


def main():
    """
    Main function for the script

    Process args, loop through each file in the directory and convert
    to the desired tree format
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

    # Create a random shuffle list of the file names in the original directory
    file_names = create_shuffle_list(org_dir)

    # Create train_dir, dev_dir, test_dir
    train_dir, dev_dir, test_dir = create_dirs(split_dir)

    # Set up the number of samples for each train/dev/test set
    num_samples = 10471
    stop_train = int(num_samples * 0.7)
    stop_dev = int(num_samples * 0.85)

    # Write directory and write count
    write_dir = train_dir
    count = 0

    # Loop through the files, which then loop through the trees and write to write_dir
    for filename in file_names:
        # Skip files that are not .mrg
        if not filename.endswith('.mrg'):
            continue
        # File is .mrg. Start processing
        file_dir = os.path.join(org_dir, filename)
        with open(file_dir, 'r') as reader, open(write_dir, 'a') as writer:
            content = reader.readlines()
            for line in content:
                # Write to write_dir
                writer.write(line)
                # Check current count to switch write_dir
                count += 1
                # Switch to writing dev set
                if count > stop_train:
                    write_dir = dev_dir
                # Switch to writing test set
                if count > stop_dev:
                    write_dir = test_dir


if __name__ == '__main__':
    main()
