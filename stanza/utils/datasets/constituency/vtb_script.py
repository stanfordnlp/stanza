"""
Script for processing the VTB files and turning their trees into the desired tree syntax

The VTB original trees are stored in the directory:
VietTreebank_VLSP_SP73/Kho ngu lieu 10000 cay cu phap

The script requires two arguments:
1. Original directory storing the original trees
2. New directory storing the converted trees
"""


import os
import argparse


def convert_file(org_dir, new_dir):
    """
    :param org_dir: original directory storing original trees
    :param new_dir: new directory storing formatted constituency trees

    This function writes new trees to the corresponding files in new_dir
    """
    with open(org_dir, 'r') as reader, open(new_dir, 'w') as writer:
        content = reader.readlines()
        for line in content:
            line = ' '.join(line.split())
            if line == '':
                continue
            elif line == '<s>':
                writer.write('(ROOT ')
            elif line == '</s>':
                writer.write(')\n')
            else:
                writer.write(line)


def main():
    """
    Main function for the script

    Process args, loop through each file in the directory and convert
    to the desired tree format
    """
    parser = argparse.ArgumentParser(
        description="Script that converts a VTB Tree into the desired format",
    )
    parser.add_argument(
        'org_dir',
        help='The location of the original directory storing original trees '
    )
    parser.add_argument(
        'new_dir',
        help='The location of new directory storing the new formatted trees'
    )

    args = parser.parse_args()

    org_dir = args.org_dir
    new_dir = args.new_dir

    for filename in os.listdir(org_dir):
        file_name, file_extension = os.path.splitext(filename)
        # Only convert .prd files, skip the .raw files
        if file_extension == '.raw':
            continue
        file_path = os.path.join(org_dir, filename)
        new_path = os.path.join(new_dir, file_name)
        new_file_path = f'{new_path}.mrg'
        # Convert the tree and write to new_file_path
        convert_file(file_path, new_file_path)


if __name__ == '__main__':
    main()
