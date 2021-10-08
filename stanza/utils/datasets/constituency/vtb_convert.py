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


def is_closed_tree(tree):
    """
    Checks if the tree is properly closed
    :param tree: tree as a string
    :return: True if closed otherwise False
    """
    count = 0
    for char in tree:
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
    return count == 0


def is_valid_line(line):
    """
    Check if a line being read is a valid constituent
    :param line: constituent being read
    :return: True if it has open and closing parenthesis. This can be done better with regex but requires extra
    work and knowledge of the constituents tags
    """
    if line.startswith('(') and line.endswith(')'):
        return True

    return False


def convert_file(org_dir, new_dir):
    """
    :param org_dir: original directory storing original trees
    :param new_dir: new directory storing formatted constituency trees

    This function writes new trees to the corresponding files in new_dir
    """
    with open(org_dir, 'r') as reader, open(new_dir, 'w') as writer:
        content = reader.readlines()
        # Tree string will only be written if the currently read
        # tree is a valid tree. It will not be written if it
        # does not have a '(' that signifies the presence of constituents
        tree = ""
        reading_tree = False
        for line in content:
            line = ' '.join(line.split())
            if line == '':
                continue
            elif line == '<s>':
                tree = ""
                tree += '(ROOT '
                reading_tree = True
            elif line == '</s>' and reading_tree:
                # one tree in 25432.prd is not valid because
                # it is just a bunch of blank lines
                if tree.strip() == '(ROOT':
                    tree = ""
                    continue
                tree += ')\n'
                if not is_closed_tree(tree):
                    tree = ""
                    continue
                # TODO: this eliminates 5 trees.  maybe those
                # trees can be salvaged?
                if tree.find("WP") >= 0:
                    continue
                tree = tree.replace("SE-SPL", "S-SPL")
                writer.write(tree)
                reading_tree = False
                tree = ""
            else:
                if is_valid_line(line) and reading_tree:
                    tree += line
                else:
                    tree = ""
                    reading_tree = False


def convert_dir(org_dir, new_dir):
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

    convert_dir(org_dir, new_dir)


if __name__ == '__main__':
    main()
