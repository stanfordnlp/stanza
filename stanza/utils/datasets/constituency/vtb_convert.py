"""
Script for processing the VTB files and turning their trees into the desired tree syntax

The VTB original trees are stored in the directory:
VietTreebank_VLSP_SP73/Kho ngu lieu 10000 cay cu phap
The script requires two arguments:
1. Original directory storing the original trees
2. New directory storing the converted trees
"""

import argparse
import os

from collections import Counter

from stanza.models.constituency.tree_reader import read_trees, MixedTreeError

REMAPPING = {
    '(MPD':     '(MDP',
    '(MP ':     '(M ',
    '(MP(':     '(M(',
    '(Np(':     '(NP(',
    '(Np (':    '(NP (',
    '(NLOC':    '(NP-LOC',
    '(N-P-LOC': '(NP-LOC',
    '(NPDOB':   '(NP-DOB',
    '(NPSUB':   '(NP-SUB',
    '(NPTMP':   '(NP-TMP',
    '(PPLOC':   '(PP-LOC',
    '(SBA ':    '(SBAR ',
    '(SBA-':    '(SBAR-',
    '(SBA(':    '(SBAR(',
    '(SBAS':    '(SBAR',
    '(SABR':    '(SBAR',
    '(SE-SPL':  '(S-SPL',
    '(SBARR':   '(SBAR',
    'PPADV':    'PP-ADV',
    '(PR':      '(PP',
    '(PPP':     '(PP',
    'VP0ADV':   'VP-ADV',
    '(S1':      '(S',
    '(S2':      '(S',
    '(S3':      '(S',
    'BP-SUB':   'NP-SUB',
    'APPPD':    'AP-PPD',
    'APPRD':    'AP-PPD',
    'Np--H':    'Np-H',
    '(WHRPP':   '(WHRP',
    # the only PV tags are after S, so this should be the right conversion
    '(PV':     '(S-PV',
}

def unify_label(tree):
    for old, new in REMAPPING.items():
        tree = tree.replace(old, new)

    return tree


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

    The idea is that some "trees" are just a long list of words with
    no tree structure and need to be eliminated.

    :param line: constituent being read
    :return: True if it has open OR closing parenthesis.
    """
    if line.startswith('(') or line.endswith(')'):
        return True

    return False

# not clear if TP is supposed to be NP or PP - needs a native speaker to decode
WEIRD_LABELS = ["WP", "YP", "SNP", "STC", "UPC", "(TP"]

def convert_file(orig_file, new_file):
    """
    :param orig_file: original directory storing original trees
    :param new_file: new directory storing formatted constituency trees
    This function writes new trees to the corresponding files in new_file
    """
    errors = Counter()
    with open(orig_file, 'r') as reader, open(new_file, 'w') as writer:
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
                    errors["empty"] += 1
                    continue
                tree += ')\n'
                if not is_closed_tree(tree):
                    #print("Rejecting the following tree from {} for being unclosed: |{}|".format(orig_file, tree))
                    tree = ""
                    errors["unclosed"] += 1
                    continue
                # TODO: these blocks eliminate 11 trees
                # maybe those trees can be salvaged?
                bad_label = False
                for weird_label in WEIRD_LABELS:
                    if tree.find(weird_label) >= 0:
                        bad_label = True
                        errors[weird_label] += 1
                        break
                if bad_label:
                    continue
                try:
                    # test that the tree can be read in properly
                    read_trees(tree)
                    # Unify the labels
                    tree = unify_label(tree)
                    writer.write(tree)
                    reading_tree = False
                    tree = ""
                except MixedTreeError:
                    #print("Skipping an illegal tree: {}".format(tree))
                    errors["illegal"] += 1
            else:  # content line
                if is_valid_line(line) and reading_tree:
                    tree += line
                elif reading_tree:
                    errors["invalid"] += 1
                    #print("Invalid tree error in {}: |{}|, rejected because of line |{}|".format(orig_file, tree, line))
                    tree = ""
                    reading_tree = False

    return errors


def convert_dir(org_dir, new_dir):
    errors = Counter()
    for filename in os.listdir(org_dir):
        file_name, file_extension = os.path.splitext(filename)
        # Only convert .prd files, skip the .raw files
        if file_extension == '.raw':
            continue
        file_path = os.path.join(org_dir, filename)
        new_path = os.path.join(new_dir, file_name)
        new_file_path = f'{new_path}.mrg'
        # Convert the tree and write to new_file_path
        errors += convert_file(file_path, new_file_path)

    errors = "\n  ".join(sorted(["%s: %s" % x for x in  errors.items()]))
    print("Found the following error counts when processing {}:\n  {}".format(org_dir, errors))

def main():
    """
    Converts files from the 2009 version of VLSP to .mrg files
    
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
