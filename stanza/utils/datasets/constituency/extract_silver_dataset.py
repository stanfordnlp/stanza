"""
After running build_silver_dataset.py, this extracts the trees of a certain match level

For example

python3 stanza/utils/datasets/constituency/extract_silver_dataset.py --parsed_trees /u/nlp/data/constituency-parser/italian/2024_it_vit_electra/a*.trees --keep_score 0 --output_file /u/nlp/data/constituency-parser/italian/2024_it_vit_electra/it_silver_0.mrg

for i in `echo 0 1 2 3 4 5 6 7 8 9 10`; do python3 stanza/utils/datasets/constituency/extract_silver_dataset.py --parsed_trees /u/nlp/data/constituency-parser/italian/2024_it_vit_electra/a*.trees --keep_score $i --output_file /u/nlp/data/constituency-parser/italian/2024_it_vit_electra/it_silver_$i.mrg; done
"""

import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="After finding common trees using build_silver_dataset, this extracts them all or just the ones from a particular level of accuracy")
    parser.add_argument('--parsed_trees', type=str, nargs='+', help='Input file(s) of trees parsed into the build_silver_dataset json format.')
    parser.add_argument('--keep_score', type=int, default=None, help='Which agreement level to keep.  None keeps all') 
    parser.add_argument('--output_file', type=str, default=None, help='Where to put the output file')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    trees = []
    for filename in args.parsed_trees:
        with open(filename, encoding='utf-8') as fin:
            for line in fin.readlines():
                tree = json.loads(line)
                if args.keep_score is None or tree['count'] == args.keep_score:
                    tree = tree['tree']
                    trees.append(tree)

    if args.output_file is None:
        for tree in trees:
            print(tree)
    else:
        with open(args.output_file, 'w', encoding='utf-8') as fout:
            for tree in trees:
                fout.write(tree)
                fout.write('\n')

if __name__ == '__main__':
    main()

