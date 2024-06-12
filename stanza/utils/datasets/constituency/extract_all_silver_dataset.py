"""
After running build_silver_dataset.py, this extracts the trees of all match levels at once

For example

python stanza/utils/datasets/constituency/extract_all_silver_dataset.py --output_prefix /u/nlp/data/constituency-parser/chinese/2024_zh_wiki/zh_silver_ --parsed_trees /u/nlp/data/constituency-parser/chinese/2024_zh_wiki/zh_wiki_a*trees

cat /u/nlp/data/constituency-parser/chinese/2024_zh_wiki/zh_silver_[012345678].mrg | sort | uniq | shuf > /u/nlp/data/constituency-parser/chinese/2024_zh_wiki/zh_silver_sort.mrg

shuf /u/nlp/data/constituency-parser/chinese/2024_zh_wiki/zh_silver_sort.mrg | head -n 200000 > /u/nlp/data/constituency-parser/chinese/2024_zh_wiki/zh_silver_200K.mrg
"""

import argparse
from collections import defaultdict
import json

def parse_args():
    parser = argparse.ArgumentParser(description="After finding common trees using build_silver_dataset, this extracts them all or just the ones from a particular level of accuracy")
    parser.add_argument('--parsed_trees', type=str, nargs='+', help='Input file(s) of trees parsed into the build_silver_dataset json format.')
    parser.add_argument('--output_prefix', type=str, default=None, help='Prefix to use for outputting trees')
    parser.add_argument('--output_suffix', type=str, default=".mrg", help='Suffix to use for outputting trees')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    trees = defaultdict(list)
    for filename in args.parsed_trees:
        with open(filename, encoding='utf-8') as fin:
            for line in fin.readlines():
                tree = json.loads(line)
                trees[tree['count']].append(tree['tree'])

    for score, tree_list in trees.items():
        filename = "%s%s%s" % (args.output_prefix, score, args.output_suffix)
        with open(filename, 'w', encoding='utf-8') as fout:
            for tree in tree_list:
                fout.write(tree)
                fout.write('\n')

if __name__ == '__main__':
    main()


