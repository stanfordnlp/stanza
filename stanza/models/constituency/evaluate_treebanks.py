"""
Read multiple treebanks, score the results.

Reports the k-best score if multiple predicted treebanks are given.
"""

import argparse

from stanza.models.constituency import tree_reader
from stanza.server.parser_eval import EvaluateParser, ParseResult


def main():
    parser = argparse.ArgumentParser(description='Get scores for one or more treebanks against the gold')
    parser.add_argument('gold', type=str, help='Which file to load as the gold trees')
    parser.add_argument('pred', type=str, nargs='+', help='Which file(s) are the predictions.  If more than one is given, the evaluation will be "k-best" with the first prediction treated as the canonical')
    args = parser.parse_args()

    print("Loading gold treebank: " + args.gold)
    gold = tree_reader.read_treebank(args.gold)
    print("Loading predicted treebanks: " + args.pred)
    pred = [tree_reader.read_treebank(x) for x in args.pred]

    full_results = [ParseResult(parses[0], [*parses[1:]])
                    for parses in zip(gold, *pred)]

    if len(pred) <= 1:
        kbest = None
    else:
        kbest = len(pred)

    with EvaluateParser(kbest=kbest) as evaluator:
        response = evaluator.process(full_results)

if __name__ == '__main__':
    main()
