"""
An indentity lemmatizer that mimics the behavior of a normal lemmatizer but directly uses word as lemma.
"""

import os
import argparse
import random

from stanfordnlp.models.lemma.data import DataLoader
from stanfordnlp.models.lemma import scorer
from stanfordnlp.models.common import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/lemma', help='Directory for all lemma data.')
    parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--output_file', type=str, default=None, help='Output CoNLL-U file.')
    parser.add_argument('--gold_file', type=str, default=None, help='Output CoNLL-U file.')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')

    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    random.seed(args.seed)

    args = vars(args)

    print("[Launching identity lemmatizer...]")

    if args['mode'] == 'train':
        print("[No training is required; will only generate evaluation output...]")

    batch = DataLoader(args['eval_file'], args['batch_size'], args, evaluation=True, conll_only=True)
    system_pred_file = args['output_file']
    gold_file = args['gold_file']

    # use identity mapping for prediction
    preds = batch.conll.get(['word'])

    # write to file and score
    batch.conll.write_conll_with_lemmas(preds, system_pred_file)
    if gold_file is not None:
        _, _, score = scorer.score(system_pred_file, gold_file)

        print("Lemma score:")
        print("{} {:.2f}".format(args['lang'], score*100))

if __name__ == '__main__':
    main()
