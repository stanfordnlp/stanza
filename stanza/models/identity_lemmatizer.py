"""
An identity lemmatizer that mimics the behavior of a normal lemmatizer but directly uses word as lemma.
"""

import os
import argparse
import logging
import random

from stanza.models.lemma.data import DataLoader
from stanza.models.lemma import scorer
from stanza.models.common import utils
from stanza.models.common.doc import *
from stanza.utils.conll import CoNLL
from stanza.models import _training_logging

logger = logging.getLogger('stanza')

def parse_args(args=None):
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

    args = parser.parse_args(args=args)
    return args

def main(args=None):
    args = parse_args(args=args)

    random.seed(args.seed)

    args = vars(args)

    logger.info("[Launching identity lemmatizer...]")

    if args['mode'] == 'train':
        logger.info("[No training is required; will only generate evaluation output...]")
    
    document = CoNLL.conll2doc(input_file=args['eval_file'])
    batch = DataLoader(document, args['batch_size'], args, evaluation=True, conll_only=True)
    system_pred_file = args['output_file']
    gold_file = args['gold_file']

    # use identity mapping for prediction
    preds = batch.doc.get([TEXT])

    # write to file and score
    batch.doc.set([LEMMA], preds)
    CoNLL.write_doc2conll(batch.doc, system_pred_file)
    if gold_file is not None:
        _, _, score = scorer.score(system_pred_file, gold_file)

        logger.info("Lemma score:")
        logger.info("{} {:.2f}".format(args['lang'], score*100))

if __name__ == '__main__':
    main()
