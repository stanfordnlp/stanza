"""
Use the concepts in "Dataset Cartography" and "Mind Your Outliers" to find trees with the least variance over a training run

https://arxiv.org/pdf/2009.10795.pdf
https://arxiv.org/abs/2107.02331

The idea here is that high variance trees are more likely to be wrong in the first place.  Using this will filter a silver dataset to have better trees.

for example:

nlprun -d a6000 -p high "export CLASSPATH=/sailhome/horatio/CoreNLP/classes:/sailhome/horatio/CoreNLP/lib/*:$CLASSPATH; python3 stanza/utils/datasets/constituency/silver_variance.py --eval_file /u/nlp/data/constituency-parser/italian/2024_it_vit_electra/it_silver_0.mrg saved_models/constituency/it_vit.top.each.silver0.constituency_0*0.pt --output_file filtered_silver0.mrg" -o filter.out
"""

import argparse

import logging

import numpy

from stanza.models.common import utils
from stanza.models.common.foundation_cache import FoundationCache
from stanza.models.constituency import retagging
from stanza.models.constituency import tree_reader
from stanza.models.constituency.parser_training import run_dev_set
from stanza.models.constituency.trainer import Trainer
from stanza.models.constituency.utils import retag_trees
from stanza.server.parser_eval import EvaluateParser
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

logger = logging.getLogger('stanza.constituency.trainer')

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Script to filter trees by how much variance they show over multiple checkpoints of a parser training run.")

    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--output_file', type=str, default=None, help='Output file after sorting by variance.')

    parser.add_argument('--charlm_forward_file', type=str, default=None, help="Exact path to use for forward charlm")
    parser.add_argument('--charlm_backward_file', type=str, default=None, help="Exact path to use for backward charlm")
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')

    utils.add_device_args(parser)

    # TODO: use the training scripts to pick the charlm & pretrain if needed
    parser.add_argument('--lang', default='it', help='Language to use')

    parser.add_argument('--eval_batch_size', type=int, default=50, help='How many trees to batch when running eval')
    parser.add_argument('models', type=str, nargs='+', default=None, help="Which model(s) to load")

    parser.add_argument('--keep', type=float, default=0.5, help="How many trees to keep after sorting by variance")
    parser.add_argument('--reverse', default=False, action='store_true', help='Actually, keep the high variance trees')

    retagging.add_retag_args(parser)

    args = vars(parser.parse_args())

    retagging.postprocess_args(args)

    return args

def main():
    args = parse_args()
    retag_pipeline = retagging.build_retag_pipeline(args)
    foundation_cache = retag_pipeline[0].foundation_cache if retag_pipeline else FoundationCache()

    print("Analyzing with the following models:\n  " + "\n  ".join(args['models']))

    treebank = tree_reader.read_treebank(args['eval_file'])
    logger.info("Read %d trees for analysis", len(treebank))

    f1_history = []
    retagged_treebank = None

    chunk_size = 5000
    with EvaluateParser() as evaluator:
        for model_filename in args['models']:
            print("Starting processing with %s" % model_filename)
            trainer = Trainer.load(model_filename, args=args, foundation_cache=foundation_cache)
            if retag_pipeline is not None and retagged_treebank is None:
                retag_method = trainer.model.args['retag_method']
                retag_xpos = trainer.model.args['retag_xpos']
                logger.info("Retagging trees using the %s tags from the %s package...", retag_method, args['retag_package'])
                retagged_treebank = retag_trees(treebank, retag_pipeline, retag_xpos)
                logger.info("Retagging finished")

            current_history = []
            for chunk_start in range(0, len(treebank), chunk_size):
                chunk = treebank[chunk_start:chunk_start+chunk_size]
                retagged_chunk = retagged_treebank[chunk_start:chunk_start+chunk_size] if retagged_treebank else None
                f1, kbestF1, treeF1 = run_dev_set(trainer.model, retagged_chunk, chunk, args, evaluator)
                current_history.extend(treeF1)

            f1_history.append(current_history)

    f1_history = numpy.array(f1_history)
    f1_variance = numpy.var(f1_history, axis=0)
    f1_sorted = sorted([(x, idx) for idx, x in enumerate(f1_variance)], reverse=args['reverse'])

    num_keep = int(len(f1_sorted) * args['keep'])
    with open(args['output_file'], "w", encoding="utf-8") as fout:
        for _, idx in f1_sorted[:num_keep]:
            fout.write(str(treebank[idx]))
            fout.write("\n")

if __name__ == "__main__":
    main()
