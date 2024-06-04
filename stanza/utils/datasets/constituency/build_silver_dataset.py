"""
Given two ensembles and a tokenized file, output the trees for which those ensembles agree and report how many of the sub-models agree on those trees.

For example:

python3 -m stanza.utils.datasets.constituency.build_silver_dataset --tokenized_file /u/nlp/data/constituency-parser/italian/2024_wiki_tokenization/it_wiki_tokenized_AA.txt --lang it --output_file asdf.out --e1 saved_models/constituency/it_vit_electra_100?_top_constituency.pt --e2 saved_models/constituency/it_vit_electra_100?_constituency.pt

for i in `echo f g h i j k l m n o p q r s t`; do nlprun -d a6000 "python3 -m stanza.utils.datasets.constituency.build_silver_dataset --tokenized_file /u/nlp/data/constituency-parser/italian/2024_wiki_tokenization/it_wiki_tok_6M_a$i.txt --lang it --output_file /u/nlp/data/constituency-parser/italian/2024_it_vit_electra/a$i.trees --e1 saved_models/constituency/it_vit_electra_100?_top_constituency.pt --e2 saved_models/constituency/it_vit_electra_100?_constituency.pt" -o /u/nlp/data/constituency-parser/italian/2024_it_vit_electra/a$i.out; done

for i in `echo a b c d`; do nlprun -d a6000 "python3 -m stanza.utils.datasets.constituency.build_silver_dataset --tokenized_file /u/nlp/data/constituency-parser/english/en_wiki_2023/shuf_1M.a$i --lang en --output_file /u/nlp/data/constituency-parser/english/2024_en_ptb3_electra/forward_a$i.trees --e1 saved_models/constituency/en_ptb3_electra-large_100?_in_constituency.pt --e2 saved_models/constituency/en_ptb3_electra-large_100?_top_constituency.pt" -o /u/nlp/data/constituency-parser/english/2024_en_ptb3_electra/forward_a$i.out; done
"""

import argparse
import json

import logging

from stanza.models.common import utils
from stanza.models.common.foundation_cache import FoundationCache
from stanza.models.constituency import retagging
from stanza.models.constituency import text_processing
from stanza.models.constituency import tree_reader
from stanza.models.constituency.ensemble import Ensemble
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

logger = logging.getLogger('stanza.constituency.trainer')

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Script that uses multiple ensembles to find trees where both ensembles agree")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--tokenized_file', type=str, default=None, help='Input file of tokenized text for parsing with parse_text.')
    input_group.add_argument('--tree_file', type=str, default=None, help='Input file of already parsed text for reparsing with parse_text.')
    parser.add_argument('--output_file', type=str, default=None, help='Where to put the output file')

    parser.add_argument('--charlm_forward_file', type=str, default=None, help="Exact path to use for forward charlm")
    parser.add_argument('--charlm_backward_file', type=str, default=None, help="Exact path to use for backward charlm")
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')

    utils.add_device_args(parser)

    parser.add_argument('--lang', default='en', help='Language to use')

    parser.add_argument('--eval_batch_size', type=int, default=50, help='How many trees to batch when running eval')
    parser.add_argument('--e1', type=str, nargs='+', default=None, help="Which model(s) to load in the first ensemble")
    parser.add_argument('--e2', type=str, nargs='+', default=None, help="Which model(s) to load in the second ensemble")

    parser.add_argument('--mode', default='predict', choices=['parse_text', 'predict'])

    # another option would be to include the tree idx in each entry in an existing saved file
    # the processing could then pick up at exactly the last known idx
    parser.add_argument('--start_tree', type=int, default=0, help='Where to start... most useful if the previous incarnation crashed')
    parser.add_argument('--end_tree', type=int, default=None, help='Where to end.  If unset, will process to the end of the file')

    retagging.add_retag_args(parser)

    args = vars(parser.parse_args())

    retagging.postprocess_args(args)
    args['num_generate'] = 0

    return args

def main():
    args = parse_args()
    utils.log_training_args(args, logger, name="ensemble")

    retag_pipeline = retagging.build_retag_pipeline(args)
    foundation_cache = retag_pipeline[0].foundation_cache if retag_pipeline else FoundationCache()

    logger.info("Building ensemble #1 out of %s", args['e1'])
    e1 = Ensemble(args, filenames=args['e1'], foundation_cache=foundation_cache)
    e1.to(args.get('device', None))
    logger.info("Building ensemble #2 out of %s", args['e2'])
    e2 = Ensemble(args, filenames=args['e2'], foundation_cache=foundation_cache)
    e2.to(args.get('device', None))

    if args['tokenized_file']:
        tokenized_sentences = text_processing.read_tokenized_file(args['tokenized_file'])
    elif args['tree_file']:
        treebank = tree_reader.read_treebank(args['tree_file'])
        tokenized_sentences = [x.leaf_labels() for x in treebank]
        if args['lang'] == 'vi':
            tokenized_sentences = [[x.replace("_", " ") for x in sentence] for sentence in tokenized_sentences]
    logger.info("Read %d tokenized sentences", len(tokenized_sentences))

    all_models = e1.models + e2.models

    chunk_size = 1000
    with open(args['output_file'], 'w', encoding='utf-8') as fout:
        end_tree = len(tokenized_sentences) if args['end_tree'] is None else args['end_tree']
        for chunk_start in tqdm(range(args['start_tree'], end_tree, chunk_size)):
            chunk = tokenized_sentences[chunk_start:chunk_start+chunk_size]
            logger.info("Processing trees %d to %d", chunk_start, chunk_start+len(chunk))
            parsed1 = text_processing.parse_tokenized_sentences(args, e1, retag_pipeline, chunk)
            parsed1 = [x.predictions[0].tree for x in parsed1]
            parsed2 = text_processing.parse_tokenized_sentences(args, e2, retag_pipeline, chunk)
            parsed2 = [x.predictions[0].tree for x in parsed2]
            matching = [t for t, t2 in zip(parsed1, parsed2) if t == t2]
            logger.info("%d trees matched", len(matching))
            model_counts = [0] * len(matching)
            for model in all_models:
                model_chunk = model.parse_sentences_no_grad(iter(matching), model.build_batch_from_trees, args['eval_batch_size'], model.predict)
                model_chunk = [x.predictions[0].tree for x in model_chunk]
                for idx, (t1, t2) in enumerate(zip(matching, model_chunk)):
                    if t1 == t2:
                        model_counts[idx] += 1
            for count, tree in zip(model_counts, matching):
                line = {"tree": "%s" % tree, "count": count}
                fout.write(json.dumps(line))
                fout.write("\n")


if __name__ == '__main__':
    main()
