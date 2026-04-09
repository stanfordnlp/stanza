"""
Given two ensembles and a tokenized file, output the trees for which those ensembles agree and report how many of the sub-models agree on those trees.

For example:

python3 -m stanza.utils.datasets.depparse.build_silver_dataset --tokenized_file /u/nlp/data/constituency-parser/english/en_wiki_2023/shuf_1M.aj.p1 --shorthand en_ewt --output_file en_silver.aj.p1.conllu --e1 saved_models/depparse/en_ewt_bert_graph.100[12345].pt --e2 saved_models/depparse/en_ewt_bert_baseline_v2_100[12345].pt

python3 -m stanza.utils.datasets.depparse.build_silver_dataset --tokenized_file /u/nlp/data/constituency-parser/italian/2024_wiki_tokenization/it_wiki_tok_6M_aa.txt --shorthand it_vit --output_file /u/nlp/data/dependency-parser/silver/it/it_silver.aa.conllu --e1 saved_models/depparse/it_vit_bert_graph.100[12345].pt --e2 saved_models/depparse/it_vit_bert_baseline_v3_100[12345].pt

nlprun -da6000 "python3 -m stanza.utils.datasets.depparse.build_silver_dataset --tokenized_file /u/nlp/data/constituency-parser/italian/2024_wiki_tokenization/it_wiki_tok_6M_aa.txt --shorthand it_vit --output_file /u/nlp/data/dependency-parser/silver/it/it_silver.aa.conllu --e1 saved_models/depparse/it_vit_bert_graph.100[12345].pt --e2 saved_models/depparse/it_vit_bert_baseline_v3_100[12345].pt" -o /u/nlp/data/dependency-parser/silver/it/it_silver.aa.out

for i in `echo aa ab ac ad ae af ag ah ai aj`; do nlprun -da6000 "python3 -m stanza.utils.datasets.depparse.build_silver_dataset --tokenized_file /u/nlp/data/constituency-parser/italian/2024_wiki_tokenization/it_wiki_tok_6M_$i.txt --shorthand it_vit --output_file /u/nlp/data/dependency-parser/silver/it/it_silver.$i.conllu --e1 saved_models/depparse/it_vit_bert_graph.100[12345].pt --e2 saved_models/depparse/it_vit_bert_baseline_v3_100[12345].pt" -o /u/nlp/data/dependency-parser/silver/it/it_silver.$i.out; done

for i in `echo aa ab ac ad ae af ag ah ai aj`; do nlprun -da6000 "python3 -m stanza.utils.datasets.depparse.build_silver_dataset --tokenized_file /u/nlp/data/constituency-parser/italian/2024_wiki_tokenization/it_wiki_tok_6M_$i.txt --shorthand de_gsd --output_file /u/nlp/data/dependency-parser/silver/it/it_silver.$i.conllu --e1 saved_models/depparse/de_gsd_bert_graph.100[12345].pt --e2 saved_models/depparse/de_gsd_bert_baseline_v3_100[12345].pt" -o /u/nlp/data/dependency-parser/silver/it/it_silver.$i.out; done
"""

import argparse
import json

import logging


import stanza
from stanza.models.common import pretrain
from stanza.models.common import utils
from stanza.models.common.doc import HEAD, DEPREL, Document
from stanza.models.common.foundation_cache import FoundationCache
from stanza.models.constituency import retagging
from stanza.models.constituency import text_processing
from stanza.models.constituency import tree_reader
from stanza.models.depparse.data import DataLoader
from stanza.models.depparse.ensemble import build_ensemble
from stanza.models.depparse.model import GraphParser, EnsembleGraphParser
from stanza.models.depparse.transition.model import TransitionParser, EnsembleTransitionParser
from stanza.models.depparse.trainer import Trainer
from stanza.models.depparse.utils import predict_dataset
from stanza.utils.get_tqdm import get_tqdm
# TODO: this could be refactored to a common location, not training-specific
from stanza.utils.training.common import choose_depparse_charlm, find_charlm_file, choose_depparse_pretrain

tqdm = get_tqdm()

logger = logging.getLogger('stanza.depparse')

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Script that uses multiple ensembles to find trees where both ensembles agree")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--tokenized_file', type=str, default=None, help='Input file of tokenized text for parsing with parse_text.')
    input_group.add_argument('--tree_file', type=str, default=None, help='Input file of already parsed text for reparsing with parse_text.')
    parser.add_argument('--output_file', type=str, default=None, help='Where to put the output file')

    parser.add_argument('--charlm_forward_file', type=str, default=None, help="Exact path to use for forward charlm")
    parser.add_argument('--charlm_backward_file', type=str, default=None, help="Exact path to use for backward charlm")
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')

    parser.add_argument('--chunk_size', type=int, default=100, help='How many sentences to process at once')

    utils.add_device_args(parser)

    parser.add_argument('--shorthand', default='en_ewt', help='Language/package to use (relevant for finding charlm & wordvec)')

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

def is_equal_dependencies(s1, s2):
    assert len(s1.words) == len(s2.words)
    for w1, w2 in zip(s1.words, s2.words):
        if w1.head != w2.head or w1.deprel != w2.deprel:
            return False
    return True

def count_matches(new_sentences, doc_dict, pipelines):
    equal_counts = [0 for _ in new_sentences]
    for pipe in pipelines:
        doc = Document(doc_dict)
        doc = pipe(doc)
        for sent_idx, (s1, s2) in enumerate(zip(new_sentences, doc.sentences)):
            if is_equal_dependencies(s1, s2):
                equal_counts[sent_idx] += 1
    return equal_counts

def main():
    args = parse_args()
    utils.log_training_args(args, logger, name="ensemble")

    short_language, dataset = args['shorthand'].split("_")

    pt = None
    if args['wordvec_pretrain_file'] is None:
        args['wordvec_pretrain_file'] = choose_depparse_pretrain(short_language, dataset)
        logger.info("Found wordvec_pretrain_file %s", args['wordvec_pretrain_file'])
    if args['wordvec_pretrain_file']:
        pt = pretrain.Pretrain(args['wordvec_pretrain_file'])
        args['pretrain'] = True

    if args['charlm_forward_file'] is None:
        charlm = choose_depparse_charlm(short_language, dataset, "default")
        args['charlm_forward_file'] = find_charlm_file("forward", short_language, charlm)
        logger.info("Found charlm_forward_file %s", args['charlm_forward_file'])

    if args['charlm_backward_file'] is None:
        charlm = choose_depparse_charlm(short_language, dataset, "default")
        args['charlm_backward_file'] = find_charlm_file("backward", short_language, charlm)
        logger.info("Found charlm_backward_file %s", args['charlm_backward_file'])

    foundation_cache = FoundationCache()

    logger.info("Building ensemble #1 out of %s", args['e1'])
    e1_tr, e1_trainers = build_ensemble(args, pt, args['e1'], foundation_cache=foundation_cache, device=args['device'])
    logger.info("Building ensemble #2 out of %s", args['e2'])
    e2_tr, e2_trainers = build_ensemble(args, pt, args['e2'], foundation_cache=foundation_cache, device=args['device'])

    # TODO: pass pretrain args etc
    e1_pipe = stanza.Pipeline(short_language, processors="tokenize,pos,lemma,depparse", tokenize_pretokenized=True, depparse_trainer=e1_tr, depparse_batch_size=1000, foundation_cache=foundation_cache, device=args['device'])
    e2_pipe = stanza.Pipeline(short_language, processors="tokenize,pos,lemma,depparse", tokenize_pretokenized=True, depparse_trainer=e2_tr, depparse_batch_size=1000, foundation_cache=foundation_cache, device=args['device'])

    logger.info("Loaded ensemble #1: %s", e1_pipe.processors['depparse']._trainer.model_name)
    logger.info("  Data direction reversed: %s", e1_tr.args['reversed'])
    logger.info("Loaded ensemble #2: %s", e2_pipe.processors['depparse']._trainer.model_name)
    logger.info("  Data direction reversed: %s", e2_tr.args['reversed'])

    if args['tokenized_file']:
        tokenized_sentences, _ = text_processing.read_tokenized_file(args['tokenized_file'])
    # TODO: read a conllu file instead of a tokenized_file
    elif args['tree_file']:
        treebank = tree_reader.read_treebank(args['tree_file'])
        tokenized_sentences = [x.leaf_labels() for x in treebank]
        if args['lang'] == 'vi':
            tokenized_sentences = [[x.replace("_", " ") for x in sentence] for sentence in tokenized_sentences]
    logger.info("Read %d tokenized sentences", len(tokenized_sentences))

    e1_pipelines = [stanza.Pipeline(short_language, processors="tokenize,depparse", tokenize_pretokenized=True, depparse_trainer=tr, depparse_batch_size=1000, foundation_cache=foundation_cache, download_method=None, depparse_pretagged=True, device=args['device'])
                    for tr in e1_trainers]
    e2_pipelines = [stanza.Pipeline(short_language, processors="tokenize,depparse", tokenize_pretokenized=True, depparse_trainer=tr, depparse_batch_size=1000, foundation_cache=foundation_cache, download_method=None, depparse_pretagged=True, device=args['device'])
                    for tr in e2_trainers]

    chunk_size = args['chunk_size']
    with open(args['output_file'], 'w', encoding='utf-8') as fout:
        end_tree = len(tokenized_sentences) if args['end_tree'] is None else args['end_tree']
        for chunk_start in range(args['start_tree'], end_tree, chunk_size):
            chunk = tokenized_sentences[chunk_start:chunk_start+chunk_size]
            logger.info("Processing trees %d to %d", chunk_start, chunk_start+len(chunk))

            e1_doc = e1_pipe(chunk)
            e2_doc = e2_pipe(chunk)

            e1_sentences = e1_doc.sentences
            e2_sentences = e2_doc.sentences
            new_sentences = []
            for sent_idx, (e1_s, e2_s) in enumerate(zip(e1_sentences, e2_sentences)):
                if is_equal_dependencies(e1_s, e2_s):
                    e1_s.sent_id = sent_idx + chunk_start
                    new_sentences.append(e1_s)
            logger.debug("Ensembles matched on %d trees", len(new_sentences))
            e1_doc.sentences = new_sentences
            doc_dict = e1_doc.to_dict()

            e1_matches = count_matches(new_sentences, doc_dict, e1_pipelines)
            for sentence, count in zip(new_sentences, e1_matches):
                sentence.add_comment("e1_matches = %d / %d" % (count, len(e1_pipelines)))
            e2_matches = count_matches(new_sentences, doc_dict, e2_pipelines)
            for sentence, count in zip(new_sentences, e2_matches):
                sentence.add_comment("e2_matches = %d / %d" % (count, len(e2_pipelines)))
            for sentence, e1_count, e2_count in zip(new_sentences, e1_matches, e2_matches):
                sentence.add_comment("total_matches = %d / %d" % (e1_count + e2_count, len(e1_pipelines) + len(e2_pipelines)))
            fout.write("{:C}\n\n".format(e1_doc))

if __name__ == '__main__':
    main()
