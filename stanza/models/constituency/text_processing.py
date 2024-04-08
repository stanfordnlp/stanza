import os

import logging

from stanza.models.common import utils
from stanza.models.constituency.utils import retag_tags
from stanza.models.constituency.trainer import Trainer
from stanza.utils.get_tqdm import get_tqdm

logger = logging.getLogger('stanza')
tqdm = get_tqdm()

def read_tokenized_file(tokenized_file):
    """
    Read sentences from a tokenized file, potentially replacing _ with space for languages such as VI
    """
    with open(tokenized_file, encoding='utf-8') as fin:
        lines = fin.readlines()
    lines = [x.strip() for x in lines]
    lines = [x for x in lines if x]
    docs = [[word if all(x == '_' for x in word) else word.replace("_", " ") for word in sentence.split()] for sentence in lines]
    return docs


def parse_tokenized_sentences(args, model, retag_pipeline, sentences):
    """
    Parse the given sentences, return a list of ParseResult objects
    """
    tags = retag_tags(sentences, retag_pipeline, model.uses_xpos())
    words = [[(word, tag) for word, tag in zip(s_words, s_tags)] for s_words, s_tags in zip(sentences, tags)]
    logger.info("Retagging finished.  Parsing tagged text")

    assert len(words) == len(sentences)
    treebank = model.parse_sentences_no_grad(iter(tqdm(words)), model.build_batch_from_tagged_words, args['eval_batch_size'], model.predict, keep_scores=False)
    return treebank

def parse_text(args, model, retag_pipeline, tokenized_file=None, predict_file=None):
    """
    Use the given model to parse text and write it

    refactored so it can be used elsewhere, such as Ensemble
    """
    model.eval()

    if predict_file is None:
        if args['predict_file']:
            predict_file = args['predict_file']
            if args['predict_dir']:
                predict_file = os.path.join(args['predict_dir'], predict_file)

    if tokenized_file is None:
        tokenized_file = args['tokenized_file']

    if tokenized_file is not None:
        docs = read_tokenized_file(tokenized_file)
        logger.info("Processing %d lines", len(docs))

        with utils.output_stream(predict_file) as fout:
            chunk_size = 10000
            for chunk_start in range(0, len(docs), chunk_size):
                chunk = docs[chunk_start:chunk_start+chunk_size]
                logger.info("Processing trees %d to %d", chunk_start, chunk_start+len(chunk))
                treebank = parse_tokenized_sentences(args, model, retag_pipeline, chunk)

                for tree_idx, result in enumerate(treebank):
                    tree = result.predictions[0].tree
                    tree.tree_id = chunk_start + tree_idx + 1
                    fout.write(args['predict_format'].format(tree))
                    fout.write("\n")

def parse_dir(args, model, retag_pipeline, tokenized_dir, predict_dir):
    os.makedirs(predict_dir, exist_ok=True)
    for filename in os.listdir(tokenized_dir):
        input_path = os.path.join(tokenized_dir, filename)
        output_path = os.path.join(predict_dir, os.path.splitext(filename)[0] + ".mrg")
        logger.info("Processing %s to %s", input_path, output_path)
        parse_text(args, model, retag_pipeline, tokenized_file=input_path, predict_file=output_path)


def load_model_parse_text(args, model_file, retag_pipeline):
    """
    Load a model, then parse text and write it to stdout or args['predict_file']

    retag_pipeline: a list of Pipeline meant to use for retagging
    """
    foundation_cache = retag_pipeline[0].foundation_cache if retag_pipeline else FoundationCache()
    load_args = {
        'wordvec_pretrain_file': args['wordvec_pretrain_file'],
        'charlm_forward_file': args['charlm_forward_file'],
        'charlm_backward_file': args['charlm_backward_file'],
        'device': args['device'],
    }
    trainer = Trainer.load(model_file, args=load_args, foundation_cache=foundation_cache)
    model = trainer.model
    model.eval()
    logger.info("Loaded model from %s", model_file)

    if args['tokenized_dir']:
        if not args['predict_dir']:
            raise ValueError("Must specific --predict_dir to go with --tokenized_dir")
        parse_dir(args, model, retag_pipeline, args['tokenized_dir'], args['predict_dir'])
    else:
        parse_text(args, model, retag_pipeline)

