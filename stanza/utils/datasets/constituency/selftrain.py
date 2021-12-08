"""
Common methods for the various self-training data collection scripts
"""

import logging
import os
import random

import stanza
from stanza.models.common import utils
from stanza.models.constituency.utils import TextTooLongError

logger = logging.getLogger('stanza')
tqdm = utils.get_tqdm()

def common_args(parser):
    parser.add_argument(
        '--output_file',
        default='data/constituency/vi_silver.mrg',
        help='Where to write the silver trees'
    )
    parser.add_argument(
        '--lang',
        default='vi',
        help='Which language tools to use for tokenization and POS'
    )
    parser.add_argument(
        '--num_sentences',
        type=int,
        default=-1,
        help='How many sentences to get per file (max)'
    )
    parser.add_argument(
        '--models',
        default='saved_models/constituency/vi_vlsp21_inorder.pt',
        help='What models to use for parsing.  comma-separated'
    )

def build_ssplit_pipe(ssplit, lang):
    if ssplit:
        return stanza.Pipeline(lang, processors="tokenize")
    else:
        return stanza.Pipeline(lang, processors="tokenize", tokenize_no_ssplit=True)

def build_tag_pipe(ssplit, lang):
    if ssplit:
        return stanza.Pipeline(lang, processors="tokenize,pos")
    else:
        return stanza.Pipeline(lang, processors="tokenize,pos", tokenize_no_ssplit=True)

def build_parser_pipes(lang, models):
    """
    Build separate pipelines for each parser model we want to use
    """
    parser_pipes = []
    for model_name in models.split(","):
        if os.path.exists(model_name):
            # if the model name exists as a file, treat it as the path to the model
            pipe = stanza.Pipeline(lang, processors="constituency", constituency_model_path=model_name, constituency_pretagged=True)
        else:
            # otherwise, assume it is a package name?
            pipe = stanza.Pipeline(lang, processors={"constituency": model_name}, constituency_pretagged=True, package=None)
        parser_pipes.append(pipe)
    return parser_pipes

def split_docs(docs, ssplit_pipe, max_len=140, max_word_len=100, chunk_size=2000):
    """
    Using the ssplit pipeline, break up the documents into sentences

    Filters out sentences which are too long or have words too long.

    This step is necessary because some web text has unstructured
    sentences which overwhelm the tagger, or even text with no
    whitespace which breaks the charlm in the tokenizer or tagger
    """
    raw_sentences = 0
    filtered_sentences = 0
    new_docs = []

    logger.info("Splitting raw docs into sentences: %d", len(docs))
    for chunk_start in tqdm(range(0, len(docs), chunk_size)):
        chunk = docs[chunk_start:chunk_start+chunk_size]
        chunk = [stanza.Document([], text=t) for t in chunk]
        chunk = ssplit_pipe(chunk)
        sentences = [s for d in chunk for s in d.sentences]
        raw_sentences += len(sentences)
        sentences = [s for s in sentences if len(s.words) < max_len]
        sentences = [s for s in sentences if max(len(w.text) for w in s.words) < max_word_len]
        filtered_sentences += len(sentences)
        new_docs.extend([s.text for s in sentences])

    logger.info("Split sentences: %d", raw_sentences)
    logger.info("Sentences filtered for length: %d", filtered_sentences)
    return new_docs

def find_matching_trees(docs, num_sentences, accepted_trees, tag_pipe, parser_pipes, shuffle=True, chunk_size=10, max_len=140):
    """
    Find trees where all the parsers in parser_pipes agree

    docs should be a list of strings.
      one sentence per string or a whole block of text as long as the tag_pipe can break it into sentences

    num_sentences > 0 gives an upper limit on how many sentences to extract.
      If < 0, all possible sentences are extracted
    """
    if num_sentences < 0:
        tqdm_total = len(docs)
    else:
        tqdm_total = num_sentences

    with tqdm(total=tqdm_total, leave=False) as pbar:
        if shuffle:
            random.shuffle(docs)
        new_trees = set()
        for chunk_start in range(0, len(docs), chunk_size):
            chunk = docs[chunk_start:chunk_start+chunk_size]
            chunk = [stanza.Document([], text=t) for t in chunk]
            tag_pipe(chunk)

            chunk = [d for d in chunk if len(d.sentences) > 0]
            if max_len is not None:
                # for now, we don't have a good way to deal with sentences longer than the bert maxlen
                chunk = [d for d in chunk if max(len(s.words) for s in d.sentences) < max_len]
            if len(chunk) == 0:
                continue

            parses = []
            try:
                for pipe in parser_pipes:
                    pipe(chunk)
                    trees = ["{:L}".format(sent.constituency) for doc in chunk for sent in doc.sentences]
                    parses.append(trees)
            except TextTooLongError as e:
                # easiest is to skip this chunk - could theoretically save the other sentences
                continue

            for tree in zip(*parses):
                if num_sentences < 0:
                    pbar.update(1)
                if len(set(tree)) != 1:
                    continue
                tree = tree[0]
                if tree in accepted_trees:
                    continue
                if tree not in new_trees:
                    new_trees.add(tree)
                    if num_sentences >= 0:
                        pbar.update(1)
                if num_sentences >= 0 and len(new_trees) >= num_sentences:
                    return new_trees

    return new_trees

