"""
Utility functions.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import json
import logging
import lzma
import os
import random
import re
import sys
import unicodedata
import zipfile

import torch
import numpy as np

from stanza.models.common.constant import lcode2lang
import stanza.models.common.seq2seq_constant as constant
from stanza.resources.default_packages import TRANSFORMER_NICKNAMES
import stanza.utils.conll18_ud_eval as ud_eval
from stanza.utils.conll18_ud_eval import UDError

logger = logging.getLogger('stanza')

# filenames
def get_wordvec_file(wordvec_dir, shorthand, wordvec_type=None):
    """ Lookup the name of the word vectors file, given a directory and the language shorthand.
    """
    lcode, tcode = shorthand.split('_', 1)
    lang = lcode2lang[lcode]
    # locate language folder
    word2vec_dir = os.path.join(wordvec_dir, 'word2vec', lang)
    fasttext_dir = os.path.join(wordvec_dir, 'fasttext', lang)
    lang_dir = None
    if wordvec_type is not None:
        lang_dir = os.path.join(wordvec_dir, wordvec_type, lang)
        if not os.path.exists(lang_dir):
            raise FileNotFoundError("Word vector type {} was specified, but directory {} does not exist".format(wordvec_type, lang_dir))
    elif os.path.exists(word2vec_dir): # first try word2vec
        lang_dir = word2vec_dir
    elif os.path.exists(fasttext_dir): # otherwise try fasttext
        lang_dir = fasttext_dir
    else:
        raise FileNotFoundError("Cannot locate word vector directory for language: {}  Looked in {} and {}".format(lang, word2vec_dir, fasttext_dir))
    # look for wordvec filename in {lang_dir}
    filename = os.path.join(lang_dir, '{}.vectors'.format(lcode))
    if os.path.exists(filename + ".xz"):
        filename = filename + ".xz"
    elif os.path.exists(filename + ".txt"):
        filename = filename + ".txt"
    return filename

@contextmanager
def output_stream(filename=None):
    """
    Yields the given file if a file is given, or returns sys.stdout if filename is None

    Opens the file in a context manager so it closes nicely
    """
    if filename is None:
        yield sys.stdout
    else:
        with open(filename, "w", encoding="utf-8") as fout:
            yield fout


@contextmanager
def open_read_text(filename, encoding="utf-8"):
    """
    Opens a file as an .xz file or .gz if it ends with .xz or .gz, or regular text otherwise.

    Use as a context

    eg:
    with open_read_text(filename) as fin:
        do stuff

    File will be closed once the context exits
    """
    if filename.endswith(".xz"):
        with lzma.open(filename, mode='rt', encoding=encoding) as fin:
            yield fin
    elif filename.endswith(".gz"):
        with gzip.open(filename, mode='rt', encoding=encoding) as fin:
            yield fin
    else:
        with open(filename, encoding=encoding) as fin:
            yield fin

@contextmanager
def open_read_binary(filename):
    """
    Opens a file as an .xz file or .gz if it ends with .xz or .gz, or regular binary file otherwise.

    If a .zip file is given, it can be read if there is a single file in there

    Use as a context

    eg:
    with open_read_binary(filename) as fin:
        do stuff

    File will be closed once the context exits
    """
    if filename.endswith(".xz"):
        with lzma.open(filename, mode='rb') as fin:
            yield fin
    elif filename.endswith(".gz"):
        with gzip.open(filename, mode='rb') as fin:
            yield fin
    elif filename.endswith(".zip"):
        with zipfile.ZipFile(filename) as zin:
            input_names = zin.namelist()
            if len(input_names) == 0:
                raise ValueError("Empty zip archive")
            if len(input_names) > 1:
                raise ValueError("zip file %s has more than one file in it")
            with zin.open(input_names[0]) as fin:
                yield fin
    else:
        with open(filename, mode='rb') as fin:
            yield fin

# training schedule
def get_adaptive_eval_interval(cur_dev_size, thres_dev_size, base_interval):
    """ Adjust the evaluation interval adaptively.
    If cur_dev_size <= thres_dev_size, return base_interval;
    else, linearly increase the interval (round to integer times of base interval).
    """
    if cur_dev_size <= thres_dev_size:
        return base_interval
    else:
        alpha = round(cur_dev_size / thres_dev_size)
        return base_interval * alpha

# ud utils
def ud_scores(gold_conllu_file, system_conllu_file):
    try:
        gold_ud = ud_eval.load_conllu_file(gold_conllu_file)
    except UDError as e:
        raise UDError("Could not read %s" % gold_conllu_file) from e

    try:
        system_ud = ud_eval.load_conllu_file(system_conllu_file)
    except UDError as e:
        raise UDError("Could not read %s" % system_conllu_file) from e
    evaluation = ud_eval.evaluate(gold_ud, system_ud)

    return evaluation

def harmonic_mean(a, weights=None):
    if any([x == 0 for x in a]):
        return 0
    else:
        assert weights is None or len(weights) == len(a), 'Weights has length {} which is different from that of the array ({}).'.format(len(weights), len(a))
        if weights is None:
            return len(a) / sum([1/x for x in a])
        else:
            return sum(weights) / sum(w/x for x, w in zip(a, weights))

# torch utils
def get_optimizer(name, model, lr, betas=(0.9, 0.999), eps=1e-8, momentum=0, weight_decay=None, bert_learning_rate=0.0, charlm_learning_rate=0.0):
    base_parameters = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith("bert_model.")
                       and not n.startswith("charmodel_forward.") and not n.startswith("charmodel_backward.")]
    parameters = [{'param_group_name': 'base', 'params': base_parameters}]

    charlm_parameters = [p for n, p in model.named_parameters()
                         if p.requires_grad and (n.startswith("charmodel_forward.") or n.startswith("charmodel_backward."))]
    if len(charlm_parameters) > 0 and charlm_learning_rate > 0:
        parameters.append({'param_group_name': 'charlm', 'params': charlm_parameters, 'lr': lr * charlm_learning_rate})

    bert_parameters = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("bert_model.")]
    if len(bert_parameters) > 0 and bert_learning_rate > 0:
        parameters.append({'param_group_name': 'bert', 'params': bert_parameters, 'lr': lr * bert_learning_rate})

    extra_args = {}
    if weight_decay is not None:
        extra_args["weight_decay"] = weight_decay
    if name == 'amsgrad':
        return torch.optim.Adam(parameters, amsgrad=True, lr=lr, betas=betas, eps=eps, **extra_args)
    elif name == 'amsgradw':
        return torch.optim.AdamW(parameters, amsgrad=True, lr=lr, betas=betas, eps=eps, **extra_args)
    elif name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum, **extra_args)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, **extra_args)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, betas=betas, eps=eps, **extra_args)
    elif name == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr, betas=betas, eps=eps, **extra_args)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, **extra_args) # use default lr
    elif name == 'adadelta':
        return torch.optim.Adadelta(parameters, lr=lr, **extra_args)
    elif name == 'madgrad':
        try:
            import madgrad
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Could not create madgrad optimizer.  Perhaps the madgrad package is not installed") from e
        return madgrad.MADGRAD(parameters, lr=lr, momentum=momentum, **extra_args)
    else:
        raise ValueError("Unsupported optimizer: {}".format(name))

def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def flatten_indices(seq_lens, width):
    flat = []
    for i, l in enumerate(seq_lens):
        for j in range(l):
            flat.append(i * width + j)
    return flat

def keep_partial_grad(grad, topk):
    """
    Keep only the topk rows of grads.
    """
    assert topk < grad.size(0)
    grad.data[topk:].zero_()
    return grad

# other utils
def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            logger.info("Directory {} does not exist; creating...".format(d))
        # exist_ok: guard against race conditions
        os.makedirs(d, exist_ok=True)

def save_config(config, path, verbose=True):
    with open(path, 'w') as outfile:
        json.dump(config, outfile, indent=2)
    if verbose:
        print("Config saved to file {}".format(path))
    return config

def load_config(path, verbose=True):
    with open(path) as f:
        config = json.load(f)
    if verbose:
        print("Config loaded from file {}".format(path))
    return config

def print_config(config):
    info = "Running with the following configs:\n"
    for k,v in config.items():
        info += "\t{} : {}\n".format(k, str(v))
    logger.info("\n" + info + "\n")

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def unmap_with_copy(indices, src_tokens, vocab):
    """
    Unmap a list of list of indices, by optionally copying from src_tokens.
    """
    result = []
    for ind, tokens in zip(indices, src_tokens):
        words = []
        for idx in ind:
            if idx >= 0:
                words.append(vocab.id2word[idx])
            else:
                idx = -idx - 1 # flip and minus 1
                words.append(tokens[idx])
        result += [words]
    return result

def prune_decoded_seqs(seqs):
    """
    Prune decoded sequences after EOS token.
    """
    out = []
    for s in seqs:
        if constant.EOS in s:
            idx = s.index(constant.EOS_TOKEN)
            out += [s[:idx]]
        else:
            out += [s]
    return out

def prune_hyp(hyp):
    """
    Prune a decoded hypothesis
    """
    if constant.EOS_ID in hyp:
        idx = hyp.index(constant.EOS_ID)
        return hyp[:idx]
    else:
        return hyp

def prune(data_list, lens):
    assert len(data_list) == len(lens)
    nl = []
    for d, l in zip(data_list, lens):
        nl.append(d[:l])
    return nl

def sort(packed, ref, reverse=True):
    """
    Sort a series of packed list, according to a ref list.
    Also return the original index before the sort.
    """
    assert (isinstance(packed, tuple) or isinstance(packed, list)) and isinstance(ref, list)
    packed = [ref] + [range(len(ref))] + list(packed)
    sorted_packed = [list(t) for t in zip(*sorted(zip(*packed), reverse=reverse))]
    return tuple(sorted_packed[1:])

def unsort(sorted_list, oidx):
    """
    Unsort a sorted list, based on the original idx.
    """
    assert len(sorted_list) == len(oidx), "Number of list elements must match with original indices."
    if len(sorted_list) == 0:
        return []
    _, unsorted = [list(t) for t in zip(*sorted(zip(oidx, sorted_list)))]
    return unsorted

def sort_with_indices(data, key=None, reverse=False):
    """
    Sort data and return both the data and the original indices.

    One useful application is to sort by length, which can be done with key=len
    Returns the data as a sorted list, then the indices of the original list.
    """
    if not data:
        return [], []
    if key:
        ordered = sorted(enumerate(data), key=lambda x: key(x[1]), reverse=reverse)
    else:
        ordered = sorted(enumerate(data), key=lambda x: x[1], reverse=reverse)

    result = tuple(zip(*ordered))
    return result[1], result[0]

def split_into_batches(data, batch_size):
    """
    Returns a list of intervals so that each interval is either <= batch_size or one element long.

    Long elements are not dropped from the intervals.
    data is a list of lists
    batch_size is how long to make each batch
    return value is a list of pairs, start_idx end_idx
    """
    intervals = []
    interval_start = 0
    interval_size = 0
    for idx, line in enumerate(data):
        if len(line) > batch_size:
            # guess we'll just hope the model can handle a batch of this size after all
            if interval_size > 0:
                intervals.append((interval_start, idx))
            intervals.append((idx, idx+1))
            interval_start = idx+1
            interval_size = 0
        elif len(line) + interval_size > batch_size:
            # this line puts us over batch_size
            intervals.append((interval_start, idx))
            interval_start = idx
            interval_size = len(line)
        else:
            interval_size = interval_size + len(line)
    if interval_size > 0:
        # there's some leftover
        intervals.append((interval_start, len(data)))
    return intervals

def tensor_unsort(sorted_tensor, oidx):
    """
    Unsort a sorted tensor on its 0-th dimension, based on the original idx.
    """
    assert sorted_tensor.size(0) == len(oidx), "Number of list elements must match with original indices."
    backidx = [x[0] for x in sorted(enumerate(oidx), key=lambda x: x[1])]
    return sorted_tensor[backidx]


def set_random_seed(seed):
    """
    Set a random seed on all of the things which might need it.
    torch, np, python random, and torch.cuda
    """
    if seed is None:
        seed = random.randint(0, 1000000000)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed

def find_missing_tags(known_tags, test_tags):
    if isinstance(known_tags, list) and isinstance(known_tags[0], list):
        known_tags = set(x for y in known_tags for x in y)
    if isinstance(test_tags, list) and isinstance(test_tags[0], list):
        test_tags = sorted(set(x for y in test_tags for x in y))
    missing_tags = sorted(x for x in test_tags if x not in known_tags)
    return missing_tags

def warn_missing_tags(known_tags, test_tags, test_set_name):
    """
    Print a warning if any tags present in the second list are not in the first list.

    Can also handle a list of lists.
    """
    missing_tags = find_missing_tags(known_tags, test_tags)
    if len(missing_tags) > 0:
        logger.warning("Found tags in {} missing from the expected tag set: {}".format(test_set_name, missing_tags))
        return True
    return False

def checkpoint_name(save_dir, save_name, checkpoint_name):
    """
    Will return a recommended checkpoint name for the given dir, save_name, optional checkpoint_name

    For example, can pass in args['save_dir'], args['save_name'], args['checkpoint_save_name']
    """
    if checkpoint_name:
        model_dir = os.path.split(checkpoint_name)[0]
        if model_dir == save_dir:
            return checkpoint_name
        return os.path.join(save_dir, checkpoint_name)

    model_dir = os.path.split(save_name)[0]
    if model_dir != save_dir:
        save_name = os.path.join(save_dir, save_name)
    if save_name.endswith(".pt"):
        return save_name[:-3] + "_checkpoint.pt"

    return save_name + "_checkpoint"

def default_device():
    """
    Pick a default device based on what's available on this system
    """
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def add_device_args(parser):
    """
    Add args which specify cpu, cuda, or arbitrary device
    """
    parser.add_argument('--device', type=str, default=default_device(), help='Which device to run on - use a torch device string name')
    parser.add_argument('--cuda', dest='device', action='store_const', const='cuda', help='Run on CUDA')
    parser.add_argument('--cpu', dest='device', action='store_const', const='cpu', help='Ignore CUDA and run on CPU')

def load_elmo(elmo_model):
    # This import is here so that Elmo integration can be treated
    # as an optional feature
    import elmoformanylangs

    logger.info("Loading elmo: %s" % elmo_model)
    elmo_model = elmoformanylangs.Embedder(elmo_model)
    return elmo_model

def log_training_args(args, args_logger, name="training"):
    """
    For record keeping purposes, log the arguments when training
    """
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    keys = sorted(args.keys())
    log_lines = ['%s: %s' % (k, args[k]) for k in keys]
    args_logger.info('ARGS USED AT %s TIME:\n%s\n', name.upper(), '\n'.join(log_lines))

def embedding_name(args):
    """
    Return the generic name of the biggest embedding used by a model.

    Used by POS and depparse, for example.

    TODO: Probably will make the transformer names a bit more informative,
    such as electra, roberta, etc.  Maybe even phobert for VI, for example
    """
    embedding = "nocharlm"
    if args['wordvec_pretrain_file'] is None and args['wordvec_file'] is None:
        embedding = "nopretrain"
    if args.get('charlm', True) and (args['charlm_forward_file'] or args['charlm_backward_file']):
        embedding = "charlm"
    if args['bert_model']:
        if args['bert_model'] in TRANSFORMER_NICKNAMES:
            embedding = TRANSFORMER_NICKNAMES[args['bert_model']]
        else:
            embedding = "transformer"

    return embedding

def standard_model_file_name(args, model_type):
    """
    Returns a model file name based on some common args found in the various models.

    The expectation is that the args will have something like

      parser.add_argument('--save_name', type=str, default="{shorthand}_{embedding}_parser.pt", help="File name to save the model")

    Then the model shorthand, embedding type, and other args will be
    turned into arguments in a format string
    """
    embedding = embedding_name(args)

    finetune = ""
    transformer_lr = ""
    if args.get("bert_finetune", False):
        finetune = "finetuned"
        if "bert_learning_rate" in args:
            transformer_lr = "{}".format(args["bert_learning_rate"])

    model_file = args['save_name'].format(shorthand=args['shorthand'],
                                          embedding=embedding,
                                          finetune=finetune,
                                          transformer_lr=transformer_lr)
    model_file = re.sub("_+", "_", model_file)

    model_dir = os.path.split(model_file)[0]

    if not os.path.exists(os.path.join(args['save_dir'], model_file)) and os.path.exists(model_file):
        return model_file
    return os.path.join(args['save_dir'], model_file)
