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
def dispatch_optimizer(name, parameters, opt_logger, lr=None, betas=None, eps=None, momentum=None, **extra_args):
    extra_logging = ""
    if len(extra_args) > 0:
        extra_logging = ", " + ", ".join("%s=%s" % (x, y) for x, y in extra_args.items())

    if name == 'amsgrad':
        opt_logger.debug("Building Adam w/ amsgrad with lr=%f, betas=%s, eps=%f%s", lr, betas, eps, extra_logging)
        return torch.optim.Adam(parameters, amsgrad=True, lr=lr, betas=betas, eps=eps, **extra_args)
    elif name == 'amsgradw':
        opt_logger.debug("Building AdamW w/ amsgrad with lr=%f, betas=%s, eps=%f%s", lr, betas, eps, extra_logging)
        return torch.optim.AdamW(parameters, amsgrad=True, lr=lr, betas=betas, eps=eps, **extra_args)
    elif name == 'sgd':
        opt_logger.debug("Building SGD with lr=%f, momentum=%f%s", lr, momentum, extra_logging)
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum, **extra_args)
    elif name == 'adagrad':
        opt_logger.debug("Building Adagrad with lr=%f%s", lr, extra_logging)
        return torch.optim.Adagrad(parameters, lr=lr, **extra_args)
    elif name == 'adam':
        opt_logger.debug("Building Adam with lr=%f, betas=%s, eps=%f%s", lr, betas, eps, extra_logging)
        return torch.optim.Adam(parameters, lr=lr, betas=betas, eps=eps, **extra_args)
    elif name == 'adamw':
        opt_logger.debug("Building AdamW with lr=%f, betas=%s, eps=%f%s", lr, betas, eps, extra_logging)
        return torch.optim.AdamW(parameters, lr=lr, betas=betas, eps=eps, **extra_args)
    elif name == 'adamax':
        opt_logger.debug("Building Adamax%s", extra_logging)
        return torch.optim.Adamax(parameters, **extra_args) # use default lr
    elif name == 'adadelta':
        opt_logger.debug("Building Adadelta with lr=%f%s", lr, extra_logging)
        return torch.optim.Adadelta(parameters, lr=lr, **extra_args)
    elif name == 'adabelief':
        try:
            from adabelief_pytorch import AdaBelief
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Could not create adabelief optimizer.  Perhaps the adabelief-pytorch package is not installed") from e
        opt_logger.debug("Building AdaBelief with lr=%f, eps=%f%s", lr, eps, extra_logging)
        # TODO: add weight_decouple and rectify as extra args?
        return AdaBelief(parameters, lr=lr, eps=eps, weight_decouple=True, rectify=True, **extra_args)
    elif name == 'madgrad':
        try:
            import madgrad
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Could not create madgrad optimizer.  Perhaps the madgrad package is not installed") from e
        opt_logger.debug("Building MADGRAD with lr=%f, momentum=%f%s", lr, momentum, extra_logging)
        return madgrad.MADGRAD(parameters, lr=lr, momentum=momentum, **extra_args)
    elif name == 'mirror_madgrad':
        try:
            import madgrad
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Could not create mirror_madgrad optimizer.  Perhaps the madgrad package is not installed") from e
        opt_logger.debug("Building MirrorMADGRAD with lr=%f, momentum=%f%s", lr, momentum, extra_logging)
        return madgrad.MirrorMADGRAD(parameters, lr=lr, momentum=momentum, **extra_args)
    else:
        raise ValueError("Unsupported optimizer: {}".format(name))


def get_optimizer(name, model, lr, betas=(0.9, 0.999), eps=1e-8, momentum=0, weight_decay=None, bert_learning_rate=0.0, bert_weight_decay=None, charlm_learning_rate=0.0, is_peft=False, bert_finetune_layers=None, opt_logger=None):
    opt_logger = opt_logger if opt_logger is not None else logger
    base_parameters = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith("bert_model.")
                       and not n.startswith("charmodel_forward.") and not n.startswith("charmodel_backward.")]
    parameters = [{'param_group_name': 'base', 'params': base_parameters}]

    charlm_parameters = [p for n, p in model.named_parameters()
                         if p.requires_grad and (n.startswith("charmodel_forward.") or n.startswith("charmodel_backward."))]
    if len(charlm_parameters) > 0 and charlm_learning_rate > 0:
        parameters.append({'param_group_name': 'charlm', 'params': charlm_parameters, 'lr': lr * charlm_learning_rate})

    if not is_peft:
        bert_parameters = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("bert_model.")]

        # bert_finetune_layers limits the bert finetuning to the *last* N layers of the model
        if len(bert_parameters) > 0 and bert_finetune_layers is not None:
            num_layers = model.bert_model.config.num_hidden_layers
            start_layer = num_layers - bert_finetune_layers
            bert_parameters = []
            for layer_num in range(start_layer, num_layers):
                bert_parameters.extend([param for name, param in model.named_parameters()
                                        if param.requires_grad and name.startswith("bert_model.") and "layer.%d." % layer_num in name])

        if len(bert_parameters) > 0 and bert_learning_rate > 0:
            opt_logger.debug("Finetuning %d bert parameters with LR %s and WD %s", len(bert_parameters), lr * bert_learning_rate, bert_weight_decay)
            parameters.append({'param_group_name': 'bert', 'params': bert_parameters, 'lr': lr * bert_learning_rate})
            if bert_weight_decay is not None:
                parameters[-1]['weight_decay'] = bert_weight_decay
    else:
        # some optimizers seem to train some even with a learning rate of 0...
        if bert_learning_rate > 0:
            # because PEFT handles what to hand to an optimizer, we don't want to touch that
            parameters.append({'param_group_name': 'bert', 'params': model.bert_model.parameters(), 'lr': lr * bert_learning_rate})
            if bert_weight_decay is not None:
                parameters[-1]['weight_decay'] = bert_weight_decay

    extra_args = {}
    if weight_decay is not None:
        extra_args["weight_decay"] = weight_decay

    return dispatch_optimizer(name, parameters, opt_logger=opt_logger, lr=lr, betas=betas, eps=eps, momentum=momentum, **extra_args)

def get_split_optimizer(name, model, lr, betas=(0.9, 0.999), eps=1e-8, momentum=0, weight_decay=None, bert_learning_rate=0.0, bert_weight_decay=None, charlm_learning_rate=0.0, is_peft=False, bert_finetune_layers=None):
    """Same as `get_optimizer`, but splits the optimizer for Bert into a seperate optimizer"""
    base_parameters = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith("bert_model.")
                       and not n.startswith("charmodel_forward.") and not n.startswith("charmodel_backward.")]
    parameters = [{'param_group_name': 'base', 'params': base_parameters}]

    charlm_parameters = [p for n, p in model.named_parameters()
                         if p.requires_grad and (n.startswith("charmodel_forward.") or n.startswith("charmodel_backward."))]
    if len(charlm_parameters) > 0 and charlm_learning_rate > 0:
        parameters.append({'param_group_name': 'charlm', 'params': charlm_parameters, 'lr': lr * charlm_learning_rate})

    bert_parameters = None
    if not is_peft:
        trainable_parameters = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("bert_model.")]

        # bert_finetune_layers limits the bert finetuning to the *last* N layers of the model
        if len(trainable_parameters) > 0 and bert_finetune_layers is not None:
            num_layers = model.bert_model.config.num_hidden_layers
            start_layer = num_layers - bert_finetune_layers
            trainable_parameters = []
            for layer_num in range(start_layer, num_layers):
                trainable_parameters.extend([param for name, param in model.named_parameters()
                                             if param.requires_grad and name.startswith("bert_model.") and "layer.%d." % layer_num in name])

        if len(trainable_parameters) > 0:
            bert_parameters = [{'param_group_name': 'bert', 'params': trainable_parameters, 'lr': lr * bert_learning_rate}]
    else:
        # because PEFT handles what to hand to an optimizer, we don't want to touch that
        bert_parameters = [{'param_group_name': 'bert', 'params': model.bert_model.parameters(), 'lr': lr * bert_learning_rate}]

    extra_args = {}
    if weight_decay is not None:
        extra_args["weight_decay"] = weight_decay

    optimizers = {
        "general_optimizer": dispatch_optimizer(name, parameters, opt_logger=logger, lr=lr, betas=betas, eps=eps, momentum=momentum, **extra_args)
    }
    if bert_parameters is not None and bert_learning_rate > 0.0:
        if bert_weight_decay is not None:
            extra_args['weight_decay'] = bert_weight_decay
        optimizers["bert_optimizer"] = dispatch_optimizer(name, bert_parameters, opt_logger=logger, lr=lr, betas=betas, eps=eps, momentum=momentum, **extra_args)
    return optimizers


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
    # some of these calls are probably redundant
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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

def standard_model_file_name(args, model_type, **kwargs):
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

    use_peft = "nopeft"
    if args.get("bert_finetune", False) and args.get("use_peft", False):
        use_peft = "peft"

    bert_finetuning = ""
    if args.get("bert_finetune", False):
        if args.get("use_peft", False):
            bert_finetuning = "peft"
        else:
            bert_finetuning = "ft"

    seed = args.get('seed', None)
    if seed is None:
        seed = ""
    else:
        seed = str(seed)

    format_args = {
        "batch_size":      args['batch_size'],
        "bert_finetuning": bert_finetuning,
        "embedding":       embedding,
        "finetune":        finetune,
        "peft":            use_peft,
        "seed":            seed,
        "shorthand":       args['shorthand'],
        "transformer_lr":  transformer_lr,
    }
    format_args.update(**kwargs)
    model_file = args['save_name'].format(**format_args)
    model_file = re.sub("_+", "_", model_file)

    model_dir = os.path.split(model_file)[0]

    if not os.path.exists(os.path.join(args['save_dir'], model_file)) and os.path.exists(model_file):
        return model_file
    return os.path.join(args['save_dir'], model_file)

def escape_misc_space(space):
    spaces = []
    for char in space:
        if char == ' ':
            spaces.append('\\s')
        elif char == '\t':
            spaces.append('\\t')
        elif char == '\r':
            spaces.append('\\r')
        elif char == '\n':
            spaces.append('\\n')
        elif char == '|':
            spaces.append('\\p')
        elif char == '\\':
            spaces.append('\\\\')
        elif char == ' ':
            spaces.append('\\u00A0')
        else:
            spaces.append(char)
    escaped_space = "".join(spaces)
    return escaped_space

def unescape_misc_space(misc_space):
    spaces = []
    pos = 0
    while pos < len(misc_space):
        if misc_space[pos:pos+2] == '\\s':
            spaces.append(' ')
            pos += 2
        elif misc_space[pos:pos+2] == '\\t':
            spaces.append('\t')
            pos += 2
        elif misc_space[pos:pos+2] == '\\r':
            spaces.append('\r')
            pos += 2
        elif misc_space[pos:pos+2] == '\\n':
            spaces.append('\n')
            pos += 2
        elif misc_space[pos:pos+2] == '\\p':
            spaces.append('|')
            pos += 2
        elif misc_space[pos:pos+2] == '\\\\':
            spaces.append('\\')
            pos += 2
        elif misc_space[pos:pos+6] == '\\u00A0':
            spaces.append(' ')
            pos += 6
        else:
            spaces.append(misc_space[pos])
            pos += 1
    unescaped_space = "".join(spaces)
    return unescaped_space

def space_before_to_misc(space):
    """
    Convert whitespace to SpacesBefore specifically for the start of a document.

    In general, UD datasets do not have both SpacesAfter on a token and SpacesBefore on the next token.

    The space(s) are only marked on one of the tokens.

    Only at the very beginning of a document is it necessary to mark what spaces occurred before the actual text,
    and the default assumption is that there is no space if there is no SpacesBefore annotation.
    """
    if not space:
        return ""
    escaped_space = escape_misc_space(space)
    return "SpacesBefore=%s" % escaped_space

def space_after_to_misc(space):
    """
    Convert whitespace back to the escaped format - either SpaceAfter=No or SpacesAfter=...
    """
    if not space:
        return "SpaceAfter=No"
    if space == " ":
        return ""
    escaped_space = escape_misc_space(space)
    return "SpacesAfter=%s" % escaped_space

def misc_to_space_before(misc):
    """
    Find any SpacesBefore annotation in the MISC column and turn it into a space value
    """
    if not misc:
        return ""
    pieces = misc.split("|")
    for piece in pieces:
        if not piece.lower().startswith("spacesbefore="):
            continue
        misc_space = piece.split("=", maxsplit=1)[1]
        return unescape_misc_space(misc_space)
    return ""

def misc_to_space_after(misc):
    """
    Convert either SpaceAfter=No or the SpacesAfter annotation

    see https://universaldependencies.org/misc.html#spacesafter

    We compensate for some treebanks using SpaceAfter=\n instead of SpacesAfter=\n
    On the way back, though, those annotations will be turned into SpacesAfter
    """
    if not misc:
        return " "
    pieces = misc.split("|")
    if any(piece.lower() == "spaceafter=no" for piece in pieces):
        return ""
    if "SpaceAfter=Yes" in pieces:
        # as of UD 2.11, the Cantonese treebank had this as a misc feature
        return " "
    if "SpaceAfter=No~" in pieces:
        # as of UD 2.11, a weird typo in the Russian Taiga dataset
        return ""
    for piece in pieces:
        if piece.startswith("SpaceAfter=") or piece.startswith("SpacesAfter="):
            misc_space = piece.split("=", maxsplit=1)[1]
            return unescape_misc_space(misc_space)
    return " "

def log_norms(model):
    lines = ["NORMS FOR MODEL PARAMTERS"]
    pieces = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            pieces.append((name, "%.6g" % torch.norm(param).item(), "%d" % param.numel()))
    name_len = max(len(x[0]) for x in pieces)
    norm_len = max(len(x[1]) for x in pieces)
    line_format = "  %-" + str(name_len) + "s   %" + str(norm_len) + "s     %s"
    for line in pieces:
        lines.append(line_format % line)
    logger.info("\n".join(lines))

def attach_bert_model(model, bert_model, bert_tokenizer, use_peft, force_bert_saved):
    if use_peft:
        # we use a peft-specific pathway for saving peft weights
        model.add_unsaved_module('bert_model', bert_model)
        model.bert_model.train()
    elif force_bert_saved:
        model.bert_model = bert_model
    elif bert_model is not None:
        model.add_unsaved_module('bert_model', bert_model)
        for _, parameter in bert_model.named_parameters():
            parameter.requires_grad = False
    else:
        model.bert_model = None
    model.add_unsaved_module('bert_tokenizer', bert_tokenizer)

def build_save_each_filename(base_filename):
    """
    If the given name doesn't have %d in it, add %4d at the end of the filename

    This way, there's something to count how many models have been saved
    """
    try:
        base_filename % 1
    except TypeError:
        # so models.pt -> models_0001.pt, etc
        pieces = os.path.splitext(model_save_each_file)
        base_filename = pieces[0] + "_%04d" + pieces[1]
    return base_filename
