"""
Utility functions.
"""
import os
from collections import Counter
import random
import json
import unicodedata
import torch

from stanfordnlp.models.common.constant import lcode2lang
import stanfordnlp.models.common.seq2seq_constant as constant
import stanfordnlp.utils.conll18_ud_eval as ud_eval

# filenames
def get_wordvec_file(wordvec_dir, shorthand):
    """ Lookup the name of the word vectors file, given a directory and the language shorthand.
    """
    lcode, tcode = shorthand.split('_')
    lang = lcode2lang[lcode] if lcode != 'no' else lcode2lang[shorthand]
    if lcode == 'zh':
        lang = 'ChineseT'
    return os.path.join(wordvec_dir, lang, '{}.vectors.xz'.format(\
            lcode if lcode != 'no' else (shorthand if shorthand != 'no_nynorsklia' else 'no_nynorsk')))

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
    gold_ud = ud_eval.load_conllu_file(gold_conllu_file)
    system_ud = ud_eval.load_conllu_file(system_conllu_file)
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
def get_optimizer(name, parameters, lr, betas=(0.9, 0.999), eps=1e-8):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, betas=betas, eps=eps)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters) # use default lr
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def flatten_indices(seq_lens, width):
    flat = []
    for i, l in enumerate(seq_lens):
        for j in range(l):
            flat.append(i * width + j)
    return flat

def set_cuda(var, cuda):
    if cuda:
        return var.cuda()
    return var

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
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)

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
    print("\n" + info + "\n")
    return

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
    _, unsorted = [list(t) for t in zip(*sorted(zip(oidx, sorted_list)))]
    return unsorted

def tensor_unsort(sorted_tensor, oidx):
    """
    Unsort a sorted tensor on its 0-th dimension, based on the original idx.
    """
    assert sorted_tensor.size(0) == len(oidx), "Number of list elements must match with original indices."
    backidx = [x[0] for x in sorted(enumerate(oidx), key=lambda x: x[1])]
    return sorted_tensor[backidx]
