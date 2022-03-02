"""
Utilities for the processing of constituency treebanks
"""

def split_treebank(treebank, train_size, dev_size):
    """
    Split a treebank deterministically
    """
    train_end = int(len(treebank) * train_size)
    dev_end = int(len(treebank) * (train_size + dev_size))
    return treebank[:train_end], treebank[train_end:dev_end], treebank[dev_end:]
