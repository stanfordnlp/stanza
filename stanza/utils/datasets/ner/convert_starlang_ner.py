"""
Convert the starlang trees to a NER dataset

Has to hide quite a few trees with missing NER labels
"""

import re

from stanza.models.constituency import tree_reader
import stanza.utils.datasets.constituency.convert_starlang as convert_starlang

TURKISH_WORD_RE = re.compile(r"[{]turkish=([^}]+)[}]")
TURKISH_LABEL_RE = re.compile(r"[{]namedEntity=([^}]+)[}]")



def read_tree(text):
    """
    Reads in a tree, then extracts the word and the NER

    One problem is that it is unknown if there are cases of two separate items occurring consecutively

    Note that this is quite similar to the convert_starlang script for constituency.  
    """
    trees = tree_reader.read_trees(text)
    if len(trees) > 1:
        raise ValueError("Tree file had two trees!")
    tree = trees[0]
    words = []
    for label in tree.leaf_labels():
        match = TURKISH_WORD_RE.search(label)
        if match is None:
            raise ValueError("Could not find word in |{}|".format(label))
        word = match.group(1)
        word = word.replace("-LCB-", "{").replace("-RCB-", "}")

        match = TURKISH_LABEL_RE.search(label)
        if match is None:
            raise ValueError("Could not find ner in |{}|".format(label))
        tag = match.group(1)
        if tag == 'NONE' or tag == "null":
            tag = 'O'
        words.append((word, tag))

    return words

def read_starlang(paths):
    return convert_starlang.read_starlang(paths, conversion=read_tree, log=False)

def main():
    train, dev, test = convert_starlang.main(conversion=read_tree, log=False)

if __name__ == '__main__':
    main()

