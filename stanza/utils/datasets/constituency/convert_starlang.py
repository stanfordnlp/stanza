
import os
import re

from tqdm import tqdm

from stanza.models.constituency import parse_tree
from stanza.models.constituency import tree_reader

TURKISH_RE = re.compile(r"[{]turkish=([^}]+)[}]")

DISALLOWED_LABELS = ('DT', 'DET', 's', 'vp', 'AFVP', 'CONJ', 'INTJ', '-XXX-')

def read_tree(text):
    """
    Reads in a tree, then extracts specifically the word from the specific format used

    Also converts LCB/RCB as needed
    """
    trees = tree_reader.read_trees(text)
    if len(trees) > 1:
        raise ValueError("Tree file had two trees!")
    tree = trees[0]
    labels = tree.leaf_labels()
    new_labels = []
    for label in labels:
        match = TURKISH_RE.search(label)
        if match is None:
            raise ValueError("Could not find word in |{}|".format(label))
        word = match.group(1)
        word = word.replace("-LCB-", "{").replace("-RCB-", "}")
        new_labels.append(word)

    tree = tree.replace_words(new_labels)
    #tree = tree.remap_constituent_labels(LABEL_MAP)
    con_labels = tree.get_unique_constituent_labels([tree])
    if any(label in DISALLOWED_LABELS for label in con_labels):
        raise ValueError("found an unexpected phrasal node {}".format(label))
    return tree

def read_files(filenames, conversion, log):
    trees = []
    for filename in filenames:
        with open(filename, encoding="utf-8") as fin:
            text = fin.read()
        try:
            tree = conversion(text)
            if tree is not None:
                trees.append(tree)
        except ValueError as e:
            if log:
                print("-----------------\nFound an error in {}: {} Original text: {}".format(filename, e, text))
    return trees

def read_starlang(paths, conversion=read_tree, log=True):
    """
    Read the starlang trees, converting them using the given method.

    read_tree or any other conversion turns one file at a time to a sentence.
    log is whether or not to log a ValueError - the NER division has many missing labels
    """
    if isinstance(paths, str):
        paths = (paths,)

    train_files = []
    dev_files = []
    test_files = []

    for path in paths:
        tree_files = [os.path.join(path, x) for x in os.listdir(path)]
        train_files.extend([x for x in tree_files if x.endswith(".train")])
        dev_files.extend([x for x in tree_files if x.endswith(".dev")])
        test_files.extend([x for x in tree_files if x.endswith(".test")])

    print("Reading %d total files" % (len(train_files) + len(dev_files) + len(test_files)))
    train_treebank = read_files(tqdm(train_files), conversion=conversion, log=log)
    dev_treebank = read_files(tqdm(dev_files), conversion=conversion, log=log)
    test_treebank = read_files(tqdm(test_files), conversion=conversion, log=log)

    return train_treebank, dev_treebank, test_treebank

def main(conversion=read_tree, log=True):
    paths = ["extern_data/constituency/turkish/TurkishAnnotatedTreeBank-15",
             "extern_data/constituency/turkish/TurkishAnnotatedTreeBank2-15",
             "extern_data/constituency/turkish/TurkishAnnotatedTreeBank2-20"]
    train_treebank, dev_treebank, test_treebank = read_starlang(paths, conversion=conversion, log=log)

    print("Train: %d" % len(train_treebank))
    print("Dev: %d" % len(dev_treebank))
    print("Test: %d" % len(test_treebank))

    print(train_treebank[0])
    return train_treebank, dev_treebank, test_treebank

if __name__ == '__main__':
    main()
