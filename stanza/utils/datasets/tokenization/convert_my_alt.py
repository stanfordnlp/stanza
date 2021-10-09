"""Converts the Myanmar ALT corpus to a tokenizer dataset.

The ALT corpus is in the form of constituency trees, which basically
means there is no guidance on where the whitespace belongs.  However,
in Myanmar writing, whitespace is apparently not actually required
anywhere.  The plan will be to make sentences where there is no
whitespace at all, along with a random selection of sentences
where some whitespace is randomly inserted.

The treebank is available here:

https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/

The following files describe the splits of the data:

https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/URL-train.txt
https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/URL-dev.txt
https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/URL-test.txt

and this is the actual treebank:

https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/my-alt-190530.zip

Download each of the files, then unzip the my-alt zip in place.
The expectation is this will produce a file
  my-alt-190530/data

The default expected path to the Myanmar data is
  extern_data/constituency/myanmar/my_alt/my-alt-190530/data
"""

import os
import random

from stanza.models.constituency.tree_reader import read_trees

def read_split(input_dir, section):
    """
    Reads the split description for train, dev, or test

    Format (at least for the Myanmar section of ALT) is:
      one description per line
      each line is URL.<number> <URL>
      we actually don't care about the URL itself
      all we want is the number, which we use to split up
      the tree file later

    Returns a set of numbers (as strings)
    """
    urls = set()
    filename = os.path.join(input_dir, "myanmar", "my_alt", "URL-%s.txt" % section)
    with open(filename) as fin:
        lines = fin.readlines()
    for line in lines:
        line = line.strip()
        if not line or not line.startswith("URL"):
            continue
        # split into URL.100161 and a bunch of description we don't care about
        line = line.split(maxsplit=1)
        # get just the number
        line = line[0].split(".")
        assert len(line) == 2
        assert line[0] == 'URL'
        urls.add(line[1])
    return urls
    
SPLITS = ("train", "dev", "test")

def read_dataset_splits(input_dir):
    """
    Call read_split for train, dev, and test

    Returns three sets: train, dev, test in order
    """
    url_splits = [read_split(input_dir, section) for section in SPLITS]
    for url_split, split in zip(url_splits, SPLITS):
        print("Split %s has %d files in it" % (split, len(url_split)))
    return url_splits

def read_alt_treebank(constituency_input_dir):
    """
    Read the splits, read the trees, and split the trees based on the split descriptions

    Trees in ALT are:
      <tree id> <tree brackets>
    The tree id will look like
      SNT.<url_id>.<line>
    All we care about from this id is the url_id, which we crossreference in the splits
    to figure out which split the tree is in.

    The tree itself we don't process much, although we do convert it to a ParseTree

    The result is three lists: train, dev, test trees
    """
    train_split, dev_split, test_split = read_dataset_splits(constituency_input_dir)

    datafile = os.path.join(constituency_input_dir, "myanmar", "my_alt", "my-alt-190530", "data")
    print("Reading trees from %s" % datafile)
    with open(datafile) as fin:
        tree_lines = fin.readlines()

    train_trees = []
    dev_trees = []
    test_trees = []

    for idx, tree_line in enumerate(tree_lines):
        tree_line = tree_line.strip()
        if not tree_line:
            continue
        dataset, tree_text = tree_line.split(maxsplit=1)
        dataset = dataset.split(".", 2)[1]

        trees = read_trees(tree_text)
        if len(trees) != 1:
            raise ValueError("Unexpected number of trees in line %d: %d" % (idx, len(trees)))
        tree = trees[0]

        if dataset in train_split:
            train_trees.append(tree)
        elif dataset in dev_split:
            dev_trees.append(tree)
        elif dataset in test_split:
            test_trees.append(tree)
        else:
            raise ValueError("Could not figure out which split line %d belongs to" % idx)

    return train_trees, dev_trees, test_trees

def write_sentence(fout, words, spaces):
    """
    Write a sentence based on the list of words.

    spaces is a fraction of the words which should randomly have spaces
    If 0.0, none of the words will have spaces
    This is because the Myanmar language doesn't require spaces, but
      spaces always separate words
    """
    full_text = "".join(words)
    fout.write("# text = %s\n" % full_text)

    for word_idx, word in enumerate(words):
        fake_dep = "root" if word_idx == 0 else "dep"
        fout.write("%d\t%s\t%s" % ((word_idx+1), word, word))
        fout.write("\t_\t_\t_")
        fout.write("\t%d\t%s" % (word_idx, fake_dep))
        fout.write("\t_\t")
        if random.random() > spaces:
            fout.write("SpaceAfter=No")
        else:
            fout.write("_")
        fout.write("\n")
    fout.write("\n")


def write_dataset(filename, trees, split):
    """
    Write all of the trees to the given filename
    """
    count = 0
    with open(filename, "w") as fout:
        # TODO: make some fraction have random spaces inserted
        for tree in trees:
            count = count + 1
            words = tree.leaf_labels()
            write_sentence(fout, words, spaces=0.0)
            # We include a small number of spaces to teach the model
            # that spaces always separate a word
            if split == 'train' and random.random() < 0.1:
                count = count + 1
                write_sentence(fout, words, spaces=0.05)
    print("Wrote %d sentences from %d trees to %s" % (count, len(trees), filename))

def convert_my_alt(constituency_input_dir, tokenizer_dir):
    """
    Read and then convert the Myanmar ALT treebank
    """
    random.seed(1234)
    tree_splits = read_alt_treebank(constituency_input_dir)

    output_filenames = [os.path.join(tokenizer_dir, "my_alt.%s.gold.conllu") % split for split in SPLITS]

    for filename, trees, split in zip(output_filenames, tree_splits, SPLITS):
        write_dataset(filename, trees, split)

def main():
    convert_my_alt("extern_data/constituency", "data/tokenize")

if __name__ == "__main__":
    main()
