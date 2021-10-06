"""
Converts Turin's constituency dataset

Turin University put out a freely available constituency dataset in 2011.
It is not as large as VIT or ISST, but it is free, which is nice.

The 2011 parsing task combines trees from several sources:
http://www.di.unito.it/~tutreeb/evalita-parsingtask-11.html

There is another site for Turin treebanks:
http://www.di.unito.it/~tutreeb/treebanks.html

Weirdly, the most recent versions of the Evalita trees are not there.
The most relevant parts are the ParTUT downloads.  As of Sep. 2021:

http://www.di.unito.it/~tutreeb/corpora/Par-TUT/tutINpenn/italian/JRCAcquis_It.pen
http://www.di.unito.it/~tutreeb/corpora/Par-TUT/tutINpenn/italian/UDHR_It.pen
http://www.di.unito.it/~tutreeb/corpora/Par-TUT/tutINpenn/italian/CC_It.pen
http://www.di.unito.it/~tutreeb/corpora/Par-TUT/tutINpenn/italian/FB_It.pen
http://www.di.unito.it/~tutreeb/corpora/Par-TUT/tutINpenn/italian/WIT3_It.pen

We can't simply cat all these files together as there are a bunch of
asterisks as comments and the files may have some duplicates.  For
example, the JRCAcquis piece has many duplicates.  Also, some don't
pass validation for one reason or another.

One oddity of these data files is that the MWT are denoted by doubling
the token.  The token is not split as would be expected, though.  We try
to use stanza's MWT tokenizer for IT to split the tokens, with some
rules added by hand in BIWORD_SPLITS.  Two are still unsplit, though...
"""

import glob
import os
import re
import sys

import stanza
from stanza.models.constituency import parse_tree
from stanza.models.constituency import tree_reader

def load_without_asterisks(in_file, encoding='utf-8'):
    with open(in_file, encoding=encoding) as fin:
        new_lines = [x if x.find("********") < 0 else "\n" for x in fin.readlines()]
    if len(new_lines) > 0 and not new_lines[-1].endswith("\n"):
        new_lines[-1] = new_lines[-1] + "\n"
    return new_lines

CONSTITUENT_SPLIT = re.compile("[-=#+0-9]")

# JRCA is almost entirely duplicates
# WIT3 follows a different annotation scheme
FILES_TO_ELIMINATE = ["JRCAcquis_It.pen", "WIT3_It.pen"]

# assuming this is a typo
REMAP_NODES = { "Sbar" : "SBAR" }

REMAP_WORDS = { "-LSB-": "[", "-RSB-": "]" }

# these mostly seem to be mistakes
# maybe Vbar and ADVbar should be converted to something else?
NODES_TO_ELIMINATE = ["C", "PHRASP", "PRDT", "Vbar", "parte", "ADVbar"]

UNKNOWN_SPLITS = set()

# a map of splits that the tokenizer or MWT doesn't handle well
BIWORD_SPLITS = { "offertogli": ("offerto", "gli"),
                  "offertegli": ("offerte", "gli"),
                  "formatasi": ("formata", "si"),
                  "formatosi": ("formato", "si"),
                  "multiplexarlo": ("multiplexar", "lo"),
                  "esibirsi": ("esibir", "si"),
                  "pagarne": ("pagar", "ne"),
                  "recarsi": ("recar", "si"),
                  "trarne": ("trar", "ne"),
                  "esserci": ("esser", "ci"),
                  "aprirne": ("aprir", "ne"),
                  "farle": ("far", "le"),
                  "disporne": ("dispor", "ne"),
                  "andargli": ("andar", "gli"),
                  "CONSIDERARSI": ("CONSIDERAR", "SI"),
                  "conferitegli": ("conferite", "gli"),
                  "formatasi": ("formata", "si"),
                  "formatosi": ("formato", "si"),
                  "Formatisi": ("Formati", "si"),
                  "multiplexarlo": ("multiplexar", "lo"),
                  "esibirsi": ("esibir", "si"),
                  "pagarne": ("pagar", "ne"),
                  "recarsi": ("recar", "si"),
                  "trarne": ("trar", "ne"),
                  "temerne": ("temer", "ne"),
                  "esserci": ("esser", "ci"),
                  "esservi": ("esser", "vi"),
                  "restituirne": ("restituir", "ne"),
                  "col": ("con", "il"),
                  "cogli": ("con", "gli"),
                  "dirgli": ("dir", "gli"),
                  "opporgli": ("oppor", "gli"),
                  "eccolo": ("ecco", "lo"),
                  "Eccolo": ("Ecco", "lo"),
                  "Eccole": ("Ecco", "le"),
                  "farci": ("far", "ci"),
                  "farli": ("far", "li"),
                  "farne": ("far", "ne"),
                  "farsi": ("far", "si"),
                  "farvi": ("far", "vi"),
                  "Connettiti": ("Connetti", "ti"),
                  "APPLICARSI": ("APPLICAR", "SI"),
                  # This is not always two words, but if it IS two words,
                  # it gets split like this
                  "assicurati": ("assicura", "ti"),
                  "Fatti": ("Fai", "te"),
                  "ai": ("a", "i"),
                  "Ai": ("A", "i"),
                  "AI": ("A", "I"),
                  "al": ("a", "il"),
                  "Al": ("A", "il"),
                  "AL": ("A", "IL"),
                  "coi": ("con", "i"),
                  "colla": ("con", "la"),
                  "colle": ("con", "le"),
                  "dal": ("da", "il"),
                  "Dal": ("Da", "il"),
                  "DAL": ("DA", "IL"),
                  "dei": ("di", "i"),
                  "Dei": ("Di", "i"),
                  "DEI": ("DI", "I"),
                  "del": ("di", "il"),
                  "Del": ("Di", "il"),
                  "DEL": ("DI", "IL"),
                  "nei": ("in", "i"),
                  "NEI": ("IN", "I"),
                  "nel": ("in", "il"),
                  "Nel": ("In", "il"),
                  "NEL": ("IN", "IL"),
                  "pel": ("per", "il"),
                  "sui": ("su", "i"),
                  "Sui": ("Su", "i"),
                  "sul": ("su", "il"),
                  "Sul": ("Su", "il"),
                  ",": (",", ","),
                  ".": (".", "."),
                  '"': ('"', '"'),
                  '-': ('-', '-'),
                  '-LRB-': ('-LRB-', '-LRB-'),
                  "garantirne": ("garantir", "ne"),
                  "aprirvi": ("aprir", "vi"),
                  "esimersi": ("esimer", "si"),
                  "opporsi": ("oppor", "si"),
}

CAP_BIWORD = re.compile("[A-Z]+_[A-Z]+")

def split_mwe(tree, pipeline):
    words = list(tree.leaf_labels())
    found = False
    for idx, word in enumerate(words[:-3]):
        if word == words[idx+1] and word == words[idx+2] and word == words[idx+3]:
            raise ValueError("Oh no, 4 consecutive words")

    for idx, word in enumerate(words[:-2]):
        if word == words[idx+1] and word == words[idx+2]:
            doc = pipeline(word)
            assert len(doc.sentences) == 1
            if len(doc.sentences[0].words) != 3:
                raise RuntimeError("Word {} not tokenized into 3 parts... thought all 3 part words were handled!".format(word))
            words[idx] = doc.sentences[0].words[0].text
            words[idx+1] = doc.sentences[0].words[1].text
            words[idx+2] = doc.sentences[0].words[2].text
            found = True

    for idx, word in enumerate(words[:-1]):
        if word == words[idx+1]:
            if word in BIWORD_SPLITS:
                first_word = BIWORD_SPLITS[word][0]
                second_word = BIWORD_SPLITS[word][1]
            elif CAP_BIWORD.match(word):
                first_word, second_word = word.split("_")
            else:
                doc = pipeline(word)
                assert len(doc.sentences) == 1
                if len(doc.sentences[0].words) == 2:
                    first_word = doc.sentences[0].words[0].text
                    second_word = doc.sentences[0].words[1].text
                else:
                    if word not in UNKNOWN_SPLITS:
                        UNKNOWN_SPLITS.add(word)
                        print("Could not figure out how to split {}\n  {}\n  {}".format(word, " ".join(words), tree))
                    continue

            words[idx] = first_word
            words[idx+1] = second_word
            found = True

    if found:
        tree = tree.replace_words(words)
    return tree


def load_trees(filename, pipeline):
    # some of the files are in latin-1 encoding rather than utf-8
    try:
        raw_text = load_without_asterisks(filename, "utf-8")
    except UnicodeDecodeError:
        raw_text = load_without_asterisks(filename, "latin-1")

    # also, some have messed up validation (it will be logged)
    # hence the broken_ok=True argument
    trees = tree_reader.read_trees("".join(raw_text), broken_ok=True)

    filtered_trees = []
    for tree in trees:
        if tree.children[0].label is None:
            print("Skipping a broken tree (missing label) in {}: {}".format(filename, tree))
            continue

        try:
            words = tuple(tree.leaf_labels())
        except ValueError:
            print("Skipping a broken tree (missing preterminal) in {}: {}".format(filename, tree))
            continue

        if any('www.facebook' in pt.label for pt in tree.preterminals()):
            print("Skipping a tree with a weird preterminal label in {}: {}".format(filename, tree))
            continue

        tree = tree.prune_none().simplify_labels(CONSTITUENT_SPLIT)

        if len(tree.children) > 1:
            print("Found a tree with a non-unary root!  {}: {}".format(filename, tree))
            continue
        if tree.children[0].is_preterminal():
            print("Found a tree with a single preterminal node!  {}: {}".format(filename, tree))
            continue

        # The expectation is that the retagging will handle this anyway
        for pt in tree.preterminals():
            if not pt.label:
                pt.label = "UNK"
                print("Found a tree with a blank preterminal label.  Setting it to UNK.  {}: {}".format(filename, tree))

        tree = tree.remap_constituent_labels(REMAP_NODES)
        tree = tree.remap_words(REMAP_WORDS)

        tree = split_mwe(tree, pipeline)
        if tree is None:
            continue

        constituents = set(parse_tree.Tree.get_unique_constituent_labels(tree))
        for weird_label in NODES_TO_ELIMINATE:
            if weird_label in constituents:
                break
        else:
            weird_label = None
        if weird_label is not None:
            print("Skipping a tree with a weird label {} in {}: {}".format(weird_label, filename, tree))
            continue

        filtered_trees.append(tree)

    return filtered_trees

def save_trees(out_file, trees):
    print("Saving {} trees to {}".format(len(trees), out_file))
    with open(out_file, "w", encoding="utf-8") as fout:
        for tree in trees:
            fout.write(str(tree))
            fout.write("\n")

def convert_it_turin(input_path, output_path):
    pipeline = stanza.Pipeline("it", processors="tokenize, mwt", tokenize_no_ssplit=True)

    os.makedirs(output_path, exist_ok=True)

    evalita_dir = os.path.join(input_path, "evalita")

    evalita_test = os.path.join(evalita_dir, "evalita11_TESTgold_CONPARSE.penn")
    it_test = os.path.join(output_path, "it_turin_test.mrg")
    test_trees = load_trees(evalita_test, pipeline)
    save_trees(it_test, test_trees)

    known_text = set()
    for tree in test_trees:
        words = tuple(tree.leaf_labels())
        assert words not in known_text
        known_text.add(words)

    evalita_train = os.path.join(output_path, "it_turin_train.mrg")
    evalita_files = glob.glob(os.path.join(evalita_dir, "*2011*penn"))
    turin_files = glob.glob(os.path.join(input_path, "turin", "*pen"))
    filenames = evalita_files + turin_files
    filtered_trees = []
    for filename in filenames:
        if os.path.split(filename)[1] in FILES_TO_ELIMINATE:
            continue

        trees = load_trees(filename, pipeline)
        file_trees = []

        for tree in trees:
            words = tuple(tree.leaf_labels())
            if words in known_text:
                print("Skipping a duplicate in {}: {}".format(filename, tree))
                continue

            known_text.add(words)

            file_trees.append(tree)

        filtered_trees.append((filename, file_trees))

    print("{} contains {} usable trees".format(evalita_test, len(test_trees)))
    print("  Unique constituents in {}: {}".format(evalita_test, parse_tree.Tree.get_unique_constituent_labels(test_trees)))

    train_trees = []
    dev_trees = []
    for filename, file_trees in filtered_trees:
        print("{} contains {} usable trees".format(filename, len(file_trees)))
        print("  Unique constituents in {}: {}".format(filename, parse_tree.Tree.get_unique_constituent_labels(file_trees)))
        for tree in file_trees:
            if len(train_trees) <= len(dev_trees) * 9:
                train_trees.append(tree)
            else:
                dev_trees.append(tree)

    it_train = os.path.join(output_path, "it_turin_train.mrg")
    save_trees(it_train, train_trees)

    it_dev = os.path.join(output_path, "it_turin_dev.mrg")
    save_trees(it_dev, dev_trees)

def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    convert_it_turin(input_path, output_path)

if __name__ == '__main__':
    main()
