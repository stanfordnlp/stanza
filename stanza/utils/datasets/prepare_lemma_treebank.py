"""
A script to prepare all lemma datasets.

For example, do
  python -m stanza.utils.datasets.prepare_lemma_treebank TREEBANK
such as
  python -m stanza.utils.datasets.prepare_lemma_treebank UD_English-EWT

and it will prepare each of train, dev, test
"""

import stanza.utils.datasets.common as common
import stanza.utils.datasets.prepare_tokenizer_treebank as prepare_tokenizer_treebank

def check_lemmas(train_file):
    """
    Check if a treebank has any lemmas in it

    For example, in Vietnamese-VTB, all the words and lemmas are exactly the same
    in Telugu-MTG, all the lemmas are blank
    """
    # could eliminate a few languages immediately based on UD 2.7
    # but what if a later dataset includes lemmas?
    #if short_language in ('vi', 'fro', 'th'):
    #    return False
    with open(train_file) as fin:
        for line in fin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pieces = line.split("\t")
            word = pieces[1].lower().strip()
            lemma = pieces[2].lower().strip()
            if not lemma or lemma == '_' or lemma == '-':
                continue
            if word == lemma:
                continue
            return True
    return False

def process_treebank(treebank, paths, args):
    if treebank.startswith("UD_"):
        udbase_dir = paths["UDBASE"]
        train_conllu = common.find_treebank_dataset_file(treebank, udbase_dir, "train", "conllu", fail=True)
        augment = check_lemmas(train_conllu)
        if not augment:
            print("No lemma information found in %s.  Not augmenting the dataset" % train_conllu)
    else:
        # TODO: check the data to see if there are lemmas or not
        augment = True
    prepare_tokenizer_treebank.copy_conllu_treebank(treebank, paths, paths["LEMMA_DATA_DIR"], augment=augment)

def main():
    common.main(process_treebank)

if __name__ == '__main__':
    main()


