"""
A script to prepare all lemma datasets.

For example, do
  python -m stanza.utils.datasets.prepare_lemma_treebank TREEBANK
such as
  python -m stanza.utils.datasets.prepare_lemma_treebank UD_English-EWT

and it will prepare each of train, dev, test
"""

import os

from stanza.models.common.constant import treebank_to_short_name

import stanza.utils.datasets.common as common
import stanza.utils.datasets.prepare_tokenizer_treebank as prepare_tokenizer_treebank

import stanza.utils.datasets.prepare_lemma_classifier as prepare_lemma_classifier

from stanza.utils.datasets.dataset_retagging import Tags, add_retagging_args, build_retagger
from stanza.utils.datasets.lemma import convert_de_wiktionary, convert_en_specialist, convert_sl_sloleks

# Datasets with an external lexicon.  Each function converts the
# lexicon from STANZA_EXTERN_DIR into the dictionary file given
EXTERNAL_DICTIONARIES = {
    'de_combined': convert_de_wiktionary.build_dictionary,
    'en_combined': convert_en_specialist.build_dictionary,
    'sl_combined': convert_sl_sloleks.build_dictionary,
}

def add_specific_args(parser) -> None:
    parser.add_argument('--no_lemma_classifier', dest='lemma_classifier', action='store_false', default=True,
                        help="Don't use the lemma classifier datasets.  Default is to build lemma classifier as part of the original lemmatizer")
    parser.add_argument('--no_external_dictionary', dest='external_dictionary', action='store_false', default=True,
                        help="Don't write the external lexicon for datasets which have one, such as German Wiktionary for de_combined, the SPECIALIST Lexicon for en_combined, or Sloleks for sl_combined.  Removes any dictionary file left from an earlier run")
    add_retagging_args(parser)

def dictionary_filename(lemma_dir, short_name):
    """
    The external dictionary for a dataset, which goes next to its train/dev/test files
    """
    return os.path.join(lemma_dir, f"{short_name}.dict.conllu")

def prepare_external_dictionary(paths, short_name, use_external_dictionary=True):
    """
    Write the external lexicon for short_name, if it has one, to {short_name}.dict.conllu in LEMMA_DATA_DIR

    If use_external_dictionary is False, an existing dictionary file
    is removed instead, so that run_lemma does not train with it.

    Returns the dictionary filename, or None if no dictionary was written
    """
    build_fn = EXTERNAL_DICTIONARIES.get(short_name)
    if build_fn is None:
        return None

    lemma_dir = paths["LEMMA_DATA_DIR"]
    dict_file = dictionary_filename(lemma_dir, short_name)
    if not use_external_dictionary:
        if os.path.exists(dict_file):
            print("Removing %s, as the external dictionary was turned off" % dict_file)
            os.remove(dict_file)
        return None

    os.makedirs(lemma_dir, exist_ok=True)
    print("Writing the external dictionary for %s to %s" % (short_name, dict_file))
    build_fn(paths, dict_file)
    return dict_file

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
    with open(train_file, encoding="utf-8") as fin:
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

def process_treebank(treebank, model_type, paths, args):
    short_name = treebank_to_short_name(treebank)
    # done first so that a missing lexicon fails before any of the treebank is written
    prepare_external_dictionary(paths, short_name, args.external_dictionary)

    if treebank.startswith("UD_"):
        udbase_dir = paths["UDBASE"]
        input_conllu = common.find_treebank_dataset_file(treebank, udbase_dir, "train", "conllu")
        if not input_conllu:
            input_conllu = common.find_treebank_dataset_file(treebank, udbase_dir, "test", "conllu", fail=True)
        augment = check_lemmas(input_conllu)
        if not augment:
            print("No lemma information found in %s.  Not augmenting the dataset" % input_conllu)
    else:
        # TODO: check the data to see if there are lemmas or not
        augment = True

    if args.tag_method is Tags.GOLD:
        prepare_tokenizer_treebank.copy_conllu_treebank(treebank, model_type, paths, paths["LEMMA_DATA_DIR"], args, augment=augment)
    elif args.tag_method is Tags.PREDICTED:
        retag_dataset = build_retagger(treebank, paths, args)
        prepare_tokenizer_treebank.copy_conllu_treebank(treebank, model_type, paths, paths["LEMMA_DATA_DIR"], args=args, augment=augment, postprocess=retag_dataset)
    else:
        raise ValueError("Unknown tags method: {}".format(args.tag_method))

    if args.lemma_classifier and short_name in prepare_lemma_classifier.DATASET_MAPPING:
        prepare_lemma_classifier.main(short_name)
    return True

def main():
    common.main(process_treebank, common.ModelType.LEMMA, add_specific_args)

if __name__ == '__main__':
    main()


