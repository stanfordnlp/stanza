import os
import sys

from stanza.utils.datasets.common import find_treebank_dataset_file, UnknownDatasetError
from stanza.utils.default_paths import get_default_paths
from stanza.models.lemma_classifier import prepare_dataset
from stanza.models.common.short_name_to_treebank import short_name_to_treebank

SECTIONS = ("train", "dev", "test")

def process_treebank(paths, short_name, word, upos, allowed_lemmas, sections=SECTIONS):
    treebank = short_name_to_treebank(short_name)
    udbase_dir = paths["UDBASE"]

    output_dir = paths["LEMMA_CLASSIFIER_DATA_DIR"]
    os.makedirs(output_dir, exist_ok=True)

    output_filenames = []

    for section in sections:
        filename = find_treebank_dataset_file(treebank, udbase_dir, section, "conllu", fail=True)
        output_filename = os.path.join(output_dir, "%s.%s.lemma" % (short_name, section))
        args = ["--conll_path", filename,
                "--target_word", word,
                "--target_upos", upos,
                "--output_path", output_filename]
        if allowed_lemmas is not None:
            args.extend(["--allowed_lemmas", allowed_lemmas])
        prepare_dataset.main(args)
        output_filenames.append(output_filename)

    return output_filenames

def process_ja_gsd(paths, short_name):
    # this one looked promising, but only has 10 total dev & test cases
    # 行っ VERB Counter({'行う': 60, '行く': 38})
    # could possibly do
    # ない AUX Counter({'ない': 383, '無い': 99})
    # なく AUX Counter({'無い': 53, 'ない': 42})
    # currently this one has enough in the dev & test data
    # and functions well
    # だ AUX Counter({'だ': 237, 'た': 67})
    word = "だ"
    upos = "AUX"
    allowed_lemmas = None

    process_treebank(paths, short_name, word, upos, allowed_lemmas)

def process_fa_perdt(paths, short_name):
    word = "شد"
    upos = "VERB"
    allowed_lemmas = "کرد|شد"

    process_treebank(paths, short_name, word, upos, allowed_lemmas)

def process_hi_hdtb(paths, short_name):
    word = "के"
    upos = "ADP"
    allowed_lemmas = "का|के"

    process_treebank(paths, short_name, word, upos, allowed_lemmas)

def process_ar_padt(paths, short_name):
    word = "أن"
    upos = "SCONJ"
    allowed_lemmas = "أَن|أَنَّ"

    process_treebank(paths, short_name, word, upos, allowed_lemmas)

DATASET_MAPPING = {
    "ar_padt":           process_ar_padt,
    "fa_perdt":          process_fa_perdt,
    "hi_hdtb":           process_hi_hdtb,
    "ja_gsd":            process_ja_gsd,
}


def main(dataset_name):
    paths = get_default_paths()
    print("Processing %s" % dataset_name)

    # obviously will want to multiplex to multiple languages / datasets
    if dataset_name in DATASET_MAPPING:
        DATASET_MAPPING[dataset_name](paths, dataset_name)
    else:
        raise UnknownDatasetError(dataset_name, f"dataset {dataset_name} currently not handled by prepare_lemma_classifier.py")
    print("Done processing %s" % dataset_name)

if __name__ == '__main__':
    main(sys.argv[1])
