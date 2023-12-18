import os
import sys

from stanza.utils.datasets.common import find_treebank_dataset_file
from stanza.utils.default_paths import get_default_paths
from stanza.models.lemma_classifier import prepare_dataset

SECTIONS = ("train", "dev", "test")

# TODO: refactor this!
class UnknownDatasetError(ValueError):
    def __init__(self, dataset, text):
        super().__init__(text)
        self.dataset = dataset

def process_fa_perdt(paths, short_name):
    word = "شد"
    upos = "VERB"
    allowed_lemmas = "کرد|شد"

    # TODO: there's a function somewhere which maps "fa_perdt" to UD_Persian-PerDT
    treebank = "UD_Persian-PerDT"

    udbase_dir = paths["UDBASE"]

    # TODO: make this a path in default_paths
    output_dir = os.path.join("data", "lemma_classifier")
    os.makedirs(output_dir, exist_ok=True)

    for section in SECTIONS:
        filename = find_treebank_dataset_file(treebank, udbase_dir, section, "conllu", fail=True)
        output_filename = os.path.join(output_dir, "%s.%s.lemma" % (short_name, section))
        args = ["--conll_path", filename,
                "--target_word", word,
                "--target_upos", upos,
                "--output_path", output_filename,
                "--allowed_lemmas", allowed_lemmas]
        prepare_dataset.main(args)


DATASET_MAPPING = {
    "fa_perdt":          process_fa_perdt,
}


def main(dataset_name):
    paths = get_default_paths()
    print("Processing %s" % dataset_name)

    # obviously will want to multiplex to multiple languages / datasets
    if dataset_name in DATASET_MAPPING:
        DATASET_MAPPING[dataset_name](paths, dataset_name)
    else:
        raise UnknownDatasetError(dataset_name, f"dataset {dataset_name} currently not handled by prepare_ner_dataset")
    print("Done processing %s" % dataset_name)

if __name__ == '__main__':
    main(sys.argv[1])
