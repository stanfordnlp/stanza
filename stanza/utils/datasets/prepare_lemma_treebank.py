"""
A script to prepare all lemma datasets.

For example, do
  python -m stanza.utils.datasets.prepare_lemma_treebank TREEBANK
such as
  python -m stanza.utils.datasets.prepare_lemma_treebank UD_English-EWT

and it will prepare each of train, dev, test
"""

import os
import shutil
import tempfile

import stanza.utils.datasets.common as common
import stanza.utils.datasets.prepare_tokenizer_treebank as prepare_tokenizer_treebank

from stanza.models.common.constant import treebank_to_short_name

def copy_lemmas(tokenizer_dir, tokenizer_file, lemma_dir, lemma_file, short_name):
    original = f"{tokenizer_dir}/{short_name}.{tokenizer_file}.conllu"
    copied = f"{lemma_dir}/{short_name}.{lemma_file}.conllu"

    shutil.copyfile(original, copied)

def process_treebank(treebank, paths):
    lemma_dir = paths["LEMMA_DATA_DIR"]
    os.makedirs(lemma_dir, exist_ok=True)

    short_name = treebank_to_short_name(treebank)
    short_language = short_name.split("_")[0]

    with tempfile.TemporaryDirectory() as tokenizer_dir:
        paths = dict(paths)
        paths["TOKENIZE_DATA_DIR"] = tokenizer_dir

        # first we process the tokenization data
        # TODO: we can skip processing the labels for the lemma datasets
        prepare_tokenizer_treebank.process_treebank(treebank, paths)

        # now we copy the processed conllu data files
        os.makedirs(lemma_dir, exist_ok=True)
        copy_lemmas(tokenizer_dir, "train.gold", lemma_dir, "train.in", short_name)
        copy_lemmas(tokenizer_dir, "dev.gold", lemma_dir, "dev.gold", short_name)
        copy_lemmas(tokenizer_dir, "dev.gold", lemma_dir, "dev.in", short_name)
        copy_lemmas(tokenizer_dir, "test.gold", lemma_dir, "test.gold", short_name)
        copy_lemmas(tokenizer_dir, "test.gold", lemma_dir, "test.in", short_name)


def main():
    common.main(process_treebank)

if __name__ == '__main__':
    main()


