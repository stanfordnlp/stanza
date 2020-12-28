"""
A script to prepare all MWT datasets.

For example, do
  python -m stanza.utils.datasets.prepare_mwt_treebank TREEBANK
such as
  python -m stanza.utils.datasets.prepare_mwt_treebank UD_English-EWT

and it will prepare each of train, dev, test
"""

import argparse
import os
import shutil
import tempfile

import stanza.utils.datasets.common as common
import stanza.utils.datasets.prepare_tokenizer_treebank as prepare_tokenizer_treebank

from stanza.utils.datasets.contract_mwt import contract_mwt

def copy_conllu(tokenizer_dir, mwt_dir, short_name, dataset, particle):
    input_conllu_tokenizer = f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu"
    input_conllu_mwt = f"{mwt_dir}/{short_name}.{dataset}.{particle}.conllu"
    shutil.copyfile(input_conllu_tokenizer, input_conllu_mwt)

def process_treebank(treebank, paths, args):
    short_name = common.project_to_short_name(treebank)

    mwt_dir = paths["MWT_DATA_DIR"]
    os.makedirs(mwt_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tokenizer_dir:
        paths = dict(paths)
        paths["TOKENIZE_DATA_DIR"] = tokenizer_dir

        # first we process the tokenization data
        tokenizer_args = argparse.Namespace()
        tokenizer_args.augment = False
        tokenizer_args.prepare_labels = True
        prepare_tokenizer_treebank.process_treebank(treebank, paths, tokenizer_args)

        copy_conllu(tokenizer_dir, mwt_dir, short_name, "train", "in")
        copy_conllu(tokenizer_dir, mwt_dir, short_name, "dev", "gold")
        copy_conllu(tokenizer_dir, mwt_dir, short_name, "test", "gold")

        shutil.copyfile(prepare_tokenizer_treebank.mwt_name(tokenizer_dir, short_name, "train"),
                        prepare_tokenizer_treebank.mwt_name(mwt_dir, short_name, "train"))
        shutil.copyfile(prepare_tokenizer_treebank.mwt_name(tokenizer_dir, short_name, "dev"),
                        prepare_tokenizer_treebank.mwt_name(mwt_dir, short_name, "dev"))
        shutil.copyfile(prepare_tokenizer_treebank.mwt_name(tokenizer_dir, short_name, "test"),
                        prepare_tokenizer_treebank.mwt_name(mwt_dir, short_name, "test"))

        contract_mwt(f"{mwt_dir}/{short_name}.dev.gold.conllu",
                     f"{mwt_dir}/{short_name}.dev.in.conllu")
        contract_mwt(f"{mwt_dir}/{short_name}.test.gold.conllu",
                     f"{mwt_dir}/{short_name}.test.in.conllu")

def main():
    common.main(process_treebank)

if __name__ == '__main__':
    main()


