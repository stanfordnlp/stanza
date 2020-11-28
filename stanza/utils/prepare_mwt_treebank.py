"""
A script to prepare all MWT datasets.

As a side effect, it prepares tokenization datasets as well.

For example, do
  python -m stanza.utils.prepare_mwt_treebank TREEBANK
such as
  python -m stanza.utils.prepare_mwt_treebank UD_English-EWT

and it will prepare each of train, dev, test
"""

import os
import shutil
import sys

import stanza.utils.default_paths as default_paths
import stanza.utils.prepare_tokenizer_treebank as prepare_tokenizer_treebank

from stanza.models.common.constant import treebank_to_short_name
from stanza.utils.contract_mwt import contract_mwt

def copy_conllu(tokenizer_dir, mwt_dir, short_name, dataset, particle):
    input_conllu_tokenizer = f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu"
    input_conllu_mwt = f"{mwt_dir}/{short_name}.{dataset}.{particle}.conllu"
    shutil.copyfile(input_conllu_tokenizer, input_conllu_mwt)

def process_treebank(treebank, paths):
    tokenizer_dir = paths["TOKENIZE_DATA_DIR"]
    mwt_dir = paths["MWT_DATA_DIR"]

    short_name = treebank_to_short_name(treebank)

    # first we process the tokenization data
    prepare_tokenizer_treebank.process_treebank(treebank, paths)

    os.makedirs(mwt_dir, exist_ok=True)

    copy_conllu(tokenizer_dir, mwt_dir, short_name, "train", "in")
    copy_conllu(tokenizer_dir, mwt_dir, short_name, "dev", "gold")
    copy_conllu(tokenizer_dir, mwt_dir, short_name, "test", "gold")

    contract_mwt(f"{mwt_dir}/{short_name}.dev.gold.conllu",
                 f"{mwt_dir}/{short_name}.dev.in.conllu")
    contract_mwt(f"{mwt_dir}/{short_name}.test.gold.conllu",
                 f"{mwt_dir}/{short_name}.test.in.conllu")

def main():
    # TODO: can refactor this with the tokenizer equivalent
    if len(sys.argv) == 1:
        raise ValueError("Need to provide a treebank name")

    treebank = sys.argv[1]
    paths = default_paths.get_default_paths()
    if treebank.lower() in ('ud_all', 'all_ud'):
        treebanks = prepare_tokenizer_treebank.get_ud_treebanks(paths["UDBASE"])
        for t in treebanks:
            process_treebank(t, paths)
    else:
        process_treebank(treebank, paths)

if __name__ == '__main__':
    main()


