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

from stanza.models.common.constant import treebank_to_short_name
import stanza.utils.datasets.common as common
import stanza.utils.datasets.prepare_tokenizer_treebank as prepare_tokenizer_treebank

from stanza.utils.datasets.contract_mwt import contract_mwt

def copy_conllu(tokenizer_dir, mwt_dir, short_name, dataset, particle):
    input_conllu_tokenizer = f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu"
    input_conllu_mwt = f"{mwt_dir}/{short_name}.{dataset}.{particle}.conllu"
    shutil.copyfile(input_conllu_tokenizer, input_conllu_mwt)

def process_treebank(treebank, paths, args):
    short_name = treebank_to_short_name(treebank)

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

        for shard in ("train", "dev", "test"):
            source_filename = prepare_tokenizer_treebank.mwt_name(tokenizer_dir, short_name, shard)
            dest_filename = prepare_tokenizer_treebank.mwt_name(mwt_dir, short_name, shard)
            print("Copying from %s to %s" % (source_filename, dest_filename))
            shutil.copyfile(source_filename, dest_filename)

        contract_mwt(f"{mwt_dir}/{short_name}.dev.gold.conllu",
                     f"{mwt_dir}/{short_name}.dev.in.conllu")
        contract_mwt(f"{mwt_dir}/{short_name}.test.gold.conllu",
                     f"{mwt_dir}/{short_name}.test.in.conllu")

def main():
    common.main(process_treebank)

if __name__ == '__main__':
    main()


