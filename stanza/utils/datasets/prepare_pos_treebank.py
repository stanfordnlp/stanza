"""
A script to prepare all pos datasets.

For example, do
  python -m stanza.utils.datasets.prepare_pos_treebank TREEBANK
such as
  python -m stanza.utils.datasets.prepare_pos_treebank UD_English-EWT

and it will prepare each of train, dev, test
"""

import os
import shutil

import stanza.utils.datasets.common as common
import stanza.utils.datasets.prepare_tokenizer_treebank as prepare_tokenizer_treebank

def copy_conllu_file_or_zip(tokenizer_dir, tokenizer_file, dest_dir, dest_file, short_name):
    original = f"{tokenizer_dir}/{short_name}.{tokenizer_file}.zip"
    copied = f"{dest_dir}/{short_name}.{dest_file}.zip"

    if os.path.exists(original):
        print("Copying from %s to %s" % (original, copied))
        shutil.copyfile(original, copied)
    else:
        prepare_tokenizer_treebank.copy_conllu_file(tokenizer_dir, tokenizer_file, dest_dir, dest_file, short_name)


def process_treebank(treebank, model_type, paths, args):
    prepare_tokenizer_treebank.copy_conllu_treebank(treebank, model_type, paths, paths["POS_DATA_DIR"], postprocess=copy_conllu_file_or_zip)

def main():
    common.main(process_treebank, common.ModelType.POS)

if __name__ == '__main__':
    main()


