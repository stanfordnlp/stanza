"""
A script to prepare all pos datasets.

For example, do
  python -m stanza.utils.datasets.prepare_pos_treebank TREEBANK
such as
  python -m stanza.utils.datasets.prepare_pos_treebank UD_English-EWT

and it will prepare each of train, dev, test
"""

import stanza.utils.datasets.common as common
import stanza.utils.datasets.prepare_tokenizer_treebank as prepare_tokenizer_treebank

def process_treebank(treebank, paths, args):
    prepare_tokenizer_treebank.copy_conllu_treebank(treebank, paths, paths["POS_DATA_DIR"])

def main():
    common.main(process_treebank)

if __name__ == '__main__':
    main()


