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

def process_treebank(treebank, paths, args):
    prepare_tokenizer_treebank.copy_conllu_treebank(treebank, paths, paths["LEMMA_DATA_DIR"])

def main():
    common.main(process_treebank)

if __name__ == '__main__':
    main()


