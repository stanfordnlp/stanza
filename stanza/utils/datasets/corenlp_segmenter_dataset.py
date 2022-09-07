"""
Output a treebank's sentences in a form that can be processed by the CoreNLP CRF Segmenter

Run it as
  python3 -m stanza.utils.datasets.corenlp_segmenter_dataset <treebank>
such as
  python3 -m stanza.utils.datasets.corenlp_segmenter_dataset UD_Chinese-GSDSimp --output_dir $CHINESE_SEGMENTER_HOME
"""

import argparse
import os
import sys
import tempfile

import stanza.utils.datasets.common as common
import stanza.utils.datasets.prepare_tokenizer_treebank as prepare_tokenizer_treebank
import stanza.utils.default_paths as default_paths

from stanza.models.common.constant import treebank_to_short_name

def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('treebanks', type=str, nargs='*', default=["UD_Chinese-GSDSimp"], help='Which treebanks to run on')
    parser.add_argument('--output_dir', type=str, default='.', help='Where to put the results')
    return parser


def write_segmenter_file(output_filename, dataset):
    with open(output_filename, "w") as fout:
        for sentence in dataset:
            sentence = [x for x in sentence if not x.startswith("#")]
            sentence = [x for x in [y.strip() for y in sentence] if x]
            # eliminate MWE, although Chinese currently doesn't have any
            sentence = [x for x in sentence if x.split("\t")[0].find("-") < 0]

            text = " ".join(x.split("\t")[1] for x in sentence)
            fout.write(text)
            fout.write("\n")

def process_treebank(treebank, paths, output_dir):
    with tempfile.TemporaryDirectory() as tokenizer_dir:
        paths = dict(paths)
        paths["TOKENIZE_DATA_DIR"] = tokenizer_dir

        short_name = treebank_to_short_name(treebank)
        
        # first we process the tokenization data
        args = argparse.Namespace()
        args.augment = False
        args.prepare_labels = False
        prepare_tokenizer_treebank.process_treebank(treebank, paths, args)

        # TODO: these names should be refactored
        train_file = f"{tokenizer_dir}/{short_name}.train.gold.conllu"
        dev_file = f"{tokenizer_dir}/{short_name}.dev.gold.conllu"
        test_file = f"{tokenizer_dir}/{short_name}.test.gold.conllu"

        train_set = common.read_sentences_from_conllu(train_file)
        dev_set = common.read_sentences_from_conllu(dev_file)
        test_set = common.read_sentences_from_conllu(test_file)

        train_out = os.path.join(output_dir, f"{short_name}.train.seg.txt")
        test_out = os.path.join(output_dir, f"{short_name}.test.seg.txt")

        write_segmenter_file(train_out, train_set + dev_set)
        write_segmenter_file(test_out, test_set)

def main():
    parser = build_argparse()
    args = parser.parse_args()

    paths = default_paths.get_default_paths()
    for treebank in args.treebanks:
        process_treebank(treebank, paths, args.output_dir)

if __name__ == '__main__':
    main()

