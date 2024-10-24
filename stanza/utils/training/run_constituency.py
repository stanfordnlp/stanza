"""
Trains or scores a constituency model.

Currently a suuuuper preliminary script.

Example of how to run on multiple parsers at the same time on the Stanford workqueue:

for i in `echo 1000 1001 1002 1003 1004`; do nlprun -d a6000 "python3 stanza/utils/training/run_constituency.py vi_vlsp23 --use_bert --stage1_bert_finetun --save_name vi_vlsp23_$i.pt --seed $i --epochs 200 --force" -o vi_vlsp23_$i.out; done

"""

import logging
import os

from stanza.models import constituency_parser
from stanza.models.constituency.retagging import RETAG_METHOD
from stanza.utils.datasets.constituency import prepare_con_dataset
from stanza.utils.training import common
from stanza.utils.training.common import Mode, add_charlm_args, build_charlm_args, choose_charlm, find_wordvec_pretrain

from stanza.resources.default_packages import default_charlms, default_pretrains

logger = logging.getLogger('stanza')

def add_constituency_args(parser):
    add_charlm_args(parser)

    parser.add_argument('--use_bert', default=False, action="store_true", help='Use the default transformer for this language')

    parser.add_argument('--parse_text', dest='mode', action='store_const', const="parse_text", help='Parse a text file')

def build_wordvec_args(short_language, dataset, extra_args):
    if '--wordvec_pretrain_file' not in extra_args:
        # will throw an error if the pretrain can't be found
        wordvec_pretrain = find_wordvec_pretrain(short_language, default_pretrains)
        wordvec_args = ['--wordvec_pretrain_file', wordvec_pretrain]
    else:
        wordvec_args = []

    return wordvec_args

def build_default_args(paths, short_language, dataset, command_args, extra_args):
    if short_language in RETAG_METHOD:
        retag_args = ["--retag_method", RETAG_METHOD[short_language]]
    else:
        retag_args = []

    wordvec_args = build_wordvec_args(short_language, dataset, extra_args)

    charlm = choose_charlm(short_language, dataset, command_args.charlm, default_charlms, {})
    charlm_args = build_charlm_args(short_language, charlm, base_args=False)

    bert_args = common.choose_transformer(short_language, command_args, extra_args, warn=True, layers=True)
    default_args = retag_args + wordvec_args + charlm_args + bert_args

    return default_args

def build_model_filename(paths, short_name, command_args, extra_args):
    short_language, dataset = short_name.split("_", 1)

    default_args = build_default_args(paths, short_language, dataset, command_args, extra_args)

    train_args = ["--shorthand", short_name,
                  "--mode", "train"]
    train_args = train_args + default_args
    if command_args.save_name is not None:
        train_args.extend(["--save_name", command_args.save_name])
    if command_args.save_dir is not None:
        train_args.extend(["--save_dir", command_args.save_dir])
    args = constituency_parser.parse_args(train_args)
    save_name = constituency_parser.build_model_filename(args)
    return save_name


def run_treebank(mode, paths, treebank, short_name, temp_output_file, command_args, extra_args):
    constituency_dir = paths["CONSTITUENCY_DATA_DIR"]
    short_language, dataset = short_name.split("_")

    train_file = os.path.join(constituency_dir, f"{short_name}_train.mrg")
    dev_file   = os.path.join(constituency_dir, f"{short_name}_dev.mrg")
    test_file  = os.path.join(constituency_dir, f"{short_name}_test.mrg")

    if not os.path.exists(train_file) or not os.path.exists(dev_file) or not os.path.exists(test_file):
        logger.warning(f"The data for {short_name} is missing or incomplete.  Attempting to rebuild...")
        try:
            prepare_con_dataset.main(short_name)
        except:
            logger.error(f"Unable to build the data.  Please correctly build the files in {train_file}, {dev_file}, {test_file} and then try again.")
            raise

    default_args = build_default_args(paths, short_language, dataset, command_args, extra_args)

    if mode == Mode.TRAIN:
        train_args = ['--train_file', train_file,
                      '--eval_file', dev_file,
                      '--shorthand', short_name,
                      '--mode', 'train']
        train_args = train_args + default_args + extra_args
        logger.info("Running train step with args: {}".format(train_args))
        constituency_parser.main(train_args)

    if mode == Mode.SCORE_DEV or mode == Mode.TRAIN:
        dev_args = ['--eval_file', dev_file,
                    '--shorthand', short_name,
                    '--mode', 'predict']
        dev_args = dev_args + default_args + extra_args
        logger.info("Running dev step with args: {}".format(dev_args))
        constituency_parser.main(dev_args)

    if mode == Mode.SCORE_TEST or mode == Mode.TRAIN:
        test_args = ['--eval_file', test_file,
                     '--shorthand', short_name,
                     '--mode', 'predict']
        test_args = test_args + default_args + extra_args
        logger.info("Running test step with args: {}".format(test_args))
        constituency_parser.main(test_args)

    if mode == "parse_text":
        text_args = ['--shorthand', short_name,
                     '--mode', 'parse_text']
        text_args = text_args + default_args + extra_args
        logger.info("Processing text with args: {}".format(text_args))
        constituency_parser.main(text_args)

def main():
    common.main(run_treebank, "constituency", "constituency", add_constituency_args, sub_argparse=constituency_parser.build_argparse(), build_model_filename=build_model_filename)

if __name__ == "__main__":
    main()

