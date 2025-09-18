"""
This script allows for training or testing on dev / test of the UD tokenizer.

If run with a single treebank name, it will train or test that treebank.
If run with ud_all or all_ud, it will iterate over all UD treebanks it can find.

Mode can be set to train&dev with --train, to dev set only
with --score_dev, and to test set only with --score_test.

Treebanks are specified as a list.  all_ud or ud_all means to look for
all UD treebanks.

Extra arguments are passed to tokenizer.  In case the run script
itself is shadowing arguments, you can specify --extra_args as a
parameter to mark where the tokenizer arguments start.

Default behavior is to discard the output and just print the results.
To keep the results instead, use --save_output
"""

import logging
import math
import os

from stanza.models import tokenizer
from stanza.utils.avg_sent_len import avg_sent_len
from stanza.utils.training import common
from stanza.utils.training.common import Mode, add_charlm_args, build_tokenizer_charlm_args

logger = logging.getLogger('stanza')

def add_tokenizer_args(parser):
    add_charlm_args(parser)


def build_model_filename(paths, short_name, command_args, extra_args):
    short_language, dataset = short_name.split("_", 1)

    # TODO: can avoid downloading the charlm at this point, since we
    # might not even be training
    charlm_args = build_tokenizer_charlm_args(short_language, dataset, command_args.charlm)

    train_args = ["--shorthand", short_name,
                  "--mode", "train"]
    train_args = train_args + charlm_args + extra_args
    if command_args.save_name is not None:
        train_args.extend(["--save_name", command_args.save_name])
    if command_args.save_dir is not None:
        train_args.extend(["--save_dir", command_args.save_dir])
    args = tokenizer.parse_args(train_args)
    save_name = tokenizer.model_file_name(args)
    return save_name



def uses_dictionary(short_language):
    """
    Some of the languages (as shown here) have external dictionaries

    We found this helped the overall tokenizer performance
    If these can't be found, they can be extracted from the previous iteration of models
    """
    if short_language in ('ja', 'th', 'zh', 'zh-hans', 'zh-hant'):
        return True
    return False

def run_treebank(mode, paths, treebank, short_name,
                 temp_output_file, command_args, extra_args):
    tokenize_dir = paths["TOKENIZE_DATA_DIR"]

    short_language, dataset = short_name.split("_", 1)
    label_type = "--label_file"
    label_file = f"{tokenize_dir}/{short_name}-ud-train.toklabels"
    dev_type = "--txt_file"
    dev_file = f"{tokenize_dir}/{short_name}.dev.txt"
    test_type = "--txt_file"
    test_file = f"{tokenize_dir}/{short_name}.test.txt"
    train_type = "--txt_file"
    train_file = f"{tokenize_dir}/{short_name}.train.txt"
    train_dev_args = ["--dev_txt_file", dev_file, "--dev_label_file", f"{tokenize_dir}/{short_name}-ud-dev.toklabels"]
    
    if short_language == "zh" or short_language.startswith("zh-"):
        extra_args = ["--skip_newline"] + extra_args

    train_gold = f"{tokenize_dir}/{short_name}.train.gold.conllu"
    dev_gold = f"{tokenize_dir}/{short_name}.dev.gold.conllu"
    test_gold = f"{tokenize_dir}/{short_name}.test.gold.conllu"

    train_mwt = f"{tokenize_dir}/{short_name}-ud-train-mwt.json"
    dev_mwt = f"{tokenize_dir}/{short_name}-ud-dev-mwt.json"
    test_mwt = f"{tokenize_dir}/{short_name}-ud-test-mwt.json"

    train_pred = temp_output_file if temp_output_file else f"{tokenize_dir}/{short_name}.train.pred.conllu"
    dev_pred = temp_output_file if temp_output_file else f"{tokenize_dir}/{short_name}.dev.pred.conllu"
    test_pred = temp_output_file if temp_output_file else f"{tokenize_dir}/{short_name}.test.pred.conllu"

    charlm_args = build_tokenizer_charlm_args(short_language, dataset, command_args.charlm)

    if mode == Mode.TRAIN:
        seqlen = str(math.ceil(avg_sent_len(label_file) * 3 / 100) * 100)
        train_args = ([label_type, label_file, train_type, train_file, "--lang", short_language,
                       "--max_seqlen", seqlen, "--mwt_json_file", dev_mwt] +
                      train_dev_args +
                      ["--dev_conll_gold", dev_gold, "--conll_file", dev_pred, "--shorthand", short_name])
        if uses_dictionary(short_language):
            train_args = train_args + ["--use_dictionary"]
        train_args = train_args + charlm_args + extra_args
        logger.info("Running train step with args: {}".format(train_args))
        tokenizer.main(train_args)
    
    if mode == Mode.SCORE_DEV or mode == Mode.TRAIN:
        dev_args = ["--mode", "predict", dev_type, dev_file, "--lang", short_language,
                    "--conll_file", dev_pred, "--shorthand", short_name, "--mwt_json_file", dev_mwt]
        dev_args = dev_args + charlm_args + extra_args
        logger.info("Running dev step with args: {}".format(dev_args))
        tokenizer.main(dev_args)

        # TODO: log these results?  The original script logged them to
        # echo $results $args >> ${TOKENIZE_DATA_DIR}/${short}.results

        results = common.run_eval_script_tokens(dev_gold, dev_pred)
        logger.info("Finished running dev set on\n{}\n{}".format(treebank, results))

    if mode == Mode.SCORE_TEST or mode == Mode.TRAIN:
        test_args = ["--mode", "predict", test_type, test_file, "--lang", short_language,
                     "--conll_file", test_pred, "--shorthand", short_name, "--mwt_json_file", test_mwt]
        test_args = test_args + charlm_args + extra_args
        logger.info("Running test step with args: {}".format(test_args))
        tokenizer.main(test_args)

        results = common.run_eval_script_tokens(test_gold, test_pred)
        logger.info("Finished running test set on\n{}\n{}".format(treebank, results))

    if mode == Mode.SCORE_TRAIN:
        test_args = ["--mode", "predict", test_type, train_file, "--lang", short_language,
                     "--conll_file", train_pred, "--shorthand", short_name, "--mwt_json_file", train_mwt]
        test_args = test_args + extra_args
        logger.info("Running test step with args: {}".format(test_args))
        tokenizer.main(test_args)

        results = common.run_eval_script_tokens(train_gold, train_pred)
        logger.info("Finished running train set as a test on\n{}\n{}".format(treebank, results))



def main():
    common.main(run_treebank, "tokenize", "tokenizer", add_tokenizer_args, sub_argparse=tokenizer.build_argparse(), build_model_filename=build_model_filename)

if __name__ == "__main__":
    main()
