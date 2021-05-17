"""
This script allows for training or testing on dev / test of the UD mwt tools.

If run with a single treebank name, it will train or test that treebank.
If run with ud_all or all_ud, it will iterate over all UD treebanks it can find.

Mode can be set to train&dev with --train, to dev set only
with --score_dev, and to test set only with --score_test.

Treebanks are specified as a list.  all_ud or ud_all means to look for
all UD treebanks.

Extra arguments are passed to mwt.  In case the run script
itself is shadowing arguments, you can specify --extra_args as a
parameter to mark where the mwt arguments start.
"""


import logging
import math

from stanza.models import mwt_expander
from stanza.models.common.doc import Document
from stanza.utils.conll import CoNLL
from stanza.utils.training import common
from stanza.utils.training.common import Mode

from stanza.utils.max_mwt_length import max_mwt_length

logger = logging.getLogger('stanza')

def check_mwt(filename):
    """
    Checks whether or not there are MWTs in the given conll file
    """
    doc = CoNLL.conll2doc(filename)
    data = doc.get_mwt_expansions(False)
    return len(data) > 0

def run_treebank(mode, paths, treebank, short_name,
                 temp_output_file, command_args, extra_args):
    short_language = short_name.split("_")[0]

    mwt_dir          = paths["MWT_DATA_DIR"]

    train_file       = f"{mwt_dir}/{short_name}.train.in.conllu"
    dev_in_file      = f"{mwt_dir}/{short_name}.dev.in.conllu"
    dev_gold_file    = f"{mwt_dir}/{short_name}.dev.gold.conllu"
    dev_output_file  = temp_output_file if temp_output_file else f"{mwt_dir}/{short_name}.dev.pred.conllu"
    test_in_file     = f"{mwt_dir}/{short_name}.test.in.conllu"
    test_gold_file   = f"{mwt_dir}/{short_name}.test.gold.conllu"
    test_output_file = temp_output_file if temp_output_file else f"{mwt_dir}/{short_name}.test.pred.conllu"

    train_json       = f"{mwt_dir}/{short_name}-ud-train-mwt.json"
    dev_json         = f"{mwt_dir}/{short_name}-ud-dev-mwt.json"
    test_json        = f"{mwt_dir}/{short_name}-ud-test-mwt.json"

    if not check_mwt(train_file):
        logger.info("No training MWTS found for %s.  Skipping" % treebank)
        return
    
    if not check_mwt(dev_in_file):
        logger.warning("No dev MWTS found for %s.  Skipping" % treebank)
        return

    if mode == Mode.TRAIN:
        max_mwt_len = math.ceil(max_mwt_length([train_json, dev_json]) * 1.1 + 1)
        logger.info("Max len: %f" % max_mwt_len)
        train_args = ['--train_file', train_file,
                      '--eval_file', dev_in_file,
                      '--output_file', dev_output_file,
                      '--gold_file', dev_gold_file,
                      '--lang', short_language,
                      '--shorthand', short_name,
                      '--mode', 'train',
                      '--max_dec_len', str(max_mwt_len)]
        train_args = train_args + extra_args
        logger.info("Running train step with args: {}".format(train_args))
        mwt_expander.main(train_args)

    if mode == Mode.SCORE_DEV or mode == Mode.TRAIN:
        dev_args = ['--eval_file', dev_in_file,
                    '--output_file', dev_output_file,
                    '--gold_file', dev_gold_file,
                    '--lang', short_language,
                    '--shorthand', short_name,
                    '--mode', 'predict']
        dev_args = dev_args + extra_args
        logger.info("Running dev step with args: {}".format(dev_args))
        mwt_expander.main(dev_args)

        results = common.run_eval_script_mwt(dev_gold_file, dev_output_file)
        logger.info("Finished running dev set on\n{}\n{}".format(treebank, results))

    if mode == Mode.SCORE_TEST:
        test_args = ['--eval_file', test_in_file,
                     '--output_file', test_output_file,
                     '--gold_file', test_gold_file,
                     '--lang', short_language,
                     '--shorthand', short_name,
                     '--mode', 'predict']
        test_args = test_args + extra_args
        logger.info("Running test step with args: {}".format(test_args))
        mwt_expander.main(test_args)

        results = common.run_eval_script_mwt(test_gold_file, test_output_file)
        logger.info("Finished running test set on\n{}\n{}".format(treebank, results))

def main():
    common.main(run_treebank, "mwt", "mwt_expander")

if __name__ == "__main__":
    main()

