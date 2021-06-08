"""
This script allows for training or testing on dev / test of the UD lemmatizer.

If run with a single treebank name, it will train or test that treebank.
If run with ud_all or all_ud, it will iterate over all UD treebanks it can find.

Mode can be set to train&dev with --train, to dev set only
with --score_dev, and to test set only with --score_test.

Treebanks are specified as a list.  all_ud or ud_all means to look for
all UD treebanks.

Extra arguments are passed to the lemmatizer.  In case the run script
itself is shadowing arguments, you can specify --extra_args as a
parameter to mark where the lemmatizer arguments start.
"""

import logging
import os

from stanza.models import identity_lemmatizer
from stanza.models import lemmatizer

from stanza.utils.training import common
from stanza.utils.training.common import Mode

from stanza.utils.datasets.prepare_lemma_treebank import check_lemmas

logger = logging.getLogger('stanza')

def run_treebank(mode, paths, treebank, short_name,
                 temp_output_file, command_args, extra_args):
    short_language = short_name.split("_")[0]

    lemma_dir      = paths["LEMMA_DATA_DIR"]
    train_file     = f"{lemma_dir}/{short_name}.train.in.conllu"
    dev_in_file    = f"{lemma_dir}/{short_name}.dev.in.conllu"
    dev_gold_file  = f"{lemma_dir}/{short_name}.dev.gold.conllu"
    dev_pred_file  = temp_output_file if temp_output_file else f"{lemma_dir}/{short_name}.dev.pred.conllu"
    test_in_file   = f"{lemma_dir}/{short_name}.test.in.conllu"
    test_gold_file = f"{lemma_dir}/{short_name}.test.gold.conllu"
    test_pred_file = temp_output_file if temp_output_file else f"{lemma_dir}/{short_name}.test.pred.conllu"

    if not os.path.exists(train_file):
        logger.error("Treebank %s is not prepared for training the lemmatizer.  Could not find any training data at %s  Skipping..." % (treebank, train_file))
        return

    has_lemmas = check_lemmas(train_file)
    if not has_lemmas:
        logger.info("Treebank " + treebank + " (" + short_name +
                    ") has no lemmas.  Using identity lemmatizer")
        if mode == Mode.TRAIN or mode == Mode.SCORE_DEV:
            train_args = ["--train_file", train_file,
                          "--eval_file", dev_in_file,
                          "--output_file", dev_pred_file,
                          "--gold_file", dev_gold_file,
                          "--lang", short_name]
            logger.info("Running identity lemmatizer for {} with args {}".format(treebank, train_args))
            identity_lemmatizer.main(train_args)
        elif mode == Mode.SCORE_TEST:
            train_args = ["--train_file", train_file,
                          "--eval_file", test_in_file,
                          "--output_file", test_pred_file,
                          "--gold_file", test_gold_file,
                          "--lang", short_name]
            logger.info("Running identity lemmatizer for {} with args {}".format(treebank, train_args))
            identity_lemmatizer.main(train_args)            
    else:
        if mode == Mode.TRAIN:
            # ('UD_Czech-PDT', 'UD_Russian-SynTagRus', 'UD_German-HDT')
            if short_name in ('cs_pdt', 'ru_syntagrus', 'de_hdt'):
                num_epochs = "30"
            else:
                num_epochs = "60"

            train_args = ["--train_file", train_file,
                          "--eval_file", dev_in_file,
                          "--output_file", dev_pred_file,
                          "--gold_file", dev_gold_file,
                          "--lang", short_name,
                          "--num_epoch", num_epochs,
                          "--mode", "train"]
            train_args = train_args + extra_args
            logger.info("Running train lemmatizer for {} with args {}".format(treebank, train_args))
            lemmatizer.main(train_args)

        if mode == Mode.SCORE_DEV or mode == Mode.TRAIN:
            dev_args = ["--eval_file", dev_in_file,
                        "--output_file", dev_pred_file,
                        "--gold_file", dev_gold_file,
                        "--lang", short_name,
                        "--mode", "predict"]
            dev_args = dev_args + extra_args
            logger.info("Running dev lemmatizer for {} with args {}".format(treebank, dev_args))
            lemmatizer.main(dev_args)

        if mode == Mode.SCORE_TEST:
            test_args = ["--eval_file", test_in_file,
                         "--output_file", test_pred_file,
                         "--gold_file", test_gold_file,
                         "--lang", short_name,
                         "--mode", "predict"]
            test_args = test_args + extra_args
            logger.info("Running test lemmatizer for {} with args {}".format(treebank, test_args))
            lemmatizer.main(test_args)

def main():
    common.main(run_treebank, "lemma", "lemmatizer")

if __name__ == "__main__":
    main()

