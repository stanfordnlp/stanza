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
from stanza.models.lemma import attach_lemma_classifier

from stanza.utils.training import common
from stanza.utils.training.common import Mode, add_charlm_args, build_lemma_charlm_args, choose_lemma_charlm
from stanza.utils.training import run_lemma_classifier

from stanza.utils.datasets.prepare_lemma_treebank import check_lemmas
import stanza.utils.datasets.prepare_lemma_classifier as prepare_lemma_classifier

logger = logging.getLogger('stanza')

def add_lemma_args(parser):
    add_charlm_args(parser)

    parser.add_argument('--lemma_classifier', dest='lemma_classifier', action='store_true', default=None,
                        help="Don't use the lemma classifier datasets.  Default is to build lemma classifier as part of the original lemmatizer if the charlm is used")
    parser.add_argument('--no_lemma_classifier', dest='lemma_classifier', action='store_false',
                        help="Don't use the lemma classifier datasets.  Default is to build lemma classifier as part of the original lemmatizer if the charlm is used")

def build_model_filename(paths, short_name, command_args, extra_args):
    """
    Figure out what the model savename will be, taking into account the model settings.

    Useful for figuring out if the model already exists

    None will represent that there is no expected save_name
    """
    short_language, dataset = short_name.split("_", 1)

    lemma_dir      = paths["LEMMA_DATA_DIR"]
    train_file     = f"{lemma_dir}/{short_name}.train.in.conllu"

    if not os.path.exists(train_file):
        logger.debug("Treebank %s is not prepared for training the lemmatizer.  Could not find any training data at %s  Cannot figure out the expected save_name without looking at the data, but a later step in the process will skip the training anyway" % (short_name, train_file))
        return None

    has_lemmas = check_lemmas(train_file)
    if not has_lemmas:
        return None

    # TODO: can avoid downloading the charlm at this point, since we
    # might not even be training
    charlm_args = build_lemma_charlm_args(short_language, dataset, command_args.charlm)

    train_args = ["--train_file", train_file,
                  "--shorthand", short_name,
                  "--mode", "train"]
    train_args = train_args + charlm_args + extra_args
    args = lemmatizer.parse_args(train_args)
    save_name = lemmatizer.build_model_filename(args)
    return save_name

def run_treebank(mode, paths, treebank, short_name,
                 temp_output_file, command_args, extra_args):
    short_language, dataset = short_name.split("_", 1)

    lemma_dir      = paths["LEMMA_DATA_DIR"]
    train_file     = f"{lemma_dir}/{short_name}.train.in.conllu"
    dev_in_file    = f"{lemma_dir}/{short_name}.dev.in.conllu"
    dev_gold_file  = f"{lemma_dir}/{short_name}.dev.gold.conllu"
    dev_pred_file  = temp_output_file if temp_output_file else f"{lemma_dir}/{short_name}.dev.pred.conllu"
    test_in_file   = f"{lemma_dir}/{short_name}.test.in.conllu"
    test_gold_file = f"{lemma_dir}/{short_name}.test.gold.conllu"
    test_pred_file = temp_output_file if temp_output_file else f"{lemma_dir}/{short_name}.test.pred.conllu"

    charlm_args = build_lemma_charlm_args(short_language, dataset, command_args.charlm)

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
                          "--shorthand", short_name]
            logger.info("Running identity lemmatizer for {} with args {}".format(treebank, train_args))
            identity_lemmatizer.main(train_args)
        elif mode == Mode.SCORE_TEST:
            train_args = ["--train_file", train_file,
                          "--eval_file", test_in_file,
                          "--output_file", test_pred_file,
                          "--gold_file", test_gold_file,
                          "--shorthand", short_name]
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
                          "--shorthand", short_name,
                          "--num_epoch", num_epochs,
                          "--mode", "train"]
            train_args = train_args + charlm_args + extra_args
            logger.info("Running train lemmatizer for {} with args {}".format(treebank, train_args))
            lemmatizer.main(train_args)

        if mode == Mode.SCORE_DEV or mode == Mode.TRAIN:
            dev_args = ["--eval_file", dev_in_file,
                        "--output_file", dev_pred_file,
                        "--gold_file", dev_gold_file,
                        "--shorthand", short_name,
                        "--mode", "predict"]
            dev_args = dev_args + charlm_args + extra_args
            logger.info("Running dev lemmatizer for {} with args {}".format(treebank, dev_args))
            lemmatizer.main(dev_args)

        if mode == Mode.SCORE_TEST:
            test_args = ["--eval_file", test_in_file,
                         "--output_file", test_pred_file,
                         "--gold_file", test_gold_file,
                         "--shorthand", short_name,
                         "--mode", "predict"]
            test_args = test_args + charlm_args + extra_args
            logger.info("Running test lemmatizer for {} with args {}".format(treebank, test_args))
            lemmatizer.main(test_args)

        use_lemma_classifier = command_args.lemma_classifier
        if use_lemma_classifier is None:
            use_lemma_classifier = command_args.charlm is not None
        use_lemma_classifier = use_lemma_classifier and short_name in prepare_lemma_classifier.DATASET_MAPPING
        if use_lemma_classifier and mode == Mode.TRAIN:
            lc_charlm_args = ['--no_charlm'] if command_args.charlm is None else ['--charlm', command_args.charlm]
            lemma_classifier_args = [treebank] + lc_charlm_args
            if command_args.force:
                lemma_classifier_args.append('--force')
            run_lemma_classifier.main(lemma_classifier_args)

            save_name = build_model_filename(paths, short_name, command_args, extra_args)
            # TODO: use a temp path for the lemma_classifier or keep it somewhere
            attach_args = ['--input', save_name,
                           '--output', save_name,
                           '--classifier', 'saved_models/lemma_classifier/%s_lemma_classifier.pt' % short_name]
            attach_lemma_classifier.main(attach_args)

            # now we rerun the dev set - the HI in particular demonstrates some good improvement
            lemmatizer.main(dev_args)

def main():
    common.main(run_treebank, "lemma", "lemmatizer", add_lemma_args, sub_argparse=lemmatizer.build_argparse(), build_model_filename=build_model_filename, choose_charlm_method=choose_lemma_charlm)

if __name__ == "__main__":
    main()

