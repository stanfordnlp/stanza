

import logging
import os

from stanza.models import tagger

from stanza.utils.training import common
from stanza.utils.training.common import Mode

logger = logging.getLogger('stanza')

# TODO: move this somewhere common
def wordvec_args(short_language):
    if short_language in ("cop", "orv", "pcm", "qtd", "swl"):
        # we couldn't find word vectors for these languages:
        # coptic, naija, old russian, turkish german, swedish sign language
        logger.warning("No known word vectors for language {}  If those vectors can be found, please update the training scripts.".format(short_language))
        return ["--no_pretrain"]
    else:
        return []

def pos_batch_size(short_name):
    if short_name == 'de_hdt':
        # 'UD_German-HDT'
        return "2000"
    elif short_name == 'hr_set':
        # 'UD_Croatian-SET'
        return "3000"
    else:
        return "5000"

def run_treebank(mode, paths, treebank, short_name,
                 temp_output_file, command_args, extra_args):
    short_language = short_name.split("_")[0]

    pos_dir        = paths["POS_DATA_DIR"]
    train_file     = f"{pos_dir}/{short_name}.train.in.conllu"
    dev_in_file    = f"{pos_dir}/{short_name}.dev.in.conllu"
    dev_gold_file  = f"{pos_dir}/{short_name}.dev.gold.conllu"
    dev_pred_file  = temp_output_file if temp_output_file else f"{pos_dir}/{short_name}.dev.pred.conllu"
    test_in_file   = f"{pos_dir}/{short_name}.test.in.conllu"
    test_gold_file = f"{pos_dir}/{short_name}.test.gold.conllu"
    test_pred_file = temp_output_file if temp_output_file else f"{pos_dir}/{short_name}.test.pred.conllu"

    if mode == Mode.TRAIN:
        if not os.path.exists(train_file):
            logger.error("TRAIN FILE NOT FOUND: %s ... skipping" % train_file)
            return

        # some languages need reduced batch size
        batch_size = pos_batch_size(short_name)

        train_args = ["--wordvec_dir", paths["WORDVEC_DIR"],
                      "--train_file", train_file,
                      "--eval_file", dev_in_file,
                      "--output_file", dev_pred_file,
                      "--gold_file", dev_gold_file,
                      "--batch_size", batch_size,
                      "--lang", short_language,
                      "--shorthand", short_name,
                      "--mode", "train"]
        train_args = train_args + wordvec_args(short_language)
        train_args = train_args + extra_args
        logger.info("Running train POS for {} with args {}".format(treebank, train_args))
        tagger.main(train_args)

    if mode == Mode.SCORE_DEV or mode == Mode.TRAIN:
        dev_args = ["--wordvec_dir", paths["WORDVEC_DIR"],
                    "--eval_file", dev_in_file,
                    "--output_file", dev_pred_file,
                    "--gold_file", dev_gold_file,
                    "--lang", short_language,
                    "--shorthand", short_name,
                    "--mode", "predict"]
        dev_args = dev_args + wordvec_args(short_language)
        dev_args = dev_args + extra_args
        logger.info("Running dev POS for {} with args {}".format(treebank, dev_args))
        tagger.main(dev_args)

        results = common.run_eval_script_pos(dev_gold_file, dev_pred_file)
        logger.info("Finished running dev set on\n{}\n{}".format(treebank, results))

    if mode == Mode.SCORE_TEST:
        test_args = ["--wordvec_dir", paths["WORDVEC_DIR"],
                    "--eval_file", test_in_file,
                    "--output_file", test_pred_file,
                    "--gold_file", test_gold_file,
                    "--lang", short_language,
                    "--shorthand", short_name,
                    "--mode", "predict"]
        test_args = test_args + wordvec_args(short_language)
        test_args = test_args + extra_args
        logger.info("Running test POS for {} with args {}".format(treebank, test_args))
        tagger.main(test_args)

        results = common.run_eval_script_pos(test_gold_file, test_pred_file)
        logger.info("Finished running test set on\n{}\n{}".format(treebank, results))


def main():
    common.main(run_treebank, "pos", "tagger")

if __name__ == "__main__":
    main()

