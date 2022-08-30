import logging
import os

from stanza.models import parser

from stanza.utils.training import common
from stanza.utils.training.common import Mode
from stanza.utils.training.run_pos import wordvec_args

logger = logging.getLogger('stanza')

def run_treebank(mode, paths, treebank, short_name,
                 temp_output_file, command_args, extra_args):
    short_language, dataset = short_name.split("_")

    # TODO: refactor these blocks?
    depparse_dir   = paths["DEPPARSE_DATA_DIR"]
    train_file     = f"{depparse_dir}/{short_name}.train.in.conllu"
    dev_in_file    = f"{depparse_dir}/{short_name}.dev.in.conllu"
    dev_gold_file  = f"{depparse_dir}/{short_name}.dev.gold.conllu"
    dev_pred_file  = temp_output_file if temp_output_file else f"{depparse_dir}/{short_name}.dev.pred.conllu"
    test_in_file   = f"{depparse_dir}/{short_name}.test.in.conllu"
    test_gold_file = f"{depparse_dir}/{short_name}.test.gold.conllu"
    test_pred_file = temp_output_file if temp_output_file else f"{depparse_dir}/{short_name}.test.pred.conllu"

    if mode == Mode.TRAIN:
        if not os.path.exists(train_file):
            logger.error("TRAIN FILE NOT FOUND: %s ... skipping" % train_file)
            return

        # some languages need reduced batch size
        if short_name == 'de_hdt':
            # 'UD_German-HDT'
            batch_size = "1300"
        elif short_name in ('hr_set', 'fi_tdt', 'ru_taiga', 'cs_cltt', 'gl_treegal', 'lv_lvtb', 'ro_simonero'):
            # 'UD_Croatian-SET', 'UD_Finnish-TDT', 'UD_Russian-Taiga',
            # 'UD_Czech-CLTT', 'UD_Galician-TreeGal', 'UD_Latvian-LVTB' 'Romanian-SiMoNERo'
            batch_size = "3000"
        else:
            batch_size = "5000"

        train_args = ["--wordvec_dir", paths["WORDVEC_DIR"],
                      "--train_file", train_file,
                      "--eval_file", dev_in_file,
                      "--output_file", dev_pred_file,
                      "--gold_file", dev_gold_file,
                      "--batch_size", batch_size,
                      "--lang", short_language,
                      "--shorthand", short_name,
                      "--mode", "train"]
        train_args = train_args + wordvec_args(short_language, dataset, extra_args)
        train_args = train_args + extra_args
        logger.info("Running train depparse for {} with args {}".format(treebank, train_args))
        parser.main(train_args)

    if mode == Mode.SCORE_DEV or mode == Mode.TRAIN:
        dev_args = ["--wordvec_dir", paths["WORDVEC_DIR"],
                    "--eval_file", dev_in_file,
                    "--output_file", dev_pred_file,
                    "--gold_file", dev_gold_file,
                    "--lang", short_language,
                    "--shorthand", short_name,
                    "--mode", "predict"]
        dev_args = dev_args + wordvec_args(short_language, dataset, extra_args)
        dev_args = dev_args + extra_args
        logger.info("Running dev depparse for {} with args {}".format(treebank, dev_args))
        parser.main(dev_args)

        results = common.run_eval_script_depparse(dev_gold_file, dev_pred_file)
        logger.info("Finished running dev set on\n{}\n{}".format(treebank, results))

    if mode == Mode.SCORE_TEST:
        test_args = ["--wordvec_dir", paths["WORDVEC_DIR"],
                     "--eval_file", test_in_file,
                     "--output_file", test_pred_file,
                     "--gold_file", test_gold_file,
                     "--lang", short_language,
                     "--shorthand", short_name,
                     "--mode", "predict"]
        test_args = test_args + wordvec_args(short_language, dataset, extra_args)
        test_args = test_args + extra_args
        logger.info("Running test depparse for {} with args {}".format(treebank, test_args))
        parser.main(test_args)

        results = common.run_eval_script_depparse(test_gold_file, test_pred_file)
        logger.info("Finished running test set on\n{}\n{}".format(treebank, results))


def main():
    common.main(run_treebank, "depparse", "parser")

if __name__ == "__main__":
    main()

