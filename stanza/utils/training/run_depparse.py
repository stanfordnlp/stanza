import logging
import os

from stanza.models import parser

from stanza.utils.training import common
from stanza.utils.training.common import Mode, add_charlm_args, build_depparse_charlm_args, choose_depparse_charlm, choose_transformer
from stanza.utils.training.run_pos import wordvec_args

from stanza.resources.default_packages import default_charlms, depparse_charlms

logger = logging.getLogger('stanza')

def add_depparse_args(parser):
    add_charlm_args(parser)

    parser.add_argument('--use_bert', default=False, action="store_true", help='Use the default transformer for this language')

#  TODO: refactor with run_pos
def build_model_filename(paths, short_name, command_args, extra_args):
    short_language, dataset = short_name.split("_", 1)

    # TODO: can avoid downloading the charlm at this point, since we
    # might not even be training
    charlm_args = build_depparse_charlm_args(short_language, dataset, command_args.charlm)

    bert_args = choose_transformer(short_language, command_args, extra_args, warn=False)

    train_args = ["--shorthand", short_name,
                  "--mode", "train"]
    # TODO: also, this downloads the wordvec, which we might not want to do yet
    train_args = train_args + wordvec_args(short_language, dataset, extra_args) + charlm_args + bert_args + extra_args
    if command_args.save_name is not None:
        train_args.extend(["--save_name", command_args.save_name])
    if command_args.save_dir is not None:
        train_args.extend(["--save_dir", command_args.save_dir])
    args = parser.parse_args(train_args)
    save_name = parser.model_file_name(args)
    return save_name


def run_treebank(mode, paths, treebank, short_name,
                 temp_output_file, command_args, extra_args):
    short_language, dataset = short_name.split("_")

    # TODO: refactor these blocks?
    depparse_dir   = paths["DEPPARSE_DATA_DIR"]
    train_file     = f"{depparse_dir}/{short_name}.train.in.conllu"
    dev_in_file    = f"{depparse_dir}/{short_name}.dev.in.conllu"
    dev_pred_file  = temp_output_file if temp_output_file else f"{depparse_dir}/{short_name}.dev.pred.conllu"
    test_in_file   = f"{depparse_dir}/{short_name}.test.in.conllu"
    test_pred_file = temp_output_file if temp_output_file else f"{depparse_dir}/{short_name}.test.pred.conllu"

    eval_file = None
    if '--eval_file' in extra_args:
        eval_file = extra_args[extra_args.index('--eval_file') + 1]

    charlm_args = build_depparse_charlm_args(short_language, dataset, command_args.charlm)

    bert_args = choose_transformer(short_language, command_args, extra_args)

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
                      "--eval_file", eval_file if eval_file else dev_in_file,
                      "--output_file", dev_pred_file,
                      "--batch_size", batch_size,
                      "--lang", short_language,
                      "--shorthand", short_name,
                      "--mode", "train"]
        train_args = train_args + wordvec_args(short_language, dataset, extra_args) + charlm_args + bert_args
        train_args = train_args + extra_args
        logger.info("Running train depparse for {} with args {}".format(treebank, train_args))
        parser.main(train_args)

    if mode == Mode.SCORE_DEV or mode == Mode.TRAIN:
        dev_args = ["--wordvec_dir", paths["WORDVEC_DIR"],
                    "--eval_file", eval_file if eval_file else dev_in_file,
                    "--output_file", dev_pred_file,
                    "--lang", short_language,
                    "--shorthand", short_name,
                    "--mode", "predict"]
        dev_args = dev_args + wordvec_args(short_language, dataset, extra_args) + charlm_args + bert_args
        dev_args = dev_args + extra_args
        logger.info("Running dev depparse for {} with args {}".format(treebank, dev_args))
        parser.main(dev_args)

        if '--no_gold_labels' not in extra_args:
            results = common.run_eval_script_depparse(eval_file if eval_file else dev_in_file, dev_pred_file)
            logger.info("Finished running dev set on\n{}\n{}".format(treebank, results))
        if not temp_output_file:
            logger.info("Output saved to %s", dev_pred_file)

    if mode == Mode.SCORE_TEST:
        test_args = ["--wordvec_dir", paths["WORDVEC_DIR"],
                     "--eval_file", eval_file if eval_file else test_in_file,
                     "--output_file", test_pred_file,
                     "--lang", short_language,
                     "--shorthand", short_name,
                     "--mode", "predict"]
        test_args = test_args + wordvec_args(short_language, dataset, extra_args) + charlm_args + bert_args
        test_args = test_args + extra_args
        logger.info("Running test depparse for {} with args {}".format(treebank, test_args))
        parser.main(test_args)

        if '--no_gold_labels' not in extra_args:
            results = common.run_eval_script_depparse(eval_file if eval_file else test_in_file, test_pred_file)
            logger.info("Finished running test set on\n{}\n{}".format(treebank, results))
        if not temp_output_file:
            logger.info("Output saved to %s", test_pred_file)


def main():
    common.main(run_treebank, "depparse", "parser", add_depparse_args, sub_argparse=parser.build_argparse(), build_model_filename=build_model_filename, choose_charlm_method=choose_depparse_charlm)

if __name__ == "__main__":
    main()

