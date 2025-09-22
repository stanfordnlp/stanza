
import io
import logging
import os

from stanza.models import tagger

from stanza.utils.training import common
from stanza.utils.training.common import Mode, add_charlm_args, build_pos_charlm_args, choose_pos_charlm, find_wordvec_pretrain, build_pos_wordvec_args

logger = logging.getLogger('stanza')

def add_pos_args(parser):
    add_charlm_args(parser)

    parser.add_argument('--use_bert', default=False, action="store_true", help='Use the default transformer for this language')

def build_model_filename(paths, short_name, command_args, extra_args):
    short_language, dataset = short_name.split("_", 1)

    # TODO: can avoid downloading the charlm at this point, since we
    # might not even be training
    charlm_args = build_pos_charlm_args(short_language, dataset, command_args.charlm)
    bert_args = common.choose_transformer(short_language, command_args, extra_args, warn=False)

    train_args = ["--shorthand", short_name,
                  "--mode", "train"]
    # TODO: also, this downloads the wordvec, which we might not want to do yet
    train_args = train_args + build_pos_wordvec_args(short_language, dataset, extra_args) + charlm_args + bert_args + extra_args
    if command_args.save_name is not None:
        train_args.extend(["--save_name", command_args.save_name])
    if command_args.save_dir is not None:
        train_args.extend(["--save_dir", command_args.save_dir])
    args = tagger.parse_args(train_args)
    save_name = tagger.model_file_name(args)
    return save_name



def run_treebank(mode, paths, treebank, short_name,
                 temp_output_file, command_args, extra_args):
    short_language, dataset = short_name.split("_", 1)

    pos_dir        = paths["POS_DATA_DIR"]
    train_file     = f"{pos_dir}/{short_name}.train.in.conllu"
    if short_name == 'vi_vlsp22':
        train_file += f";{pos_dir}/vi_vtb.train.in.conllu"
    dev_in_file    = f"{pos_dir}/{short_name}.dev.in.conllu"
    dev_pred_file  = f"{pos_dir}/{short_name}.dev.pred.conllu"
    test_in_file   = f"{pos_dir}/{short_name}.test.in.conllu"
    test_pred_file = f"{pos_dir}/{short_name}.test.pred.conllu"

    charlm_args = build_pos_charlm_args(short_language, dataset, command_args.charlm)
    bert_args = common.choose_transformer(short_language, command_args, extra_args)

    eval_file = None
    if '--eval_file' in extra_args:
        eval_file = extra_args[extra_args.index('--eval_file') + 1]

    if mode == Mode.TRAIN:
        train_pieces = []
        for train_piece in train_file.split(";"):
            zip_piece = os.path.splitext(train_piece)[0] + ".zip"
            if os.path.exists(train_piece) and os.path.exists(zip_piece):
                logger.error("POS TRAIN FILE %s and %s both exist... this is very confusing, skipping %s" % (train_piece, zip_piece, short_name))
                return
            if os.path.exists(train_piece):
                train_pieces.append(train_piece)
            else: # not os.path.exists(train_piece):
                if os.path.exists(zip_piece):
                    train_pieces.append(zip_piece)
                    continue
                logger.error("TRAIN FILE NOT FOUND: %s ... skipping" % train_piece)
                return
        train_file = ";".join(train_pieces)

        train_args = ["--wordvec_dir", paths["WORDVEC_DIR"],
                      "--train_file", train_file,
                      "--lang", short_language,
                      "--shorthand", short_name,
                      "--mode", "train"]
        if eval_file is None:
            train_args += ['--eval_file', dev_in_file]
        train_args = train_args + build_pos_wordvec_args(short_language, dataset, extra_args) + charlm_args + bert_args
        train_args = train_args + extra_args
        logger.info("Running train POS for {} with args {}".format(treebank, train_args))
        tagger.main(train_args)

    if mode == Mode.SCORE_DEV or mode == Mode.TRAIN:
        dev_args = ["--wordvec_dir", paths["WORDVEC_DIR"],
                    "--lang", short_language,
                    "--shorthand", short_name,
                    "--mode", "predict"]
        if eval_file is None:
            dev_args += ['--eval_file', dev_in_file]
        if command_args.save_output:
            dev_args.extend(["--output_file", dev_pred_file])
        dev_args = dev_args + build_pos_wordvec_args(short_language, dataset, extra_args) + charlm_args + bert_args
        dev_args = dev_args + extra_args
        logger.info("Running dev POS for {} with args {}".format(treebank, dev_args))
        _, dev_doc = tagger.main(dev_args)
        if not command_args.save_output:
            dev_pred_file = "{:C}\n\n".format(dev_doc)
            dev_pred_file = io.StringIO(dev_pred_file)

        results = common.run_eval_script_pos(eval_file if eval_file else dev_in_file, dev_pred_file)
        logger.info("Finished running dev set on\n{}\n{}".format(treebank, results))
        if command_args.save_output:
            logger.info("Output saved to %s", dev_pred_file)

    if mode == Mode.SCORE_TEST or mode == Mode.TRAIN:
        test_args = ["--wordvec_dir", paths["WORDVEC_DIR"],
                     "--lang", short_language,
                     "--shorthand", short_name,
                     "--mode", "predict"]
        if eval_file is None:
            test_args += ['--eval_file', test_in_file]
        if command_args.save_output:
            dev_args.extend(["--output_file", test_pred_file])
        test_args = test_args + build_pos_wordvec_args(short_language, dataset, extra_args) + charlm_args + bert_args
        test_args = test_args + extra_args
        logger.info("Running test POS for {} with args {}".format(treebank, test_args))
        _, test_doc = tagger.main(test_args)
        if not command_args.save_output:
            test_pred_file = "{:C}\n\n".format(test_doc)
            test_pred_file = io.StringIO(test_pred_file)

        results = common.run_eval_script_pos(eval_file if eval_file else test_in_file, test_pred_file)
        logger.info("Finished running test set on\n{}\n{}".format(treebank, results))
        if command_args.save_output:
            logger.info("Output saved to %s", test_pred_file)


def main():
    common.main(run_treebank, "pos", "tagger", add_pos_args, tagger.build_argparse(), build_model_filename=build_model_filename, choose_charlm_method=choose_pos_charlm)

if __name__ == "__main__":
    main()

