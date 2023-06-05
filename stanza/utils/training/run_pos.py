

import logging
import os

from stanza.models import tagger

from stanza.resources.default_packages import no_pretrain_languages, default_charlms, pos_charlms, pos_pretrains, default_pretrains, specific_default_pretrains
from stanza.utils.training import common
from stanza.utils.training.common import Mode, add_charlm_args, build_charlm_args, choose_charlm, find_wordvec_pretrain

logger = logging.getLogger('stanza')

def add_pos_args(parser):
    add_charlm_args(parser)

    parser.add_argument('--use_bert', default=False, action="store_true", help='Use the default transformer for this language')

# TODO: move this somewhere common
def wordvec_args(short_language, dataset, extra_args):
    if '--wordvec_pretrain_file' in extra_args:
        return []

    if short_language in no_pretrain_languages:
        # we couldn't find word vectors for a few languages...:
        # coptic, naija, old russian, turkish german, swedish sign language
        logger.warning("No known word vectors for language {}  If those vectors can be found, please update the training scripts.".format(short_language))
        return ["--no_pretrain"]
    else:
        # for POS and depparse, there is a separate copy of the pretrain for each of the datasets
        # TODO: unify those into one pretrain
        if short_language in pos_pretrains and dataset in pos_pretrains[short_language]:
            dataset_pretrains = pos_pretrains
        elif short_language in specific_default_pretrains:
            dataset_pretrains = {}
        else:
            dataset_pretrains = {short_language: {dataset: dataset}}
        wordvec_pretrain = find_wordvec_pretrain(short_language, default_pretrains, dataset_pretrains, dataset)
        return ["--wordvec_pretrain_file", wordvec_pretrain]

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
    short_language, dataset = short_name.split("_", 1)

    pos_dir        = paths["POS_DATA_DIR"]
    train_file     = f"{pos_dir}/{short_name}.train.in.conllu"
    if short_name == 'vi_vlsp22':
        train_file += f";{pos_dir}/vi_vtb.train.in.conllu"
    dev_in_file    = f"{pos_dir}/{short_name}.dev.in.conllu"
    dev_pred_file  = temp_output_file if temp_output_file else f"{pos_dir}/{short_name}.dev.pred.conllu"
    test_in_file   = f"{pos_dir}/{short_name}.test.in.conllu"
    test_pred_file = temp_output_file if temp_output_file else f"{pos_dir}/{short_name}.test.pred.conllu"

    charlm = choose_charlm(short_language, dataset, command_args.charlm, default_charlms, pos_charlms)
    charlm_args = build_charlm_args(short_language, charlm)

    bert_args = []
    if command_args.use_bert and '--bert_model' not in extra_args:
        if short_language in common.BERT:
            bert_args = ['--bert_model', common.BERT.get(short_language)]
        else:
            logger.error("Transformer requested, but no default transformer for %s  Specify one using --bert_model" % short_language)

    eval_file = None
    if '--eval_file' in extra_args:
        eval_file = extra_args[extra_args.index('--eval_file') + 1]

    if mode == Mode.TRAIN:
        for train_piece in train_file.split(";"):
            if not os.path.exists(train_piece):
                logger.error("TRAIN FILE NOT FOUND: %s ... skipping" % train_piece)
                return

        # some languages need reduced batch size
        batch_size = pos_batch_size(short_name)

        train_args = ["--wordvec_dir", paths["WORDVEC_DIR"],
                      "--train_file", train_file,
                      "--output_file", dev_pred_file,
                      "--batch_size", batch_size,
                      "--lang", short_language,
                      "--shorthand", short_name,
                      "--mode", "train"]
        if eval_file is None:
            train_args += ['--eval_file', dev_in_file]
        train_args = train_args + wordvec_args(short_language, dataset, extra_args) + charlm_args + bert_args
        train_args = train_args + extra_args
        logger.info("Running train POS for {} with args {}".format(treebank, train_args))
        tagger.main(train_args)

    if mode == Mode.SCORE_DEV or mode == Mode.TRAIN:
        dev_args = ["--wordvec_dir", paths["WORDVEC_DIR"],
                    "--output_file", dev_pred_file,
                    "--lang", short_language,
                    "--shorthand", short_name,
                    "--mode", "predict"]
        if eval_file is None:
            dev_args += ['--eval_file', dev_in_file]
        dev_args = dev_args + wordvec_args(short_language, dataset, extra_args) + charlm_args + bert_args
        dev_args = dev_args + extra_args
        logger.info("Running dev POS for {} with args {}".format(treebank, dev_args))
        tagger.main(dev_args)

        results = common.run_eval_script_pos(eval_file if eval_file else dev_in_file, dev_pred_file)
        logger.info("Finished running dev set on\n{}\n{}".format(treebank, results))

    if mode == Mode.SCORE_TEST:
        test_args = ["--wordvec_dir", paths["WORDVEC_DIR"],
                     "--output_file", test_pred_file,
                     "--lang", short_language,
                     "--shorthand", short_name,
                     "--mode", "predict"]
        if eval_file is None:
            test_args += ['--eval_file', test_in_file]
        test_args = test_args + wordvec_args(short_language, dataset, extra_args) + charlm_args + bert_args
        test_args = test_args + extra_args
        logger.info("Running test POS for {} with args {}".format(treebank, test_args))
        tagger.main(test_args)

        results = common.run_eval_script_pos(eval_file if eval_file else test_in_file, test_pred_file)
        logger.info("Finished running test set on\n{}\n{}".format(treebank, results))


def main():
    common.main(run_treebank, "pos", "tagger", add_pos_args)

if __name__ == "__main__":
    main()

