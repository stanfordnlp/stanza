import argparse
import logging
import os
import pathlib
import re
import subprocess
import sys
import tempfile

from enum import Enum

from stanza.models.common.constant import treebank_to_short_name
from stanza.utils.datasets import common
import stanza.utils.default_paths as default_paths
from stanza.utils import conll18_ud_eval as ud_eval

logger = logging.getLogger('stanza')

class Mode(Enum):
    TRAIN = 1
    SCORE_DEV = 2
    SCORE_TEST = 3

def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_output', dest='temp_output', default=True, action='store_false', help="Save output - default is to use a temp directory.")

    parser.add_argument('treebanks', type=str, nargs='+', help='Which treebanks to run on.  Use all_ud or ud_all for all UD treebanks')

    parser.add_argument('--train', dest='mode', default=Mode.TRAIN, action='store_const', const=Mode.TRAIN, help='Run in train mode')
    parser.add_argument('--score_dev', dest='mode', action='store_const', const=Mode.SCORE_DEV, help='Score the dev set')
    parser.add_argument('--score_test', dest='mode', action='store_const', const=Mode.SCORE_TEST, help='Score the test set')

    # This argument needs to be here so we can identify if the model already exists in the user-specified home
    parser.add_argument('--save_dir', type=str, default=None, help="Root dir for saving models.  If set, will override the model's default.")

    parser.add_argument('--force', dest='force', action='store_true', default=False, help='Retrain existing models')
    return parser

SHORTNAME_RE = re.compile("[a-z-]+_[a-z0-9]+")

def main(run_treebank, model_dir, model_name, add_specific_args=None):
    """
    A main program for each of the run_xyz scripts

    It collects the arguments and runs the main method for each dataset provided.
    It also tries to look for an existing model and not overwrite it unless --force is provided
    """
    logger.info("Training program called with:\n" + " ".join(sys.argv))

    paths = default_paths.get_default_paths()

    parser = build_argparse()
    if add_specific_args is not None:
        add_specific_args(parser)
    if '--extra_args' in sys.argv:
        idx = sys.argv.index('--extra_args')
        extra_args = sys.argv[idx+1:]
        command_args = parser.parse_args(sys.argv[1:idx])
    else:
        command_args, extra_args = parser.parse_known_args()

    # Pass this through to the underlying model as well as use it here
    if command_args.save_dir:
        extra_args.extend(["--save_dir", command_args.save_dir])

    mode = command_args.mode
    treebanks = []

    for treebank in command_args.treebanks:
        # this is a really annoying typo to make if you copy/paste a
        # UD directory name on the cluster and your job dies 30s after
        # being queued for an hour
        if treebank.endswith("/"):
            treebank = treebank[:-1]
        if treebank.lower() in ('ud_all', 'all_ud'):
            ud_treebanks = common.get_ud_treebanks(paths["UDBASE"])
            treebanks.extend(ud_treebanks)
        else:
            treebanks.append(treebank)

    for treebank_idx, treebank in enumerate(treebanks):
        if treebank_idx > 0:
            logger.info("=========================================")

        if SHORTNAME_RE.match(treebank):
            short_name = treebank
        else:
            short_name = treebank_to_short_name(treebank)
        logger.debug("%s: %s" % (treebank, short_name))

        if mode == Mode.TRAIN and not command_args.force and model_name != 'ete':
            if command_args.save_dir:
                model_path = "%s/%s_%s.pt" % (command_args.save_dir, short_name, model_name)
            else:
                model_path = "saved_models/%s/%s_%s.pt" % (model_dir, short_name, model_name)
            if os.path.exists(model_path):
                logger.info("%s: %s exists, skipping!" % (treebank, model_path))
                continue
            else:
                logger.info("%s: %s does not exist, training new model" % (treebank, model_path))

        if command_args.temp_output and model_name != 'ete':
            with tempfile.NamedTemporaryFile() as temp_output_file:
                run_treebank(mode, paths, treebank, short_name,
                             temp_output_file.name, command_args, extra_args)
        else:
            run_treebank(mode, paths, treebank, short_name,
                         None, command_args, extra_args)

def run_eval_script(gold_conllu_file, system_conllu_file, evals=None):
    """ Wrapper for lemma scorer. """
    gold_ud = ud_eval.load_conllu_file(gold_conllu_file)
    system_ud = ud_eval.load_conllu_file(system_conllu_file)
    evaluation = ud_eval.evaluate(gold_ud, system_ud)

    if evals is None:
        return ud_eval.build_evaluation_table(evaluation, verbose=True, counts=False)
    else:
        results = [evaluation[key].f1 for key in evals]
        return " ".join("{:.2f}".format(100 * x) for x in results)

def run_eval_script_tokens(eval_gold, eval_pred):
    return run_eval_script(eval_gold, eval_pred, evals=["Tokens", "Sentences", "Words"])

def run_eval_script_mwt(eval_gold, eval_pred):
    return run_eval_script(eval_gold, eval_pred, evals=["Words"])

def run_eval_script_pos(eval_gold, eval_pred):
    return run_eval_script(eval_gold, eval_pred, evals=["UPOS", "XPOS", "UFeats", "AllTags"])

def run_eval_script_depparse(eval_gold, eval_pred):
    return run_eval_script(eval_gold, eval_pred, evals=["UAS", "LAS", "CLAS", "MLAS", "BLEX"])
