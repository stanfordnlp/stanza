import argparse
import logging
import os
import pathlib
import subprocess
import sys

from enum import Enum

from stanza.models.common.constant import treebank_to_short_name
from stanza.utils.datasets import common
import stanza.utils.default_paths as default_paths

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
    return parser

def main(run_treebank, model_dir, model_name):
    paths = default_paths.get_default_paths()

    parser = build_argparse()
    if '--extra_args' in sys.argv:
        idx = sys.argv.index('--extra_args')
        extra_args = sys.argv[idx+1:]
        command_args = parser.parse_args(sys.argv[:idx])
    else:
        command_args, extra_args = parser.parse_known_args()

    mode = command_args.mode
    treebanks = []

    for treebank in command_args.treebanks:
        if treebank.lower() in ('ud_all', 'all_ud'):
            ud_treebank = common.get_ud_treebanks(paths["UDBASE"])

            for t in ud_treebank:
                short_name = treebank_to_short_name(t)
                model_path = "saved_models/%s/%s_%s.pt" % (model_dir, short_name, model_name)
                logger.debug("Looking for %s" % model_path)
                if mode == Mode.TRAIN and os.path.exists(model_path):
                    logger.info("%s: %s exists, skipping!" % (t, model_path))
                else:
                    treebanks.append(t)
        else:
            treebanks.append(treebank)

    for treebank in treebanks:
        short_name = treebank_to_short_name(treebank)
        logger.debug("%s: %s" % (treebank, short_name))
        run_treebank(mode, paths, treebank, short_name, command_args, extra_args)


def run_eval_script(eval_gold, eval_pred, start_row, end_row=None):
    # TODO: this is a silly way of doing this
    # would prefer to call it as a module
    # but the eval script expects sys args and prints the results to stdout
    if end_row is None:
        end_row = start_row + 1

    path = pathlib.Path(os.path.join(os.path.split(__file__)[0], ".."))
    path = path.resolve()

    eval_script = os.path.join(path, "conll18_ud_eval.py")
    results = subprocess.check_output([eval_script, "-v", eval_gold, eval_pred])
    results = results.decode(encoding="utf-8")
    results = [x.split("|")[3].strip() for x in results.split("\n")[start_row:end_row]]
    return " ".join(results)
