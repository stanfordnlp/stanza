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

    parser.add_argument('--force', dest='force', action='store_true', default=False, help='Retrain existing models')
    return parser

SHORTNAME_RE = re.compile("[a-z-]+_[a-z0-9]+")

def main(run_treebank, model_dir, model_name, add_specific_args=None):
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

    for treebank in treebanks:
        if SHORTNAME_RE.match(treebank):
            short_name = treebank
        else:
            short_name = treebank_to_short_name(treebank)
        logger.debug("%s: %s" % (treebank, short_name))

        if mode == Mode.TRAIN and not command_args.force and model_name != 'ete':
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

def run_eval_script(eval_gold, eval_pred, start_row=None, end_row=None):
    # TODO: this is a silly way of doing this
    # would prefer to call it as a module
    # but the eval script expects sys args and prints the results to stdout
    if end_row is None and start_row is not None:
        end_row = start_row + 1

    path = pathlib.Path(os.path.join(os.path.split(__file__)[0], ".."))
    path = path.resolve()

    eval_script = os.path.join(path, "conll18_ud_eval.py")
    results = subprocess.check_output([eval_script, "-v", eval_gold, eval_pred])
    results = results.decode(encoding="utf-8")
    if start_row is None:
        return results
    else:
        results = [x.split("|")[3].strip() for x in results.split("\n")[start_row:end_row]]
        return " ".join(results)
