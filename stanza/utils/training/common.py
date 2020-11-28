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


def main(run_treebank, model_dir, model_name):
    paths = default_paths.get_default_paths()
    args = sys.argv[1:]
    if args[0].startswith("--"):
        mode = Mode[args[0][2:].upper()]
        args = args[1:]
    else:
        mode = Mode.TRAIN
    treebank = args[0]
    extra_args = args[1:]

    if treebank.lower() in ('ud_all', 'all_ud'):
        treebanks = common.get_ud_treebanks(paths["UDBASE"])

        for t in treebanks:
            short_name = treebank_to_short_name(t)
            model_path = "saved_models/%s/%s_%s.pt" % (model_dir, short_name, model_name)
            logger.debug("Looking for %s" % model_path)
            if mode == Mode.TRAIN and os.path.exists(model_path):
                logger.info("%s: %s exists, skipping!" % (t, model_path))
                continue

            logger.debug("%s: %s" % (t, short_name))
            run_treebank(mode, paths, t, short_name, extra_args)
    else:
        short_name = treebank_to_short_name(treebank)
        run_treebank(mode, paths, treebank, short_name, extra_args)    


def run_eval_script(eval_gold, eval_pred, start_row, end_row=None):
    # TODO: this is a silly way of doing this
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
