import logging
import os
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

