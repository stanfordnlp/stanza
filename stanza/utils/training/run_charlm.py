"""
Trains or scores a charlm model.
"""

import logging
import os

from stanza.models import charlm
from stanza.utils.training import common
from stanza.utils.training.common import Mode

logger = logging.getLogger('stanza')


def add_charlm_args(parser):
    """
    Extra args for the charlm: forward/backward
    """
    parser.add_argument('--forward',  dest='forward', action='store_true',  default=False, help='Train a forward model')
    parser.add_argument('--backward', dest='forward', action='store_false', default=False, help='Train a backward model')


def run_treebank(mode, paths, treebank, short_name,
                 temp_output_file, command_args, extra_args):
    short_language, dataset_name = short_name.split("_", 1)

    train_dir = os.path.join(paths["CHARLM_DATA_DIR"], short_language, dataset_name, "train")

    dev_file  = os.path.join(paths["CHARLM_DATA_DIR"], short_language, dataset_name, "dev.txt")
    if not os.path.exists(dev_file) and os.path.exists(dev_file + ".xz"):
        dev_file = dev_file + ".xz"

    test_file = os.path.join(paths["CHARLM_DATA_DIR"], short_language, dataset_name, "test.txt")
    if not os.path.exists(test_file) and os.path.exists(test_file + ".xz"):
        test_file = test_file + ".xz"

    # python -m stanza.models.charlm --train_dir $train_dir --eval_file $dev_file \
    #     --direction $direction --lang $lang --shorthand $short --mode train $args
    # python -m stanza.models.charlm --eval_file $dev_file \
    #     --direction $direction --lang $lang --shorthand $short --mode predict $args
    # python -m stanza.models.charlm --eval_file $test_file \
    #     --direction $direction --lang $lang --shorthand $short --mode predict $args

    direction = "forward" if command_args.forward else "backward"
    default_args = ['--direction', direction,
                    '--lang', short_language,
                    '--shorthand', short_name]
    if mode == Mode.TRAIN:
        train_args = ['--train_dir', train_dir,
                      '--eval_file', dev_file,
                      '--mode', 'train']
        train_args = train_args + default_args + extra_args
        logger.info("Running train step with args: %s", train_args)
        charlm.main(train_args)

    if mode == Mode.SCORE_DEV:
        dev_args = ['--eval_file', dev_file,
                    '--mode', 'predict']
        dev_args = dev_args + default_args + extra_args
        logger.info("Running dev step with args: %s", dev_args)
        charlm.main(dev_args)

    if mode == Mode.SCORE_TEST:
        test_args = ['--eval_file', test_file,
                     '--mode', 'predict']
        test_args = test_args + default_args + extra_args
        logger.info("Running test step with args: %s", test_args)
        charlm.main(test_args)


def get_model_name(args):
    """
    The charlm saves forward and backward charlms to the same dir, but with different filenames
    """
    if args.forward:
        return "forward_charlm"
    else:
        return "backward_charlm"

def main():
    common.main(run_treebank, "charlm", get_model_name, add_charlm_args)

if __name__ == "__main__":
    main()

