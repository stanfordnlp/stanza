"""
Trains or scores a constituency model.

Currently a suuuuper preliminary script.
"""

import logging
import os

from stanza.models import constituency_parser
from stanza.utils.datasets.constituency import prepare_con_dataset
from stanza.utils.training import common
from stanza.utils.training.common import Mode, build_charlm_args, choose_charlm, find_wordvec_pretrain

from stanza.resources.prepare_resources import default_charlms, default_pretrains

logger = logging.getLogger('stanza')

# xpos tagger doesn't produce PP tag on the turin treebank,
# so instead we use upos to avoid unknown tag errors
RETAG_METHOD = {
    "da": "upos",   # the DDT has no xpos tags anyway
    "es": "upos",   # AnCora has half-finished xpos tags
    "it": "upos",
    "vi": "upos",
    "pt": "upos",   # default PT model has no xpos either
}

def add_constituency_args(parser):
    parser.add_argument('--charlm', default="default", type=str, help='Which charlm to run on.  Will use the default charlm for this language/model if not set.  Set to None to turn off charlm for languages with a default charlm')
    parser.add_argument('--no_charlm', dest='charlm', action="store_const", const=None, help="Don't use a charlm, even if one is used by default for this package")

def run_treebank(mode, paths, treebank, short_name,
                 temp_output_file, command_args, extra_args):
    constituency_dir = paths["CONSTITUENCY_DATA_DIR"]
    language, dataset = short_name.split("_")

    train_file = os.path.join(constituency_dir, f"{short_name}_train.mrg")
    dev_file   = os.path.join(constituency_dir, f"{short_name}_dev.mrg")
    test_file  = os.path.join(constituency_dir, f"{short_name}_test.mrg")

    if not os.path.exists(train_file) or not os.path.exists(dev_file) or not os.path.exists(test_file):
        logger.warning(f"The data for {short_name} is missing or incomplete.  Attempting to rebuild...")
        try:
            prepare_con_dataset.main(short_name)
        except:
            logger.error(f"Unable to build the data.  Please correctly build the files in {train_file}, {dev_file}, {test_file} and then try again.")
            raise

    if language in RETAG_METHOD:
        retag_args = ["--retag_method", RETAG_METHOD[language]]
    else:
        retag_args = []

    if '--wordvec_pretrain_file' not in extra_args:
        # will throw an error if the pretrain can't be found
        wordvec_pretrain = find_wordvec_pretrain(language, default_pretrains)
        wordvec_args = ['--wordvec_pretrain_file', wordvec_pretrain]
    else:
        wordvec_args = []

    charlm = choose_charlm(language, dataset, command_args.charlm, default_charlms, {})
    charlm_args = build_charlm_args(language, charlm, base_args=False)

    default_args = retag_args + wordvec_args + charlm_args
    if language in common.BERT:
        default_args.extend(['--bert_model', common.BERT.get(language)])

    if mode == Mode.TRAIN:
        train_args = ['--train_file', train_file,
                      '--eval_file', dev_file,
                      '--shorthand', short_name,
                      '--mode', 'train']
        train_args = train_args + default_args + extra_args
        logger.info("Running train step with args: {}".format(train_args))
        constituency_parser.main(train_args)

    if mode == Mode.SCORE_DEV or mode == Mode.TRAIN:
        dev_args = ['--eval_file', dev_file,
                    '--shorthand', short_name,
                    '--mode', 'predict']
        dev_args = dev_args + default_args + extra_args
        logger.info("Running dev step with args: {}".format(dev_args))
        constituency_parser.main(dev_args)

    if mode == Mode.SCORE_TEST or mode == Mode.TRAIN:
        test_args = ['--eval_file', test_file,
                     '--shorthand', short_name,
                     '--mode', 'predict']
        test_args = test_args + default_args + extra_args
        logger.info("Running test step with args: {}".format(test_args))
        constituency_parser.main(test_args)



def main():
    common.main(run_treebank, "constituency", "constituency", add_constituency_args)

if __name__ == "__main__":
    main()

