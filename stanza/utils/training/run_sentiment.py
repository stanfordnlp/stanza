"""
Trains or tests a sentiment model using the classifier package

The prep script has separate entries for the root-only version of SST,
which is what people typically use to test.  When training a model for
SST which uses all the data, the root-only version is used for
dev and test
"""

import logging
import os

from stanza.models import classifier
from stanza.utils.training import common
from stanza.utils.training.common import Mode, build_charlm_args, choose_charlm, find_wordvec_pretrain

from stanza.resources.prepare_resources import default_charlms, default_pretrains

logger = logging.getLogger('stanza')

# TODO: refactor with ner & conparse
def add_sentiment_args(parser):
    parser.add_argument('--charlm', default="default", type=str, help='Which charlm to run on.  Will use the default charlm for this language/model if not set.  Set to None to turn off charlm for languages with a default charlm')
    parser.add_argument('--no_charlm', dest='charlm', action="store_const", const=None, help="Don't use a charlm, even if one is used by default for this package")


ALTERNATE_DATASET = {
    "en_sst2":    "en_sst2roots",
    "en_sstplus": "en_sst3roots",
}

def run_dataset(mode, paths, treebank, short_name,
                temp_output_file, command_args, extra_args):
    sentiment_dir = paths["SENTIMENT_DATA_DIR"]
    language, dataset = short_name.split("_")

    train_file = os.path.join(sentiment_dir, f"{short_name}.train.json")

    other_name = ALTERNATE_DATASET.get(short_name, short_name)
    dev_file   = os.path.join(sentiment_dir, f"{other_name}.dev.json")
    test_file  = os.path.join(sentiment_dir, f"{other_name}.test.json")

    for filename in (train_file, dev_file, test_file):
        if not os.path.exists(filename):
            raise FileNotFoundError("Cannot find %s" % filename)

    if '--wordvec_pretrain_file' not in extra_args:
        # will throw an error if the pretrain can't be found
        wordvec_pretrain = find_wordvec_pretrain(language, default_pretrains)
        wordvec_args = ['--wordvec_pretrain_file', wordvec_pretrain]
    else:
        wordvec_args = []

    charlm = choose_charlm(language, dataset, command_args.charlm, default_charlms, {})
    charlm_args = build_charlm_args(language, charlm, base_args=False)

    default_args = wordvec_args + charlm_args
        
    if mode == Mode.TRAIN:
        train_args = ['--save_name', "%s_classifier.pt" % short_name,
                      '--train_file', train_file,
                      '--dev_file', dev_file,
                      '--test_file', test_file,
                      '--shorthand', short_name,
                      '--wordvec_type', 'word2vec',   # TODO: chinese is fasttext
                      '--extra_wordvec_method', 'SUM']
        train_args = train_args + default_args + extra_args
        logger.info("Running train step with args: {}".format(train_args))
        classifier.main(train_args)

    if mode == Mode.SCORE_DEV or mode == Mode.TRAIN:
        dev_args = ['--save_name', "%s_classifier.pt" % short_name,
                    '--no_train',
                    '--test_file', dev_file,
                    '--shorthand', short_name,
                    '--wordvec_type', 'word2vec']   # TODO: chinese is fasttext
        dev_args = dev_args + default_args + extra_args
        logger.info("Running dev step with args: {}".format(dev_args))
        classifier.main(dev_args)

    if mode == Mode.SCORE_TEST or mode == Mode.TRAIN:
        test_args = ['--save_name', "%s_classifier.pt" % short_name,
                     '--no_train',
                     '--test_file', test_file,
                     '--shorthand', short_name,
                     '--wordvec_type', 'word2vec']   # TODO: chinese is fasttext
        test_args = test_args + default_args + extra_args
        logger.info("Running test step with args: {}".format(test_args))
        classifier.main(test_args)



def main():
    common.main(run_dataset, "classifier", "classifier", add_sentiment_args)

if __name__ == "__main__":
    main()

