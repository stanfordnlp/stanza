"""
Trains or scores an NER model.

Will attempt to guess the appropriate word vector file if none is
specified, and will use the charlms specified in the resources
for a given dataset or language if possible.

Example command line:
  python3 -m stanza.utils.training.run_ner.py hu_combined

This script expects the prepared data to be in
  data/ner/{lang}_{dataset}.train.json, {lang}_{dataset}.dev.json, {lang}_{dataset}.test.json

If those files don't exist, it will make an attempt to rebuild them
using the prepare_ner_dataset script.  However, this will fail if the
data is not already downloaded.  More information on where to find
most of the datasets online is in that script.  Some of the datasets
have licenses which must be agreed to, so no attempt is made to
automatically download the data.
"""

import logging
import os

from stanza.models import ner_tagger
from stanza.resources.common import DEFAULT_MODEL_DIR
from stanza.utils.datasets.ner import prepare_ner_dataset
from stanza.utils.training import common
from stanza.utils.training.common import Mode, add_charlm_args, build_charlm_args, choose_charlm, find_wordvec_pretrain

from stanza.resources.prepare_resources import default_charlms, default_pretrains, ner_charlms, ner_pretrains

# extra arguments specific to a particular dataset
DATASET_EXTRA_ARGS = {
    "da_ddt":   [ "--dropout", "0.6" ],
    "fa_arman": [ "--dropout", "0.6" ],
    "vi_vlsp":  [ "--dropout", "0.6",
                  "--word_dropout", "0.1",
                  "--locked_dropout", "0.1",
                  "--char_dropout", "0.1" ],
}

logger = logging.getLogger('stanza')

def build_pretrain_args(language, dataset, charlm="default", extra_args=None, model_dir=DEFAULT_MODEL_DIR):
    """
    Returns one list with the args for this language & dataset's charlm and pretrained embedding
    """
    charlm = choose_charlm(language, dataset, charlm, default_charlms, ner_charlms)
    charlm_args = build_charlm_args(language, charlm, model_dir=model_dir)

    wordvec_args = []
    if extra_args is None or '--wordvec_pretrain_file' not in extra_args:
        # will throw an error if the pretrain can't be found
        wordvec_pretrain = find_wordvec_pretrain(language, default_pretrains, ner_pretrains, dataset, model_dir=model_dir)
        wordvec_args = ['--wordvec_pretrain_file', wordvec_pretrain]

    return charlm_args + wordvec_args

# Technically NER datasets are not necessarily treebanks
# (usually not, in fact)
# However, to keep the naming consistent, we leave the
# method which does the training as run_treebank
# TODO: rename treebank -> dataset everywhere
def run_treebank(mode, paths, treebank, short_name,
                 temp_output_file, command_args, extra_args):
    ner_dir = paths["NER_DATA_DIR"]
    language, dataset = short_name.split("_")

    train_file = os.path.join(ner_dir, f"{short_name}.train.json")
    dev_file   = os.path.join(ner_dir, f"{short_name}.dev.json")
    test_file  = os.path.join(ner_dir, f"{short_name}.test.json")

    missing_file = [x for x in (train_file, dev_file, test_file) if not os.path.exists(x)]
    if len(missing_file) > 0:
        logger.warning(f"The data for {short_name} is missing or incomplete.  Cannot find {missing_file}  Attempting to rebuild...")
        try:
            prepare_ner_dataset.main(short_name)
        except Exception as e:
            raise FileNotFoundError(f"An exception occurred while trying to build the data for {short_name}  At least one portion of the data was missing: {missing_file}  Please correctly build these files and then try again.") from e

    pretrain_args = build_pretrain_args(language, dataset, command_args.charlm, extra_args)

    if mode == Mode.TRAIN:
        # VI example arguments:
        #   --wordvec_pretrain_file ~/stanza_resources/vi/pretrain/vtb.pt
        #   --train_file data/ner/vi_vlsp.train.json
        #   --eval_file data/ner/vi_vlsp.dev.json
        #   --lang vi
        #   --shorthand vi_vlsp
        #   --mode train
        #   --charlm --charlm_shorthand vi_conll17
        #   --dropout 0.6 --word_dropout 0.1 --locked_dropout 0.1 --char_dropout 0.1
        dataset_args = DATASET_EXTRA_ARGS.get(short_name, [])
        if language in common.BERT:
            bert_args = ['--bert_model', common.BERT.get(language)]
        else:
            bert_args = []

        train_args = ['--train_file', train_file,
                      '--eval_file', dev_file,
                      '--lang', language,
                      '--shorthand', short_name,
                      '--mode', 'train']
        train_args = train_args + pretrain_args + bert_args + dataset_args + extra_args
        logger.info("Running train step with args: {}".format(train_args))
        ner_tagger.main(train_args)

    if mode == Mode.SCORE_DEV or mode == Mode.TRAIN:
        dev_args = ['--eval_file', dev_file,
                      '--lang', language,
                      '--shorthand', short_name,
                      '--mode', 'predict']
        dev_args = dev_args + pretrain_args + extra_args
        logger.info("Running dev step with args: {}".format(dev_args))
        ner_tagger.main(dev_args)

    if mode == Mode.SCORE_TEST or mode == Mode.TRAIN:
        test_args = ['--eval_file', test_file,
                      '--lang', language,
                      '--shorthand', short_name,
                      '--mode', 'predict']
        test_args = test_args + pretrain_args + extra_args
        logger.info("Running test step with args: {}".format(test_args))
        ner_tagger.main(test_args)


def main():
    common.main(run_treebank, "ner", "nertagger", add_charlm_args)

if __name__ == "__main__":
    main()

