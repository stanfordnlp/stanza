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

import glob
import logging
import os

from stanza.models import ner_tagger
from stanza.utils.datasets.ner import prepare_ner_dataset
from stanza.utils.training import common
from stanza.utils.training.common import Mode

from stanza.resources.prepare_resources import default_charlms, ner_charlms
from stanza.resources.common import DEFAULT_MODEL_DIR

# extra arguments specific to a particular dataset
DATASET_EXTRA_ARGS = {
    "vi_vlsp": [ "--dropout", "0.6",
                 "--word_dropout", "0.1",
                 "--locked_dropout", "0.1",
                 "--char_dropout", "0.1" ],
}

logger = logging.getLogger('stanza')

def add_ner_args(parser):
    parser.add_argument('--charlm', default=None, type=str, help='Which charlm to run on.  Will use the default charlm for this language/model if not set.  Set to None to turn off charlm for languages with a default charlm')

def find_charlm(direction, language, charlm):
    saved_path = 'saved_models/charlm/{}_{}_{}_charlm.pt'.format(language, charlm, direction)
    if os.path.exists(saved_path):
        logger.info(f'Using model {saved_path} for {direction} charlm')
        return saved_path

    resource_path = '{}/{}/{}_charlm/{}.pt'.format(DEFAULT_MODEL_DIR, language, direction, charlm)
    if os.path.exists(resource_path):
        logger.info(f'Using model {resource_path} for {direction} charlm')
        return resource_path

    raise FileNotFoundError(f"Cannot find {direction} charlm in either {saved_path} or {resource_path}")

def find_wordvec_pretrain(language):
    # TODO: try to extract/remember the specific pretrain for the given model
    # That would be a good way to archive which pretrains are used for which NER models, anyway
    pretrain_path = '{}/{}/pretrain/*.pt'.format(DEFAULT_MODEL_DIR, language)
    pretrains = glob.glob(pretrain_path)
    if len(pretrains) == 0:
        raise FileNotFoundError(f"Cannot find any pretrains in {pretrain_path}  Try 'stanza.download(\"{language}\")' to get a default pretrain or use --wordvec_pretrain_path to specify a .pt file to use")
    if len(pretrains) > 1:
        raise FileNotFoundError(f"Too many pretrains to choose from in {pretrain_path}  Must specify an exact path to a --wordvec_pretrain_file")
    pretrain = pretrains[0]
    logger.info(f"Using pretrain found in {pretrain}  To use a different pretrain, specify --wordvec_pretrain_file")
    return pretrain

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

    if not os.path.exists(train_file) or not os.path.exists(dev_file) or not os.path.exists(test_file):
        logger.warning(f"The data for {short_name} is missing or incomplete.  Attempting to rebuild...")
        try:
            prepare_ner_dataset.main(short_name)
        except:
            logger.error(f"Unable to build the data.  Please correctly build the files in {train_file}, {dev_file}, {test_file} and then try again.")
            raise

    default_charlm = default_charlms.get(language, None)
    specific_charlm = ner_charlms.get(language, {}).get(dataset, None)
    if command_args.charlm:
        charlm = command_args.charlm
        if charlm == 'None':
            charlm = None
    elif specific_charlm:
        charlm = specific_charlm
    elif default_charlm:
        charlm = default_charlm
    else:
        charlm = None

    if charlm:
        forward = find_charlm('forward', language, charlm)
        backward = find_charlm('backward', language, charlm)
        charlm_args = ['--charlm',
                       '--charlm_shorthand', f'{language}_{charlm}',
                       '--charlm_forward_file', forward,
                       '--charlm_backward_file', backward]
    else:
        charlm_args = []

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

        train_args = ['--train_file', train_file,
                      '--eval_file', dev_file,
                      '--lang', language,
                      '--shorthand', short_name,
                      '--mode', 'train']
        train_args = train_args + charlm_args + dataset_args + extra_args
        if '--wordvec_pretrain_file' not in train_args:
            # will throw an error if the pretrain can't be found
            wordvec_pretrain = find_wordvec_pretrain(language)
            train_args = train_args + ['--wordvec_pretrain_file', wordvec_pretrain]
        logger.info("Running train step with args: {}".format(train_args))
        ner_tagger.main(train_args)

    if mode == Mode.SCORE_DEV or mode == Mode.TRAIN:
        dev_args = ['--eval_file', dev_file,
                      '--lang', language,
                      '--shorthand', short_name,
                      '--mode', 'predict']
        dev_args = dev_args + charlm_args + extra_args
        logger.info("Running dev step with args: {}".format(dev_args))
        ner_tagger.main(dev_args)

    if mode == Mode.SCORE_TEST or mode == Mode.TRAIN:
        test_args = ['--eval_file', test_file,
                      '--lang', language,
                      '--shorthand', short_name,
                      '--mode', 'predict']
        test_args = test_args + charlm_args + extra_args
        logger.info("Running test step with args: {}".format(test_args))
        ner_tagger.main(test_args)


def main():
    common.main(run_treebank, "ner", "nertagger", add_ner_args)

if __name__ == "__main__":
    main()

