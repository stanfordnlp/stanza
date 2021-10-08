"""
Trains or scores a constituency model.

Currently a suuuuper preliminary script.
"""

import logging
import os

from stanza.models import constituency_parser
from stanza.utils.datasets.constituency import prepare_con_dataset
from stanza.utils.training import common
from stanza.utils.training.common import Mode, find_wordvec_pretrain

logger = logging.getLogger('stanza')

# TODO: get this from the resources
RETAG_PACKAGE = {
    "en": "en_combined",
    "it": "it_combined",
    "vi": "vi_vtb",
}

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

    if mode == Mode.TRAIN:
        dataset_args = []
        if language in RETAG_PACKAGE:
            dataset_args = dataset_args + ["--retag_package", RETAG_PACKAGE[language]]

        train_args = ['--train_file', train_file,
                      '--eval_file', dev_file,
                      '--shorthand', short_name,
                      '--mode', 'train']
        train_args = train_args + dataset_args + extra_args
        if '--wordvec_pretrain_file' not in train_args:
            # will throw an error if the pretrain can't be found
            wordvec_pretrain = find_wordvec_pretrain(language)
            train_args = train_args + ['--wordvec_pretrain_file', wordvec_pretrain]
        logger.info("Running train step with args: {}".format(train_args))
        constituency_parser.main(train_args)

def main():
    common.main(run_treebank, "constituency", "constituency")

if __name__ == "__main__":
    main()

