"""Saved a huge, bloated model with an optimizer?  Use this to remove it, greatly shrinking the model size

This tries to find reasonable defaults for word vectors and charlm
(which need to be loaded so that the model knows the matrix sizes)

so ideally all that needs to be run is

python3 stanza/utils/training/remove_constituency_optimizer.py <treebanks>
python3 stanza/utils/training/remove_constituency_optimizer.py da_arboretum ...

This can also be used to load and save models as part of an update
to the serialized format
"""

import argparse
import logging
import os

from stanza.models import constituency_parser
from stanza.models.common.constant import treebank_to_short_name
from stanza.resources.prepare_resources import default_charlms, default_pretrains
from stanza.utils.training import common

logger = logging.getLogger('stanza')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')
    parser.add_argument('--charlm', default="default", type=str, help='Which charlm to run on.  Will use the default charlm for this language/model if not set.  Set to None to turn off charlm for languages with a default charlm')
    parser.add_argument('--no_charlm', dest='charlm', action="store_const", const=None, help="Don't use a charlm, even if one is used by default for this package")

    parser.add_argument('--load_dir', type=str, default="saved_models/constituency", help="Root dir for getting the models to resave.")
    parser.add_argument('--save_dir', type=str, default="resaved_models/constituency", help="Root dir for resaving the models.")

    parser.add_argument('treebanks', type=str, nargs='+', help='Which treebanks to run on.  Use all_ud or ud_all for all UD treebanks')

    args = parser.parse_args()
    return args

def main():
    """
    For each of the models specified, load and resave the model

    The resaved model will have the optimizer removed
    """
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    for treebank in args.treebanks:
        logger.info("PROCESSING %s", treebank)
        short_name = treebank_to_short_name(treebank)
        language, dataset = short_name.split("_", maxsplit=1)
        logger.info("%s: %s %s", short_name, language, dataset)

        if not args.wordvec_pretrain_file:
            # will throw an error if the pretrain can't be found
            wordvec_pretrain = common.find_wordvec_pretrain(language, default_pretrains)
            wordvec_args = ['--wordvec_pretrain_file', wordvec_pretrain]
        else:
            wordvec_args = []

        charlm = common.choose_charlm(language, dataset, args.charlm, default_charlms, {})
        charlm_args = common.build_charlm_args(language, charlm, base_args=False)

        base_name = '{}_constituency.pt'.format(short_name)
        load_name = os.path.join(args.load_dir, base_name)
        save_name = os.path.join(args.save_dir, base_name)
        resave_args = ['--mode', 'remove_optimizer',
                       '--load_name', load_name,
                       '--save_name', save_name,
                       '--save_dir', ".",
                       '--shorthand', short_name]
        resave_args = resave_args + wordvec_args + charlm_args
        constituency_parser.main(resave_args)

if __name__ == '__main__':
    main()
