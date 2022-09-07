"""
A script to prepare all depparse datasets.

Prepares each of train, dev, test.

Example:
    python -m stanza.utils.datasets.prepare_depparse_treebank {TREEBANK}
Example:
    python -m stanza.utils.datasets.prepare_depparse_treebank UD_English-EWT
"""

from enum import Enum
import logging
import os

from stanza.models import tagger
from stanza.models.common.constant import treebank_to_short_name
from stanza.resources.common import download, DEFAULT_MODEL_DIR
from stanza.resources.prepare_resources import default_charlms, pos_charlms
import stanza.utils.datasets.common as common
import stanza.utils.datasets.prepare_tokenizer_treebank as prepare_tokenizer_treebank
from stanza.utils.training.run_pos import pos_batch_size, wordvec_args
from stanza.utils.training.common import add_charlm_args, build_charlm_args, choose_charlm

logger = logging.getLogger('stanza')


class Tags(Enum):
    """Tags parameter values."""

    GOLD = 1
    PREDICTED = 2


# fmt: off
def add_specific_args(parser) -> None:
    """Add specific args."""
    parser.add_argument("--gold", dest='tag_method', action='store_const', const=Tags.GOLD, default=Tags.PREDICTED,
                        help='Use gold tags for building the depparse data')
    parser.add_argument("--predicted", dest='tag_method', action='store_const', const=Tags.PREDICTED,
                        help='Use predicted tags for building the depparse data')
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None,
                        help='Exact name of the pretrain file to read')
    parser.add_argument('--tagger_model', type=str, default=None,
                        help='Tagger save file to use.  If not specified, order searched will be saved/models, then $STANZA_RESOURCES_DIR')
    add_charlm_args(parser)
# fmt: on

def choose_tagger_model(short_language, dataset, tagger_model):
    """
    Preferentially chooses a retrained tagger model, but tries to download one if that doesn't exist
    """
    if tagger_model:
        return tagger_model

    save_path = os.path.join("saved_models", "pos", "%s_%s_tagger.pt" % (short_language, dataset))
    if os.path.exists(save_path):
        return save_path

    # TODO: just create a Pipeline for the retagging instead?
    pos_path = os.path.join(DEFAULT_MODEL_DIR, short_language, "pos", dataset + ".pt")
    download(lang=short_language, package=None, processors={"pos": dataset})
    return pos_path


def process_treebank(treebank, paths, args) -> None:
    """Process treebank."""
    if args.tag_method is Tags.GOLD:
        prepare_tokenizer_treebank.copy_conllu_treebank(treebank, paths, paths["DEPPARSE_DATA_DIR"])
    elif args.tag_method is Tags.PREDICTED:
        short_name = treebank_to_short_name(treebank)
        short_language, dataset = short_name.split("_")

        # fmt: off
        base_args = ["--wordvec_dir", paths["WORDVEC_DIR"],
                     "--lang", short_language,
                     "--shorthand", short_name,
                     "--batch_size", pos_batch_size(short_name),
                     "--mode", "predict"]
        # fmt: on

        # perhaps download a tagger if one doesn't already exist
        tagger_model = choose_tagger_model(short_language, dataset, args.tagger_model)
        tagger_dir, tagger_name = os.path.split(tagger_model)
        base_args = base_args + ['--save_dir', tagger_dir, '--save_name', tagger_name]

        # word vector file for POS
        base_args = base_args + wordvec_args(short_language, dataset, args)

        # charlm for POS
        charlm = choose_charlm(short_language, dataset, args.charlm, default_charlms, pos_charlms)
        charlm_args = build_charlm_args(short_language, charlm)
        base_args = base_args + charlm_args

        def retag_dataset(tokenizer_dir, tokenizer_file, dest_dir, dest_file, short_name):
            original = f"{tokenizer_dir}/{short_name}.{tokenizer_file}.conllu"
            retagged = f"{dest_dir}/{short_name}.{dest_file}.conllu"
            # fmt: off
            tagger_args = ["--eval_file", original,
                           "--gold_file", original,
                           "--output_file", retagged]
            # fmt: on
            tagger_args = base_args + tagger_args
            logger.info("Running tagger to retag {} to {}\n  Args: {}".format(original, retagged, tagger_args))
            tagger.main(tagger_args)

        prepare_tokenizer_treebank.copy_conllu_treebank(treebank, paths, paths["DEPPARSE_DATA_DIR"], retag_dataset)
    else:
        raise ValueError("Unknown tags method: {}".format(args.tag_method))


def main() -> None:
    """Call Process Treebank."""
    common.main(process_treebank, add_specific_args)


if __name__ == '__main__':
    main()
