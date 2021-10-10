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

import stanza.utils.datasets.common as common
import stanza.utils.datasets.prepare_tokenizer_treebank as prepare_tokenizer_treebank
from stanza.models import tagger
from stanza.utils.training.run_pos import pos_batch_size, wordvec_args

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
# fmt: on


def process_treebank(treebank, paths, args) -> None:
    """Process treebank."""
    if args.tag_method is Tags.GOLD:
        prepare_tokenizer_treebank.copy_conllu_treebank(treebank, paths, paths["DEPPARSE_DATA_DIR"])
    elif args.tag_method is Tags.PREDICTED:
        short_name = common.project_to_short_name(treebank)
        short_language = short_name.split("_")[0]

        # fmt: off
        base_args = ["--wordvec_dir", paths["WORDVEC_DIR"],
                     "--lang", short_language,
                     "--shorthand", short_name,
                     "--batch_size", pos_batch_size(short_name),
                     "--mode", "predict"]
        # fmt: on
        base_args = base_args + wordvec_args(short_language)

        def retag_dataset(tokenizer_dir, tokenizer_file, dest_dir, dest_file, short_name):
            original = f"{tokenizer_dir}/{short_name}.{tokenizer_file}.conllu"
            retagged = f"{dest_dir}/{short_name}.{dest_file}.conllu"
            # fmt: off
            tagger_args = ["--eval_file", original,
                           "--gold_file", original,
                           "--output_file", retagged]
            # fmt: on
            if args.wordvec_pretrain_file:
                tagger_args.extend(["--wordvec_pretrain_file", args.wordvec_pretrain_file])
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
