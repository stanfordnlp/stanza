"""
A script to prepare all depparse datasets.

Prepares each of train, dev, test.

Example:
    python -m stanza.utils.datasets.prepare_depparse_treebank {TREEBANK}
Example:
    python -m stanza.utils.datasets.prepare_depparse_treebank UD_English-EWT
"""

import stanza.utils.datasets.common as common
import stanza.utils.datasets.prepare_tokenizer_treebank as prepare_tokenizer_treebank
from stanza.utils.datasets.dataset_retagging import add_retagging_args, build_postprocess


def add_specific_args(parser) -> None:
    """Add specific args."""
    add_retagging_args(parser)


def process_treebank(treebank, model_type, paths, args) -> None:
    """Process treebank."""
    postprocess = build_postprocess(treebank, paths, args)
    prepare_tokenizer_treebank.copy_conllu_treebank(treebank, model_type, paths, paths["DEPPARSE_DATA_DIR"], args=args, postprocess=postprocess)
    return True

def main() -> None:
    """Call Process Treebank."""
    common.main(process_treebank, common.ModelType.DEPPARSE, add_specific_args)


if __name__ == '__main__':
    main()
