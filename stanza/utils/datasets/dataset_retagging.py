from enum import Enum
import glob
import os
import logging

logger = logging.getLogger('stanza')

from stanza.models import tagger
from stanza.models.common.constant import treebank_to_short_name
from stanza.resources.common import download, DEFAULT_MODEL_DIR, UnknownLanguageError
from stanza.resources.default_packages import default_charlms, pos_charlms
from stanza.resources.default_packages import TRANSFORMERS, TRANSFORMER_NICKNAMES
from stanza.utils.datasets.prepare_tokenizer_treebank import copy_conllu_file
from stanza.utils.training.common import build_pos_wordvec_args
from stanza.utils.training.common import add_charlm_args, build_charlm_args, choose_charlm

class Tags(Enum):
    """
    Which splits of the dataset get predicted tags

    GOLD: none of them
    PREDICTED: train, dev, and test
    GOLD_TRAIN: dev and test.  The model trains on gold tags, but is
      scored and has its checkpoints chosen with tags from the tagger
    """
    GOLD = 1
    PREDICTED = 2
    GOLD_TRAIN = 3

def add_retagging_args(parser, default_tag_method):
    """
    Add args for building a dataset specifically pertaining to retagging the dataset

    default_tag_method is the Tags value used if none of --gold,
    --predicted, or --gold_train is given
    """
    def tag_help(text, tag_method):
        if tag_method is default_tag_method:
            return text + '.  This is the default'
        return text

    parser.add_argument("--gold", dest='tag_method', action='store_const', const=Tags.GOLD, default=default_tag_method,
                        help=tag_help('Use gold tags for train, dev, and test', Tags.GOLD))
    parser.add_argument("--predicted", dest='tag_method', action='store_const', const=Tags.PREDICTED,
                        help=tag_help('Use predicted tags for train, dev, and test', Tags.PREDICTED))
    parser.add_argument("--gold_train", dest='tag_method', action='store_const', const=Tags.GOLD_TRAIN,
                        help=tag_help('Use gold tags for train and predicted tags for dev and test', Tags.GOLD_TRAIN))
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None,
                        help='Exact name of the pretrain file to read')
    parser.add_argument('--tagger_model', type=str, default=None,
                        help='Tagger save file to use.  If not specified, order searched will be saved/models, then $STANZA_RESOURCES_DIR')
    parser.add_argument('--save_dir', type=str, default=os.path.join('saved_models', 'pos'),
                        help='Where to look for recently trained POS models')
    parser.add_argument('--no_download_tagger', default=True, dest='download_tagger', action='store_false',
                        help="Don't try to automatically download a tagger for retagging the dependencies.  Will fail to make silver tags if there is no tagger model to be found")
    add_charlm_args(parser)

def choose_tagger_model(short_language, dataset, tagger_model, args):
    """
    Preferentially chooses a retrained tagger model, but tries to download one if that doesn't exist
    """
    logger.debug("Looking for tagger for lang |%s| dataset |%s|.  Suggested model |%s|.  Looking first in |%s|.", short_language, dataset, tagger_model, args.save_dir)
    if tagger_model:
        return tagger_model

    candidates = glob.glob(os.path.join(args.save_dir, "%s_%s_*.pt" % (short_language, dataset)))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        ending = TRANSFORMER_NICKNAMES.get(TRANSFORMERS.get(short_language))
        if ending is not None:
            endings = ("_%s_tagger.pt" % ending, "_charlm_tagger.pt", "_nocharlm_tagger.pt", "_nopretrain_tagger.pt")
        else:
            endings = ("_charlm_tagger.pt", "_nocharlm_tagger.pt", "_nopretrain_tagger.pt")
        for ending in endings:
            best_candidates = [x for x in candidates if x.endswith(ending)]
            if len(best_candidates) == 1:
                return best_candidates[0]
            if len(best_candidates) > 1:
                raise FileNotFoundError("Could not choose among the candidate taggers... please pick one with --tagger_model: {}".format(best_candidates))
        raise FileNotFoundError("Could not choose among the candidate taggers... please pick one with --tagger_model: {}".format(candidates))

    if not args.download_tagger:
        return None

    # TODO: just create a Pipeline for the retagging instead?
    pos_path = os.path.join(DEFAULT_MODEL_DIR, short_language, "pos", dataset + ".pt")
    if os.path.exists(pos_path):
        return pos_path
    try:
        download_list = download(lang=short_language, package=None, processors={"pos": dataset})
    except UnknownLanguageError as e:
        raise FileNotFoundError("The language %s appears to be a language new to Stanza.  Unfortunately, that means there are no taggers available for retagging the dataset.  Furthermore, there are no tagger models for this language found in %s.  You can specify a different directory for already trained tagger models with --save_dir, specify an exact tagger model name with --tagger_model, or use gold tags with --gold" % (short_language, args.save_dir)) from e
    for processor, name in download_list:
        if processor == 'pos':
            pos_path = os.path.join(DEFAULT_MODEL_DIR, short_language, "pos", name + ".pt")
            return pos_path
    else:
        raise FileNotFoundError("Could not figure out which model file to use for %s.  Just tried to download to %s the models %s" % (short_language, args.save_dir, download_list))


def build_retagger(treebank, paths, args):
    short_name = treebank_to_short_name(treebank)
    short_language, dataset = short_name.split("_", 1)

    # fmt: off
    base_args = ["--lang", short_language,
                 "--shorthand", short_name,
                 "--mode", "predict",
                 "--no_tag_breakdown"]
    # fmt: on

    # perhaps download a tagger if one doesn't already exist
    tagger_model = choose_tagger_model(short_language, dataset, args.tagger_model, args)
    if tagger_model is None:
        raise FileNotFoundError("Cannot find a tagger for language %s, dataset %s - you can specify one with the --tagger_model flag")
    else:
        logger.info("Using tagger model in %s for %s_%s", tagger_model, short_language, dataset)
    tagger_dir, tagger_name = os.path.split(tagger_model)
    base_args = base_args + ['--save_dir', tagger_dir, '--save_name', tagger_name]

    # word vector file for POS
    if args.wordvec_pretrain_file:
        base_args += ["--wordvec_pretrain_file", args.wordvec_pretrain_file]
    else:
        base_args = base_args + build_pos_wordvec_args(short_language, dataset, [])


    # charlm for POS
    charlm = choose_charlm(short_language, dataset, args.charlm, default_charlms, pos_charlms)
    charlm_args = build_charlm_args(short_language, charlm)
    base_args = base_args + charlm_args

    def retag_dataset(tokenizer_dir, tokenizer_file, dest_dir, dest_file, short_name):
        original = f"{tokenizer_dir}/{short_name}.{tokenizer_file}.conllu"
        retagged = f"{dest_dir}/{short_name}.{dest_file}.conllu"
        # fmt: off
        tagger_args = ["--eval_file", original,
                       "--output_file", retagged]
        # fmt: on
        tagger_args = base_args + tagger_args
        logger.info("Running tagger to retag {} to {}\n  Args: {}".format(original, retagged, tagger_args))
        tagger.main(tagger_args)

    return retag_dataset

def build_postprocess(treebank, paths, args):
    """
    Build the postprocess for copy_conllu_treebank which applies args.tag_method

    Returns None for Tags.GOLD, which copies each split unchanged.
    The tagger is only looked for if some split gets predicted tags.
    """
    if args.tag_method is Tags.GOLD:
        return None
    if args.tag_method not in (Tags.PREDICTED, Tags.GOLD_TRAIN):
        raise ValueError("Unknown tags method: {}".format(args.tag_method))

    retag_dataset = build_retagger(treebank, paths, args)
    if args.tag_method is Tags.PREDICTED:
        return retag_dataset

    def retag_dev_test(tokenizer_dir, tokenizer_file, dest_dir, dest_file, short_name):
        if tokenizer_file.split(".")[0] == "train":
            copy_conllu_file(tokenizer_dir, tokenizer_file, dest_dir, dest_file, short_name)
        else:
            retag_dataset(tokenizer_dir, tokenizer_file, dest_dir, dest_file, short_name)

    return retag_dev_test
