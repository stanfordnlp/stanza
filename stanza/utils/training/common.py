import argparse
import glob
import logging
import os
import pathlib
import sys
import tempfile

from enum import Enum

from stanza.resources.default_packages import default_charlms, lemma_charlms, pos_charlms, depparse_charlms, TRANSFORMERS, TRANSFORMER_LAYERS
from stanza.models.common.constant import treebank_to_short_name
from stanza.models.common.utils import ud_scores
from stanza.resources.common import download, DEFAULT_MODEL_DIR, UnknownLanguageError
from stanza.utils.datasets import common
import stanza.utils.default_paths as default_paths
from stanza.utils import conll18_ud_eval as ud_eval

logger = logging.getLogger('stanza')

class Mode(Enum):
    TRAIN = 1
    SCORE_DEV = 2
    SCORE_TEST = 3
    SCORE_TRAIN = 4

class ArgumentParserWithExtraHelp(argparse.ArgumentParser):
    def __init__(self, sub_argparse, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments

        self.sub_argparse = sub_argparse

    def print_help(self, file=None):
        super().print_help(file=file)

    def format_help(self):
        help_text = super().format_help()
        if self.sub_argparse is not None:
            sub_text = self.sub_argparse.format_help().split("\n")
            first_line = -1
            for line_idx, line in enumerate(sub_text):
                if line.strip().startswith("usage:"):
                    first_line = line_idx
                elif first_line >= 0 and not line.strip():
                    first_line = line_idx
                    break
            help_text = help_text + "\n\nmodel arguments:" + "\n".join(sub_text[first_line:])
        return help_text


def build_argparse(sub_argparse=None):
    parser = ArgumentParserWithExtraHelp(sub_argparse=sub_argparse, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_output', dest='temp_output', default=True, action='store_false', help="Save output - default is to use a temp directory.")

    parser.add_argument('treebanks', type=str, nargs='+', help='Which treebanks to run on.  Use all_ud or ud_all for all UD treebanks')

    parser.add_argument('--train', dest='mode', default=Mode.TRAIN, action='store_const', const=Mode.TRAIN, help='Run in train mode')
    parser.add_argument('--score_dev', dest='mode', action='store_const', const=Mode.SCORE_DEV, help='Score the dev set')
    parser.add_argument('--score_test', dest='mode', action='store_const', const=Mode.SCORE_TEST, help='Score the test set')
    parser.add_argument('--score_train', dest='mode', action='store_const', const=Mode.SCORE_TRAIN, help='Score the train set as a test set.  Currently only implemented for some models')

    # These arguments need to be here so we can identify if the model already exists in the user-specified home
    # TODO: when all of the model scripts handle their own names, can eliminate this argument
    parser.add_argument('--save_dir', type=str, default=None, help="Root dir for saving models.  If set, will override the model's default.")
    parser.add_argument('--save_name', type=str, default=None, help="Base name for saving models.  If set, will override the model's default.")

    parser.add_argument('--charlm_only', action='store_true', default=False, help='When asking for ud_all, filter the ones which have charlms')
    parser.add_argument('--transformer_only', action='store_true', default=False, help='When asking for ud_all, filter the ones for languages where we have transformers')

    parser.add_argument('--force', dest='force', action='store_true', default=False, help='Retrain existing models')
    return parser

def add_charlm_args(parser):
    parser.add_argument('--charlm', default="default", type=str, help='Which charlm to run on.  Will use the default charlm for this language/model if not set.  Set to None to turn off charlm for languages with a default charlm')
    parser.add_argument('--no_charlm', dest='charlm', action="store_const", const=None, help="Don't use a charlm, even if one is used by default for this package")

def main(run_treebank, model_dir, model_name, add_specific_args=None, sub_argparse=None, build_model_filename=None, choose_charlm_method=None, args=None):
    """
    A main program for each of the run_xyz scripts

    It collects the arguments and runs the main method for each dataset provided.
    It also tries to look for an existing model and not overwrite it unless --force is provided

    model_name can be a callable expecting the args
      - the charlm, for example, needs this feature, since it makes
        both forward and backward models
    """
    if args is None:
        logger.info("Training program called with:\n" + " ".join(sys.argv))
        args = sys.argv[1:]
    else:
        logger.info("Training program called with:\n" + " ".join(args))

    paths = default_paths.get_default_paths()

    parser = build_argparse(sub_argparse)
    if add_specific_args is not None:
        add_specific_args(parser)
    if '--extra_args' in sys.argv:
        idx = sys.argv.index('--extra_args')
        extra_args = sys.argv[idx+1:]
        command_args = parser.parse_args(sys.argv[:idx])
    else:
        command_args, extra_args = parser.parse_known_args(args=args)

    # Pass this through to the underlying model as well as use it here
    # we don't put --save_name here for the awkward situation of
    # --save_name being specified for an invocation with multiple treebanks
    if command_args.save_dir:
        extra_args.extend(["--save_dir", command_args.save_dir])

    if callable(model_name):
        model_name = model_name(command_args)

    mode = command_args.mode
    treebanks = []

    for treebank in command_args.treebanks:
        # this is a really annoying typo to make if you copy/paste a
        # UD directory name on the cluster and your job dies 30s after
        # being queued for an hour
        if treebank.endswith("/"):
            treebank = treebank[:-1]
        if treebank.lower() in ('ud_all', 'all_ud'):
            ud_treebanks = common.get_ud_treebanks(paths["UDBASE"])
            if choose_charlm_method is not None and command_args.charlm_only:
                logger.info("Filtering ud_all treebanks to only those which can use charlm for this model")
                ud_treebanks = [x for x in ud_treebanks
                                if choose_charlm_method(*treebank_to_short_name(x).split("_", 1), 'default') is not None]
            if command_args.transformer_only:
                logger.info("Filtering ud_all treebanks to only those which can use a transformer for this model")
                ud_treebanks = [x for x in ud_treebanks if treebank_to_short_name(x).split("_")[0] in TRANSFORMERS]
            logger.info("Expanding %s to %s", treebank, " ".join(ud_treebanks))
            treebanks.extend(ud_treebanks)
        else:
            treebanks.append(treebank)

    for treebank_idx, treebank in enumerate(treebanks):
        if treebank_idx > 0:
            logger.info("=========================================")

        short_name = treebank_to_short_name(treebank)
        logger.debug("%s: %s" % (treebank, short_name))

        save_name_args = []
        if model_name != 'ete':
            # ete is several models at once, so we don't set --save_name
            # theoretically we could handle a parametrized save_name
            if command_args.save_name:
                save_name = command_args.save_name
                # if there's more than 1 treebank, we can't save them all to this save_name
                # we have to override that value for each treebank
                if len(treebanks) > 1:
                    save_name_dir, save_name_filename = os.path.split(save_name)
                    save_name_filename = "%s_%s" % (short_name, save_name_filename)
                    save_name = os.path.join(save_name_dir, save_name_filename)
                    logger.info("Save file for %s model for %s: %s", short_name, treebank, save_name)
                save_name_args = ['--save_name', save_name]
            # some run scripts can build the model filename
            # in order to check for models that are already created
            elif build_model_filename is None:
                save_name = "%s_%s.pt" % (short_name, model_name)
                logger.info("Save file for %s model: %s", short_name, save_name)
                save_name_args = ['--save_name', save_name]
            else:
                save_name_args = []

            if mode == Mode.TRAIN and not command_args.force:
                if build_model_filename is not None:
                    model_path = build_model_filename(paths, short_name, command_args, extra_args)
                elif command_args.save_dir:
                    model_path = os.path.join(command_args.save_dir, save_name)
                else:
                    save_dir = os.path.join("saved_models", model_dir)
                    save_name_args.extend(["--save_dir", save_dir])
                    model_path = os.path.join(save_dir, save_name)

                if model_path is None:
                    # this can happen with the identity lemmatizer, for example
                    pass
                elif os.path.exists(model_path):
                    logger.info("%s: %s exists, skipping!" % (treebank, model_path))
                    continue
                else:
                    logger.info("%s: %s does not exist, training new model" % (treebank, model_path))

        if command_args.temp_output and model_name != 'ete':
            with tempfile.NamedTemporaryFile() as temp_output_file:
                run_treebank(mode, paths, treebank, short_name,
                             temp_output_file.name, command_args, extra_args + save_name_args)
        else:
            run_treebank(mode, paths, treebank, short_name,
                         None, command_args, extra_args + save_name_args)

def run_eval_script(gold_conllu_file, system_conllu_file, evals=None):
    """ Wrapper for lemma scorer. """
    evaluation = ud_scores(gold_conllu_file, system_conllu_file)

    if evals is None:
        return ud_eval.build_evaluation_table(evaluation, verbose=True, counts=False, enhanced=False)
    else:
        results = [evaluation[key].f1 for key in evals]
        max_len = max(5, max(len(e) for e in evals))
        evals_string = " ".join(("{:>%d}" % max_len).format(e) for e in evals)
        results_string = " ".join(("{:%d.2f}" % max_len).format(100 * x) for x in results)
        return evals_string + "\n" + results_string

def run_eval_script_tokens(eval_gold, eval_pred):
    return run_eval_script(eval_gold, eval_pred, evals=["Tokens", "Sentences", "Words"])

def run_eval_script_mwt(eval_gold, eval_pred):
    return run_eval_script(eval_gold, eval_pred, evals=["Words"])

def run_eval_script_pos(eval_gold, eval_pred):
    return run_eval_script(eval_gold, eval_pred, evals=["UPOS", "XPOS", "UFeats", "AllTags"])

def run_eval_script_depparse(eval_gold, eval_pred):
    return run_eval_script(eval_gold, eval_pred, evals=["UAS", "LAS", "CLAS", "MLAS", "BLEX"])


def find_wordvec_pretrain(language, default_pretrains, dataset_pretrains=None, dataset=None, model_dir=DEFAULT_MODEL_DIR):
    # try to get the default pretrain for the language,
    # but allow the package specific value to override it if that is set
    default_pt = default_pretrains.get(language, None)
    if dataset is not None and dataset_pretrains is not None:
        default_pt = dataset_pretrains.get(language, {}).get(dataset, default_pt)

    if default_pt is not None:
        default_pt_path = '{}/{}/pretrain/{}.pt'.format(model_dir, language, default_pt)
        if not os.path.exists(default_pt_path):
            logger.info("Default pretrain should be {}  Attempting to download".format(default_pt_path))
            try:
                download(lang=language, package=None, processors={"pretrain": default_pt}, model_dir=model_dir)
            except UnknownLanguageError:
                # if there's a pretrain in the directory, hiding this
                # error will let us find that pretrain later
                pass
        if os.path.exists(default_pt_path):
            if dataset is not None and dataset_pretrains is not None and language in dataset_pretrains and dataset in dataset_pretrains[language]:
                logger.info(f"Using default pretrain for {language}:{dataset}, found in {default_pt_path}  To use a different pretrain, specify --wordvec_pretrain_file")
            else:
                logger.info(f"Using default pretrain for language, found in {default_pt_path}  To use a different pretrain, specify --wordvec_pretrain_file")
            return default_pt_path

    pretrain_path = '{}/{}/pretrain/*.pt'.format(model_dir, language)
    pretrains = glob.glob(pretrain_path)
    if len(pretrains) == 0:
        # we already tried to download the default pretrain once
        # and it didn't work.  maybe the default language package
        # will have something?
        logger.warning(f"Cannot figure out which pretrain to use for '{language}'.  Will download the default package and hope for the best")
        try:
            download(lang=language, model_dir=model_dir)
        except UnknownLanguageError as e:
            # this is a very unusual situation
            # basically, there was a language which we started to add
            # to the resources, but then didn't release the models
            # as part of resources.json
            raise FileNotFoundError(f"Cannot find any pretrains in {pretrain_path}  No pretrains in the system for this language.  Please prepare an embedding as a .pt and use --wordvec_pretrain_file to specify a .pt file to use") from e
        pretrains = glob.glob(pretrain_path)
    if len(pretrains) == 0:
        raise FileNotFoundError(f"Cannot find any pretrains in {pretrain_path}  Try 'stanza.download(\"{language}\")' to get a default pretrain or use --wordvec_pretrain_file to specify a .pt file to use")
    if len(pretrains) > 1:
        raise FileNotFoundError(f"Too many pretrains to choose from in {pretrain_path}  Must specify an exact path to a --wordvec_pretrain_file")
    pt = pretrains[0]
    logger.info(f"Using pretrain found in {pt}  To use a different pretrain, specify --wordvec_pretrain_file")
    return pt

def find_charlm_file(direction, language, charlm, model_dir=DEFAULT_MODEL_DIR):
    """
    Return the path to the forward or backward charlm if it exists for the given package

    If we can figure out the package, but can't find it anywhere, we try to download it
    """
    saved_path = 'saved_models/charlm/{}_{}_{}_charlm.pt'.format(language, charlm, direction)
    if os.path.exists(saved_path):
        logger.info(f'Using model {saved_path} for {direction} charlm')
        return saved_path

    resource_path = '{}/{}/{}_charlm/{}.pt'.format(model_dir, language, direction, charlm)
    if os.path.exists(resource_path):
        logger.info(f'Using model {resource_path} for {direction} charlm')
        return resource_path

    try:
        download(lang=language, package=None, processors={f"{direction}_charlm": charlm}, model_dir=model_dir)
        if os.path.exists(resource_path):
            logger.info(f'Downloaded model, using model {resource_path} for {direction} charlm')
            return resource_path
    except ValueError as e:
        raise FileNotFoundError(f"Cannot find {direction} charlm in either {saved_path} or {resource_path}  Attempted downloading {charlm} but that did not work") from e

    raise FileNotFoundError(f"Cannot find {direction} charlm in either {saved_path} or {resource_path}  Attempted downloading {charlm} but that did not work")

def build_charlm_args(language, charlm, base_args=True, model_dir=DEFAULT_MODEL_DIR):
    """
    If specified, return forward and backward charlm args
    """
    if charlm:
        try:
            forward = find_charlm_file('forward', language, charlm, model_dir=model_dir)
            backward = find_charlm_file('backward', language, charlm, model_dir=model_dir)
        except FileNotFoundError as e:
            # if we couldn't find sd_isra when training an SD model,
            # for example, but isra exists, we try to download the
            # shorter model name
            if charlm.startswith(language + "_"):
                short_charlm = charlm[len(language)+1:]
                try:
                    forward = find_charlm_file('forward', language, short_charlm, model_dir=model_dir)
                    backward = find_charlm_file('backward', language, short_charlm, model_dir=model_dir)
                except FileNotFoundError as e2:
                    raise FileNotFoundError("Tried to find charlm %s, which doesn't exist.  Also tried %s, but didn't find that either" % (charlm, short_charlm)) from e
                logger.warning("Was asked to find charlm %s, which does not exist.  Did find %s though", charlm, short_charlm)
            else:
                raise

        char_args = ['--charlm_forward_file', forward,
                     '--charlm_backward_file', backward]
        if not base_args:
            return char_args
        return ['--charlm',
                '--charlm_shorthand', f'{language}_{charlm}'] + char_args

    return []

def choose_charlm(language, dataset, charlm, language_charlms, dataset_charlms):
    """
    charlm == "default" means the default charlm for this dataset or language
    charlm == None is no charlm
    """
    default_charlm = language_charlms.get(language, None)
    specific_charlm = dataset_charlms.get(language, {}).get(dataset, None)

    if charlm is None:
        return None
    elif charlm != "default":
        return charlm
    elif dataset in dataset_charlms.get(language, {}):
        # this way, a "" or None result gets honored
        # thus treating "not in the map" as a way for dataset_charlms to signal to use the default
        return specific_charlm
    elif default_charlm:
        return default_charlm
    else:
        return None

def choose_pos_charlm(short_language, dataset, charlm):
    """
    charlm == "default" means the default charlm for this dataset or language
    charlm == None is no charlm
    """
    return choose_charlm(short_language, dataset, charlm, default_charlms, pos_charlms)

def choose_depparse_charlm(short_language, dataset, charlm):
    """
    charlm == "default" means the default charlm for this dataset or language
    charlm == None is no charlm
    """
    return choose_charlm(short_language, dataset, charlm, default_charlms, depparse_charlms)

def choose_lemma_charlm(short_language, dataset, charlm):
    """
    charlm == "default" means the default charlm for this dataset or language
    charlm == None is no charlm
    """
    return choose_charlm(short_language, dataset, charlm, default_charlms, lemma_charlms)

def choose_transformer(short_language, command_args, extra_args, warn=True, layers=False):
    """
    Choose a transformer using the default options for this language
    """
    bert_args = []
    if command_args is not None and command_args.use_bert and '--bert_model' not in extra_args:
        if short_language in TRANSFORMERS:
            bert_args = ['--bert_model', TRANSFORMERS.get(short_language)]
            if layers and short_language in TRANSFORMER_LAYERS and '--bert_hidden_layers' not in extra_args:
                bert_args.extend(['--bert_hidden_layers', str(TRANSFORMER_LAYERS.get(short_language))])
        elif warn:
            logger.error("Transformer requested, but no default transformer for %s  Specify one using --bert_model" % short_language)

    return bert_args

def build_pos_charlm_args(short_language, dataset, charlm, base_args=True, model_dir=DEFAULT_MODEL_DIR):
    charlm = choose_pos_charlm(short_language, dataset, charlm)
    charlm_args = build_charlm_args(short_language, charlm, base_args, model_dir)
    return charlm_args

def build_lemma_charlm_args(short_language, dataset, charlm, base_args=True, model_dir=DEFAULT_MODEL_DIR):
    charlm = choose_lemma_charlm(short_language, dataset, charlm)
    charlm_args = build_charlm_args(short_language, charlm, base_args, model_dir)
    return charlm_args

def build_depparse_charlm_args(short_language, dataset, charlm, base_args=True, model_dir=DEFAULT_MODEL_DIR):
    charlm = choose_depparse_charlm(short_language, dataset, charlm)
    charlm_args = build_charlm_args(short_language, charlm, base_args, model_dir)
    return charlm_args
