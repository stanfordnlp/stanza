import argparse
import glob
import logging
import os
import pathlib
import sys
import tempfile

from enum import Enum

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

BERT = {
    # https://huggingface.co/Maltehb/danish-bert-botxo
    # contrary to normal expectations, this hurts F1
    # on a dev split by about 1 F1
    # "da": "Maltehb/danish-bert-botxo",
    #
    # the multilingual bert is a marginal improvement for conparse
    "da": "bert-base-multilingual-cased",

    # As of April 2022, the bert models available have a weird
    # tokenizer issue where soft hyphen causes it to crash.
    # We attempt to compensate for that in the dev branch
    # bert-base-german-cased
    # dev:  2022-04-27 21:21:31 INFO: de_germeval2014 87.59
    # test: 2022-04-27 21:21:59 INFO: de_germeval2014 86.95
    #
    # dbmdz/bert-base-german-cased
    # dev:  2022-04-27 22:24:59 INFO: de_germeval2014 88.22
    # test: 2022-04-27 22:25:27 INFO: de_germeval2014 87.80
    "de": "dbmdz/bert-base-german-cased",

    # https://huggingface.co/roberta-base
    "en": "roberta-base",

    # NER scores for a couple Persian options:
    # none:
    # dev:  2022-04-23 01:44:53 INFO: fa_arman 79.46
    # test: 2022-04-23 01:45:03 INFO: fa_arman 80.06
    #
    # HooshvareLab/bert-fa-zwnj-base
    # dev:  2022-04-23 02:43:44 INFO: fa_arman 80.87
    # test: 2022-04-23 02:44:07 INFO: fa_arman 80.81
    #
    # HooshvareLab/roberta-fa-zwnj-base
    # dev:  2022-04-23 16:23:25 INFO: fa_arman 81.23
    # test: 2022-04-23 16:23:48 INFO: fa_arman 81.11
    #
    # HooshvareLab/bert-base-parsbert-uncased
    # dev:  2022-04-26 10:42:09 INFO: fa_arman 82.49
    # test: 2022-04-26 10:42:31 INFO: fa_arman 83.16
    "fa": 'HooshvareLab/bert-base-parsbert-uncased',

    # NER scores for a couple options:
    # none:
    # dev:  2022-03-04 INFO: fi_turku 83.45
    # test: 2022-03-04 INFO: fi_turku 86.25
    #
    # bert-base-multilingual-cased
    # dev:  2022-03-04 INFO: fi_turku 85.23
    # test: 2022-03-04 INFO: fi_turku 89.00
    #
    # TurkuNLP/bert-base-finnish-cased-v1:
    # dev:  2022-03-04 INFO: fi_turku 88.41
    # test: 2022-03-04 INFO: fi_turku 91.36
    "fi": "TurkuNLP/bert-base-finnish-cased-v1",

    # from https://github.com/idb-ita/GilBERTo
    # annoyingly, it doesn't handle cased text
    # supposedly there is an argument "do_lower_case"
    # but that still leaves a lot of unk tokens
    # "it": "idb-ita/gilberto-uncased-from-camembert",
    #
    # from https://github.com/musixmatchresearch/umberto
    # on NER, this gets 88.37 dev and 91.02 test
    # another option is dbmdz/bert-base-italian-cased,
    # which gets 87.27 dev and 90.32 test
    "it": "Musixmatch/umberto-commoncrawl-cased-v1",

    # experiments on the cintil conparse dataset
    # ran a variety of transformer settings
    # found the following dev set scores after 400 iterations:
    # Geotrend/distilbert-base-pt-cased : not plug & play
    # no bert: 0.9082
    # xlm-roberta-base: 0.9109
    # xlm-roberta-large: 0.9254
    # adalbertojunior/distilbert-portuguese-cased: 0.9300
    # neuralmind/bert-base-portuguese-cased: 0.9307
    # neuralmind/bert-large-portuguese-cased: 0.9343
    "pt": "neuralmind/bert-large-portuguese-cased",

    # https://huggingface.co/dbmdz/bert-base-turkish-128k-cased
    # helps the Turkish model quite a bit
    "tr": "dbmdz/bert-base-turkish-128k-cased",

    # from https://github.com/VinAIResearch/PhoBERT
    # "vi": "vinai/phobert-base",
    # another option is phobert-large, but that doesn't
    # change the scores any
    "vi": "vinai/phobert-large",

    # https://github.com/ymcui/Chinese-BERT-wwm
    # there's also hfl/chinese-roberta-wwm-ext-large
    "zh-hans": "hfl/chinese-roberta-wwm-ext",

    # https://huggingface.co/allegro/herbert-base-cased
    # Scores by entity on the NKJP NER task:
    # no bert (dev/test): 88.64/88.75
    # herbert-base-cased (dev/test): 91.48/91.02,
    # herbert-large-cased (dev/test): 92.25/91.62
    # sdadas/polish-roberta-large-v2 (dev/test): 92.66/91.22
    "pl": "allegro/herbert-base-cased",
}

def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_output', dest='temp_output', default=True, action='store_false', help="Save output - default is to use a temp directory.")

    parser.add_argument('treebanks', type=str, nargs='+', help='Which treebanks to run on.  Use all_ud or ud_all for all UD treebanks')

    parser.add_argument('--train', dest='mode', default=Mode.TRAIN, action='store_const', const=Mode.TRAIN, help='Run in train mode')
    parser.add_argument('--score_dev', dest='mode', action='store_const', const=Mode.SCORE_DEV, help='Score the dev set')
    parser.add_argument('--score_test', dest='mode', action='store_const', const=Mode.SCORE_TEST, help='Score the test set')

    # These arguments need to be here so we can identify if the model already exists in the user-specified home
    parser.add_argument('--save_dir', type=str, default=None, help="Root dir for saving models.  If set, will override the model's default.")
    parser.add_argument('--save_name', type=str, default=None, help="Base name for saving models.  If set, will override the model's default.")

    parser.add_argument('--force', dest='force', action='store_true', default=False, help='Retrain existing models')
    return parser

def add_charlm_args(parser):
    parser.add_argument('--charlm', default="default", type=str, help='Which charlm to run on.  Will use the default charlm for this language/model if not set.  Set to None to turn off charlm for languages with a default charlm')
    parser.add_argument('--no_charlm', dest='charlm', action="store_const", const=None, help="Don't use a charlm, even if one is used by default for this package")

def main(run_treebank, model_dir, model_name, add_specific_args=None):
    """
    A main program for each of the run_xyz scripts

    It collects the arguments and runs the main method for each dataset provided.
    It also tries to look for an existing model and not overwrite it unless --force is provided

    model_name can be a callable expecting the args
      - the charlm, for example, needs this feature, since it makes
        both forward and backward models
    """
    logger.info("Training program called with:\n" + " ".join(sys.argv))

    paths = default_paths.get_default_paths()

    parser = build_argparse()
    if add_specific_args is not None:
        add_specific_args(parser)
    if '--extra_args' in sys.argv:
        idx = sys.argv.index('--extra_args')
        extra_args = sys.argv[idx+1:]
        command_args = parser.parse_args(sys.argv[1:idx])
    else:
        command_args, extra_args = parser.parse_known_args()

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
            else:
                save_name = "%s_%s.pt" % (short_name, model_name)
                logger.info("Save file for %s model: %s", short_name, save_name)
            save_name_args = ['--save_name', save_name]

        if mode == Mode.TRAIN and not command_args.force and model_name != 'ete':
            if command_args.save_dir:
                model_path = os.path.join(command_args.save_dir, save_name)
            else:
                save_dir = os.path.join("saved_models", model_dir)
                save_name_args.extend(["--save_dir", save_dir])
                model_path = os.path.join(save_dir, save_name)

            if os.path.exists(model_path):
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
        return " ".join("{:.2f}".format(100 * x) for x in results)

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
                download(lang=language, package=None, processors={"pretrain": default_pt})
            except UnknownLanguageError:
                # if there's a pretrain in the directory, hiding this
                # error will let us find that pretrain later
                pass
        if os.path.exists(default_pt_path):
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
            download(lang=language)
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
        download(lang=language, package=None, processors={f"{direction}_charlm": charlm})
        if os.path.exists(resource_path):
            logger.info(f'Downloaded model, using model {resource_path} for {direction} charlm')
            return resource_path
    except ValueError as e:
        # we're about to throw an error anyway
        pass

    raise FileNotFoundError(f"Cannot find {direction} charlm in either {saved_path} or {resource_path}  Attempted downloading {charlm} but that did not work")

def build_charlm_args(language, charlm, base_args=True, model_dir=DEFAULT_MODEL_DIR):
    """
    If specified, return forward and backward charlm args
    """
    if charlm:
        forward = find_charlm_file('forward', language, charlm, model_dir=model_dir)
        backward = find_charlm_file('backward', language, charlm, model_dir=model_dir)
        char_args = ['--charlm_forward_file', forward,
                     '--charlm_backward_file', backward]
        if not base_args:
            return char_args
        return ['--charlm',
                '--charlm_shorthand', f'{language}_{charlm}'] + char_args

    return []

def choose_charlm(language, dataset, charlm, language_charlms, dataset_charlms):
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

