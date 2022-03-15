import argparse
import glob
import logging
import os
import pathlib
import re
import subprocess
import sys
import tempfile

from enum import Enum

from stanza.models.common.constant import treebank_to_short_name
from stanza.resources.common import DEFAULT_MODEL_DIR
from stanza.utils.datasets import common
import stanza.utils.default_paths as default_paths
from stanza.utils import conll18_ud_eval as ud_eval

logger = logging.getLogger('stanza')

class Mode(Enum):
    TRAIN = 1
    SCORE_DEV = 2
    SCORE_TEST = 3

BERT = {
    # https://huggingface.co/dbmdz/bert-base-turkish-128k-cased
    # helps the Turkish model quite a bit
    "tr": "dbmdz/bert-base-turkish-128k-cased",

    # https://huggingface.co/Maltehb/danish-bert-botxo
    # contrary to normal expectations, this hurts F1
    # on a dev split by about 1 F1
    # "da": "Maltehb/danish-bert-botxo",

    # the multilingual bert is a marginal improvement for conparse
    "da": "bert-base-multilingual-cased",

    # from https://github.com/idb-ita/GilBERTo
    # annoyingly, it doesn't handle cased text
    # supposedly there is an argument "do_lower_case"
    # but that still leaves a lot of unk tokens
    # "it": "idb-ita/gilberto-uncased-from-camembert",

    # from https://github.com/musixmatchresearch/umberto
    # on NER, this gets 88.37 dev and 91.02 test
    # another option is dbmdz/bert-base-italian-cased,
    # which gets 87.27 dev and 90.32 test
    "it": "Musixmatch/umberto-commoncrawl-cased-v1",

    # from https://github.com/VinAIResearch/PhoBERT
    # "vi": "vinai/phobert-base",
    # another option is phobert-large, but that doesn't
    # change the scores any
    "vi": "vinai/phobert-large",

    # https://huggingface.co/roberta-base
    "en": "roberta-base",

    # https://github.com/ymcui/Chinese-BERT-wwm
    # there's also hfl/chinese-roberta-wwm-ext-large
    "zh-hans": "hfl/chinese-roberta-wwm-ext",

    # experiments on the cintil dataset
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
}

def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_output', dest='temp_output', default=True, action='store_false', help="Save output - default is to use a temp directory.")

    parser.add_argument('treebanks', type=str, nargs='+', help='Which treebanks to run on.  Use all_ud or ud_all for all UD treebanks')

    parser.add_argument('--train', dest='mode', default=Mode.TRAIN, action='store_const', const=Mode.TRAIN, help='Run in train mode')
    parser.add_argument('--score_dev', dest='mode', action='store_const', const=Mode.SCORE_DEV, help='Score the dev set')
    parser.add_argument('--score_test', dest='mode', action='store_const', const=Mode.SCORE_TEST, help='Score the test set')

    # This argument needs to be here so we can identify if the model already exists in the user-specified home
    parser.add_argument('--save_dir', type=str, default=None, help="Root dir for saving models.  If set, will override the model's default.")

    parser.add_argument('--force', dest='force', action='store_true', default=False, help='Retrain existing models')
    return parser

SHORTNAME_RE = re.compile("[a-z-]+_[a-z0-9]+")

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

        if SHORTNAME_RE.match(treebank):
            short_name = treebank
        else:
            short_name = treebank_to_short_name(treebank)
        logger.debug("%s: %s" % (treebank, short_name))

        if mode == Mode.TRAIN and not command_args.force and model_name != 'ete':
            if command_args.save_dir:
                model_path = "%s/%s_%s.pt" % (command_args.save_dir, short_name, model_name)
            else:
                model_path = "saved_models/%s/%s_%s.pt" % (model_dir, short_name, model_name)
            if os.path.exists(model_path):
                logger.info("%s: %s exists, skipping!" % (treebank, model_path))
                continue
            else:
                logger.info("%s: %s does not exist, training new model" % (treebank, model_path))

        if command_args.temp_output and model_name != 'ete':
            with tempfile.NamedTemporaryFile() as temp_output_file:
                run_treebank(mode, paths, treebank, short_name,
                             temp_output_file.name, command_args, extra_args)
        else:
            run_treebank(mode, paths, treebank, short_name,
                         None, command_args, extra_args)

def run_eval_script(gold_conllu_file, system_conllu_file, evals=None):
    """ Wrapper for lemma scorer. """
    gold_ud = ud_eval.load_conllu_file(gold_conllu_file)
    system_ud = ud_eval.load_conllu_file(system_conllu_file)
    evaluation = ud_eval.evaluate(gold_ud, system_ud)

    if evals is None:
        return ud_eval.build_evaluation_table(evaluation, verbose=True, counts=False)
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


def find_wordvec_pretrain(language, default_pretrain):
    # TODO: try to extract/remember the specific pretrain for the given model
    # That would be a good way to archive which pretrains are used for which NER models, anyway
    pretrain_path = '{}/{}/pretrain/*.pt'.format(DEFAULT_MODEL_DIR, language)
    pretrains = glob.glob(pretrain_path)
    if len(pretrains) == 0:
        raise FileNotFoundError(f"Cannot find any pretrains in {pretrain_path}  Try 'stanza.download(\"{language}\")' to get a default pretrain or use --wordvec_pretrain_file to specify a .pt file to use")
    if len(pretrains) > 1:
        default_pt = default_pretrain.get(language, None)
        if default_pt is None:
            raise FileNotFoundError(f"Too many pretrains to choose from in {pretrain_path}  No default pretrain is specified for language {language}  Must specify an exact path to a --wordvec_pretrain_file")
        for pt_file in pretrains:
            pt_name = os.path.split(pt_file)[1]
            pt_name = os.path.splitext(pt_name)[0]
            if pt_name == default_pt:
                logger.info(f"Using default pretrain for language, found in {pt_file}  To use a different pretrain, specify --wordvec_pretrain_file")
                return pt_file
        raise FileNotFoundError(f"Too many pretrains to choose from in {pretrain_path}  Could not find default pt {default_pt} for language {language}  Must specify an exact path to a --wordvec_pretrain_file")
    pt = pretrains[0]
    logger.info(f"Using pretrain found in {pt}  To use a different pretrain, specify --wordvec_pretrain_file")
    return pt

def find_charlm(direction, language, charlm):
    """
    Return the path to the forward or backward charlm if it exists for the given package
    """
    saved_path = 'saved_models/charlm/{}_{}_{}_charlm.pt'.format(language, charlm, direction)
    if os.path.exists(saved_path):
        logger.info(f'Using model {saved_path} for {direction} charlm')
        return saved_path

    resource_path = '{}/{}/{}_charlm/{}.pt'.format(DEFAULT_MODEL_DIR, language, direction, charlm)
    if os.path.exists(resource_path):
        logger.info(f'Using model {resource_path} for {direction} charlm')
        return resource_path

    raise FileNotFoundError(f"Cannot find {direction} charlm in either {saved_path} or {resource_path}")

def build_charlm_args(language, charlm, base_args=True):
    """
    If specified, return forward and backward charlm args
    """
    if charlm:
        forward = find_charlm('forward', language, charlm)
        backward = find_charlm('backward', language, charlm)
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
    elif specific_charlm:
        return specific_charlm
    elif default_charlm:
        return default_charlm
    else:
        return None

