"""
Utilities for testing
"""

import os
import re

# Environment Variables
# set this to specify working directory of tests
TEST_HOME_VAR = 'STANZA_TEST_HOME'

# Global Variables
# test working directory base name must be stanza_test
TEST_DIR_BASE_NAME = 'stanza_test'

# check the working dir is set and compliant
assert os.getenv(TEST_HOME_VAR) is not None, \
    f'Please set {TEST_HOME_VAR} environment variable for test working dir, base name must be: {TEST_DIR_BASE_NAME}'
TEST_WORKING_DIR = os.getenv(TEST_HOME_VAR)
assert os.path.basename(TEST_WORKING_DIR) == TEST_DIR_BASE_NAME, \
    f'Base name of test home dir must be: {TEST_DIR_BASE_NAME}'

TEST_MODELS_DIR = f'{TEST_WORKING_DIR}/models'

# server resources
SERVER_TEST_PROPS = f'{TEST_WORKING_DIR}/scripts/external_server.properties'

# language resources
LANGUAGE_RESOURCES = {}

TOKENIZE_MODEL = 'tokenizer.pt'
MWT_MODEL = 'mwt_expander.pt'
POS_MODEL = 'tagger.pt'
POS_PRETRAIN = 'pretrain.pt'
LEMMA_MODEL = 'lemmatizer.pt'
DEPPARSE_MODEL = 'parser.pt'
DEPPARSE_PRETRAIN = 'pretrain.pt'

MODEL_FILES = [TOKENIZE_MODEL, MWT_MODEL, POS_MODEL, POS_PRETRAIN, LEMMA_MODEL, DEPPARSE_MODEL, DEPPARSE_PRETRAIN]

# English resources
EN_KEY = 'en'
EN_SHORTHAND = 'en_ewt'
# models
EN_MODELS_DIR = f'{TEST_WORKING_DIR}/models/{EN_SHORTHAND}_models'
EN_MODEL_FILES = [f'{EN_MODELS_DIR}/{EN_SHORTHAND}_{model_fname}' for model_fname in MODEL_FILES]

# French resources
FR_KEY = 'fr'
FR_SHORTHAND = 'fr_gsd'
# regression file paths
FR_TEST_IN = f'{TEST_WORKING_DIR}/in/fr_gsd.test.txt'
FR_TEST_OUT = f'{TEST_WORKING_DIR}/out/fr_gsd.test.txt.out'
FR_TEST_GOLD_OUT = f'{TEST_WORKING_DIR}/out/fr_gsd.test.txt.out.gold'
# models
FR_MODELS_DIR = f'{TEST_WORKING_DIR}/models/{FR_SHORTHAND}_models'
FR_MODEL_FILES = [f'{FR_MODELS_DIR}/{FR_SHORTHAND}_{model_fname}' for model_fname in MODEL_FILES]

# Other language resources
AR_SHORTHAND = 'ar_padt'
DE_SHORTHAND = 'de_gsd'
KK_SHORTHAND = 'kk_ktb'
KO_SHORTHAND = 'ko_gsd'


# utils for clean up
# only allow removal of dirs/files in this approved list
REMOVABLE_PATHS = ['en_ewt_models', 'en_ewt_tokenizer.pt', 'en_ewt_mwt_expander.pt', 'en_ewt_tagger.pt',
                   'en_ewt.pretrain.pt', 'en_ewt_lemmatizer.pt', 'en_ewt_parser.pt', 'fr_gsd_models',
                   'fr_gsd_tokenizer.pt', 'fr_gsd_mwt_expander.pt', 'fr_gsd_tagger.pt', 'fr_gsd.pretrain.pt',
                   'fr_gsd_lemmatizer.pt', 'fr_gsd_parser.pt', 'ar_padt_models', 'ar_padt_tokenizer.pt',
                   'ar_padt_mwt_expander.pt', 'ar_padt_tagger.pt', 'ar_padt.pretrain.pt', 'ar_padt_lemmatizer.pt',
                   'ar_padt_parser.pt', 'de_gsd_models', 'de_gsd_tokenizer.pt', 'de_gsd_mwt_expander.pt',
                   'de_gsd_tagger.pt', 'de_gsd.pretrain.pt', 'de_gsd_lemmatizer.pt', 'de_gsd_parser.pt',
                   'kk_ktb_models', 'kk_ktb_tokenizer.pt', 'kk_ktb_mwt_expander.pt', 'kk_ktb_tagger.pt',
                   'kk_ktb.pretrain.pt', 'kk_ktb_lemmatizer.pt', 'kk_ktb_parser.pt', 'ko_gsd_models',
                   'ko_gsd_tokenizer.pt', 'ko_gsd_mwt_expander.pt', 'ko_gsd_tagger.pt', 'ko_gsd.pretrain.pt',
                   'ko_gsd_lemmatizer.pt', 'ko_gsd_parser.pt']


def safe_rm(path_to_rm):
    """
    Safely remove a directory of files or a file
    1.) check path exists, files are files, dirs are dirs
    2.) only remove things on approved list REMOVABLE_PATHS
    3.) assert no longer exists
    """
    # just return if path doesn't exist
    if not os.path.exists(path_to_rm):
        return
    # handle directory
    if os.path.isdir(path_to_rm):
        files_to_rm = [f'{path_to_rm}/{fname}' for fname in os.listdir(path_to_rm)]
        dir_to_rm = path_to_rm
    else:
        files_to_rm = [path_to_rm]
        dir_to_rm = None
    # clear out files
    for file_to_rm in files_to_rm:
        if os.path.isfile(file_to_rm) and os.path.basename(file_to_rm) in REMOVABLE_PATHS:
            os.remove(file_to_rm)
            assert not os.path.exists(file_to_rm), f'Error removing: {file_to_rm}'
    # clear out directory
    if dir_to_rm is not None and os.path.isdir(dir_to_rm):
        os.rmdir(dir_to_rm)
        assert not os.path.exists(dir_to_rm), f'Error removing: {dir_to_rm}'

def compare_ignoring_whitespace(predicted, expected):
    predicted = re.sub('[ \t]+', ' ', predicted.strip())
    expected = re.sub('[ \t]+', ' ', expected.strip())
    assert predicted == expected

