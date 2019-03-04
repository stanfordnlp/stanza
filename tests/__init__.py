"""
Utilities for testing
"""

import os
import pytest


# Environment Variables
# set this to specify test mode (run full set of tests, or smaller set on Travis)
TEST_MODE_VAR = 'STANFORDNLP_TEST_MODE'
# set this to specify working directory of tests
TEST_HOME_VAR = 'STANFORDNLP_TEST_HOME'

# Global Variables
# test working directory base name must be stanfordnlp_test
TEST_DIR_BASE_NAME = 'stanfordnlp_test'
# BASIC mode is a small set of tests to run on Travis
TEST_MODE_BASIC = 'BASIC'
# FULL mode runs all tests, this would typically be on a GPU machine with sufficient resources
TEST_MODE_FULL = 'FULL'

# check the working dir is set and compliant
assert os.getenv(TEST_HOME_VAR) is not None, \
    f'Please set {TEST_HOME_VAR} environment variable for test working dir, base name must be: {TEST_DIR_BASE_NAME}'
TEST_WORKING_DIR = os.getenv(TEST_HOME_VAR)
assert os.path.basename(TEST_WORKING_DIR) == TEST_DIR_BASE_NAME, \
    f'Base name of test home dir must be: {TEST_DIR_BASE_NAME}'


def test_mode(tm):
    """ Check if current test mode matches test mode for a module, skip if there is a mismatch """
    if not os.getenv(TEST_MODE_VAR) == tm:
        pytest.skip(f'Test module: test_run_pipeline only runs in FULL mode (set {TEST_MODE_VAR} to {test_mode})',
                    allow_module_level=True)


# langauge resources
LANGUAGE_RESOURCES = {}

# French resources
FR_KEY = 'fr'
# regression file paths
FR_TEST_IN = f'{TEST_WORKING_DIR}/in/fr_gsd.test.txt'
FR_TEST_OUT = f'{TEST_WORKING_DIR}/out/fr_gsd.test.txt.out'
FR_TEST_GOLD_OUT = f'{TEST_WORKING_DIR}/out/fr_gsd.test.txt.out.gold'
# models
FR_MODELS_DIR = f'{TEST_WORKING_DIR}/fr_gsd_models'
FR_TOKENIZE_MODEL = "fr_gsd_tokenizer.pt"
FR_MWT_MODEL = "fr_gsd_mwt_expander.pt"
FR_POS_MODEL = "fr_gsd_tagger.pt"
FR_POS_PRETRAIN = "fr_gsd.pretrain.pt"
FR_LEMMA_MODEL = "fr_gsd_lemmatizer.pt"
FR_DEPPARSE_MODEL = "fr_gsd_parser.pt"
FR_DEPPARSE_PRETRAIN = "fr_gsd.pretrain.pt"
ALL_FR_MODELS = [f'{FR_MODELS_DIR}/{fr_m}' for fr_m in [FR_TOKENIZE_MODEL, FR_MWT_MODEL, FR_POS_MODEL, FR_POS_PRETRAIN,
                                                        FR_LEMMA_MODEL, FR_DEPPARSE_MODEL, FR_DEPPARSE_PRETRAIN]]


# utils for clean up
# only allow removal of dirs/files in this approved list
REMOVABLE_PATHS = ['fr_gsd_models', 'fr_gsd_tokenizer.pt', 'fr_gsd_mwt_expander.pt', 'fr_gsd_tagger.pt',
                   'fr_gsd.pretrain.pt', 'fr_gsd_lemmatizer.pt', 'fr_gsd_parser.pt', 'fr_gsd.test.txt',
                   'fr_gsd.test.txt.out', 'fr_gsd.test.txt.out.gold']


def safe_rm(path_to_rm):
    """
    Safely remove a directory of files or a file
    1.) check path exists, files are files, dirs are dirs
    2.) only remove things on approved list REMOVABLE_PATHS
    3.) assert no longer exists
    """
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
    if os.path.isdir(dir_to_rm):
        os.rmdir(dir_to_rm)
        assert not os.path.exists(dir_to_rm), f'Error removing: {dir_to_rm}'
