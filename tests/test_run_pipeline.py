"""
Tests for the run_pipeline.py script, also serves as integration test
"""

import re
import subprocess

from datetime import datetime
from tests import *

# check testing environment, only run these tests in FULL mode
test_mode(TEST_MODE_FULL)

# French pipeline paths
# files
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
ALL_FRENCH_MODELS = [f'{FR_MODELS_DIR}/{fr_m}' for fr_m in [FR_TOKENIZE_MODEL, FR_MWT_MODEL, FR_POS_MODEL,
                                                            FR_POS_PRETRAIN, FR_LEMMA_MODEL, FR_DEPPARSE_MODEL,
                                                            FR_DEPPARSE_PRETRAIN]]


def test_fr_pipeline():
    # check input files present
    assert os.path.exists(FR_TEST_IN), f'Missing test input file: {FR_TEST_IN}'
    assert os.path.exists(FR_TEST_GOLD_OUT), f'Missing test gold output file: {FR_TEST_GOLD_OUT}'
    # verify models not downloaded and output file doesn't exist
    if os.path.exists(FR_TEST_OUT):
        os.remove(FR_TEST_OUT)
    assert not os.path.exists(FR_TEST_OUT), f'Error removing: {FR_TEST_OUT}'
    for fr_model in ALL_FRENCH_MODELS:
        if os.path.exists(fr_model):
            os.remove(fr_model)
        assert not os.path.exists(fr_model), f'Error removing: {fr_model}'
    if os.path.exists(FR_MODELS_DIR):
        os.rmdir(FR_MODELS_DIR)
    assert not os.path.exists(FR_MODELS_DIR), f'Error removing: {FR_MODELS_DIR}'
    # run french pipeline command and check results
    fr_pipeline_cmd = \
        f"python -m stanfordnlp.run_pipeline -l fr -d {TEST_WORKING_DIR} --force-download -o {FR_TEST_OUT} {FR_TEST_IN}"
    subprocess.call(fr_pipeline_cmd, shell=True)
    assert open(FR_TEST_GOLD_OUT).read() == open(FR_TEST_OUT).read(), f'Test failure: output does not match gold'
    # cleanup
    if os.path.exists(FR_TEST_OUT):
        curr_timestamp = re.sub(' ', '-', str(datetime.now()))
        os.rename(FR_TEST_OUT, f'{FR_TEST_OUT}-{curr_timestamp}')
    for fr_model in ALL_FRENCH_MODELS:
        if os.path.exists(fr_model):
            os.remove(fr_model)
    if os.path.exists(FR_MODELS_DIR):
        os.rmdir(FR_MODELS_DIR)
