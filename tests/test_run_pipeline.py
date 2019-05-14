"""
Tests for the run_pipeline.py script, also serves as integration test
"""

import re
import subprocess

from datetime import datetime
from tests import *

DOWNLOAD_TEST_DIR = f'{TEST_WORKING_DIR}/download'
FR_MODELS_DOWNLOAD_DIR = f'{DOWNLOAD_TEST_DIR}/{FR_SHORTHAND}_models'


def test_fr_pipeline():
    # check input files present
    assert os.path.exists(FR_TEST_IN), f'Missing test input file: {FR_TEST_IN}'
    assert os.path.exists(FR_TEST_GOLD_OUT), f'Missing test gold output file: {FR_TEST_GOLD_OUT}'
    # verify models not downloaded and output file doesn't exist
    safe_rm(FR_TEST_OUT)
    safe_rm(FR_MODELS_DOWNLOAD_DIR)
    # run french pipeline command and check results
    fr_pipeline_cmd = \
        f"python -m stanfordnlp.run_pipeline -l fr -d {DOWNLOAD_TEST_DIR} --force-download -o {FR_TEST_OUT} " \
        f"{FR_TEST_IN}"
    subprocess.call(fr_pipeline_cmd, shell=True)
    # cleanup
    # log this test run's final output
    if os.path.exists(FR_TEST_OUT):
        curr_timestamp = re.sub(' ', '-', str(datetime.now()))
        os.rename(FR_TEST_OUT, f'{FR_TEST_OUT}-{curr_timestamp}')
    safe_rm(FR_MODELS_DOWNLOAD_DIR)
    assert open(FR_TEST_GOLD_OUT).read() == open(f'{FR_TEST_OUT}-{curr_timestamp}').read(), \
        f'Test failure: output does not match gold'

