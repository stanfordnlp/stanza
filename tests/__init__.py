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


def test_mode(test_mode):
    """ Check if current test mode matches test mode for a module, skip if there is a mismatch """
    if not os.getenv(TEST_MODE_VAR) == test_mode:
        pytest.skip(f'Test module: test_run_pipeline only runs in FULL mode (set {TEST_MODE_VAR} to {test_mode})',
                    allow_module_level=True)

