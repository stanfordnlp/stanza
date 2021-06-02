"""
Test the requirements functionality for processors
"""

import pytest
import stanza

from stanza.pipeline.core import PipelineRequirementsException
from stanza.pipeline.processor import ProcessorRequirementsException
from stanza.tests import *

pytestmark = pytest.mark.pipeline

def check_exception_vals(req_exception, req_exception_vals):
    """
    Check the values of a ProcessorRequirementsException against a dict of expected values.
    :param req_exception: the ProcessorRequirementsException to evaluate
    :param req_exception_vals: expected values for the ProcessorRequirementsException
    :return: None
    """
    assert isinstance(req_exception, ProcessorRequirementsException)
    assert req_exception.processor_type == req_exception_vals['processor_type']
    assert req_exception.processors_list == req_exception_vals['processors_list']
    assert req_exception.err_processor.requires == req_exception_vals['requires']


def test_missing_requirements():
    """
    Try to build several pipelines with bad configs and check thrown exceptions against gold exceptions.
    :return: None
    """
    # list of (bad configs, list of gold ProcessorRequirementsExceptions that should be thrown) pairs
    bad_config_lists = [
        # missing tokenize
        (
            # input config
            {'processors': 'pos,depparse', 'dir': TEST_MODELS_DIR, 'lang': 'en'},
            # 2 expected exceptions
            [
                {'processor_type': 'POSProcessor', 'processors_list': ['pos', 'depparse'], 'provided_reqs': set([]),
                 'requires': set(['tokenize'])},
                {'processor_type': 'DepparseProcessor', 'processors_list': ['pos', 'depparse'],
                 'provided_reqs': set([]), 'requires': set(['tokenize','pos', 'lemma'])}
            ]
        ),
        # no pos when lemma_pos set to True; for english mwt should not be included in the loaded processor list
        (
            # input config
            {'processors': 'tokenize,mwt,lemma', 'dir': TEST_MODELS_DIR, 'lang': 'en', 'lemma_pos': True},
            # 1 expected exception
            [
                {'processor_type': 'LemmaProcessor', 'processors_list': ['tokenize', 'lemma'],
                 'provided_reqs': set(['tokenize', 'mwt']), 'requires': set(['tokenize', 'pos'])}
            ]
        )
    ]
    # try to build each bad config, catch exceptions, check against gold
    pipeline_fails = 0
    for bad_config, gold_exceptions in bad_config_lists:
        try:
            stanza.Pipeline(**bad_config)
        except PipelineRequirementsException as e:
            pipeline_fails += 1
            assert isinstance(e, PipelineRequirementsException)
            assert len(e.processor_req_fails) == len(gold_exceptions)
            for processor_req_e, gold_exception in zip(e.processor_req_fails,gold_exceptions):
                # compare the thrown ProcessorRequirementsExceptions against gold
                check_exception_vals(processor_req_e, gold_exception)
    # check pipeline building failed twice
    assert pipeline_fails == 2


