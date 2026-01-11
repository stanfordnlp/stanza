"""
Shared pytest fixtures and configuration
"""

import pytest
from morphseg import MorphemeSegmenter

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

@pytest.fixture(scope="session")
def english_segmenter():
    """
    Load English segmenter once for the entire test session
    """
    return MorphemeSegmenter('en')


@pytest.fixture(scope="session")
def all_segmenters():
    """
    Load all supported language segmenters
    """
    segmenters = {}
    for lang in MorphemeSegmenter.PRETRAINED_MODEL_LANGS:
        segmenters[lang] = MorphemeSegmenter(lang)
    return segmenters


def pytest_configure(config):
    """
    Custom pytest configuration
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "multilingual: marks tests that test multiple languages"
    )
