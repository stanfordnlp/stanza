"""
Test a couple basic functions - load & save an existing model
"""

import pytest

import glob
import os
import tempfile

from stanza.models.lemma import trainer
from stanza.tests import *

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

@pytest.fixture(scope="module")
def english_model():
    models_path = os.path.join(TEST_MODELS_DIR, "en", "lemma", "*")
    models = glob.glob(models_path)
    # we expect at least one English model downloaded for the tests
    assert len(models) >= 1
    model_file = models[0]
    return trainer.Trainer(model_file=model_file)

def test_load_model(english_model):
    """
    Does nothing, just tests that loading works
    """

def test_save_load_model(english_model):
    """
    Load, save, and load again
    """
    with tempfile.TemporaryDirectory() as tempdir:
        save_file = os.path.join(tempdir, "resaved", "lemma.pt")
        english_model.save(save_file)
        reloaded = trainer.Trainer(model_file=save_file)
