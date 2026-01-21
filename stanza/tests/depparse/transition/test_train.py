"""
Train the transition parser on a very small amount of training data

Uses a couple sentences of UD_English-EWT as training/dev data
"""

import os
import pytest

import torch

from stanza.models.common import pretrain
from stanza.models.depparse.trainer import Trainer
from stanza.tests import TEST_WORKING_DIR
from stanza.tests.depparse.parser_training import run_training
from stanza.tests.depparse.test_parser import TRAIN_DATA, DEV_DATA

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

class TestParser:
    @pytest.fixture(scope="class")
    def wordvec_pretrain_file(self):
        return f'{TEST_WORKING_DIR}/in/tiny_emb.pt'

    def test_train(self, tmp_path, wordvec_pretrain_file):
        """
        Simple test of a few 'epochs' of tagger training
        """
        trainer = run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, model_type="transition")
        assert trainer.args['model_type'] == 'transition'

    def test_train_bilstm_merge(self, tmp_path, wordvec_pretrain_file):
        """
        Test training with the bilstm merge method
        """
        run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, model_type="transition", extra_args=['--transition_subtree_combination', 'BILSTM'])

    def test_train_lstm_merge(self, tmp_path, wordvec_pretrain_file):
        """
        Test training with the lstm merge method
        """
        run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, model_type="transition", extra_args=['--transition_subtree_combination', 'LSTM'])

