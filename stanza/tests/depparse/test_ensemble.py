"""
Test the ensembles by training two models, building an ensemble
out of them, then running the ensemble on a couple sentences.

The actual results are not checked, and the model does not train
enough to actually perform well; this is just a check that the
training and testing functions correctly.

Uses a couple sentences of UD_English-EWT as training/dev data

Tests both graph and transition versions of the parser
"""

import os
import pytest

import torch

from stanza.models import parser
from stanza.models.common import pretrain
from stanza.models.depparse.ensemble import build_ensemble
from stanza.models.depparse.model import EnsembleGraphParser
from stanza.models.depparse.trainer import Trainer
from stanza.models.depparse.transition.model import EnsembleTransitionParser
from stanza.tests import TEST_WORKING_DIR
from stanza.tests.depparse.parser_training import run_training
from stanza.tests.depparse.test_parser import TRAIN_DATA, DEV_DATA

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

class TestEnsemble:
    @pytest.fixture(scope="class")
    def wordvec_pretrain_file(self):
        return f'{TEST_WORKING_DIR}/in/tiny_emb.pt'

    def run_ensemble_test(self, tmp_path, wordvec_pretrain_file, model_type):
        """
        Simple test of a few 'epochs' of tagger training
        """
        pt = pretrain.Pretrain(wordvec_pretrain_file)
        save_names = ["p%d.pt" % i for i in range(2)]
        trainers = [run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, model_type=model_type, save_name=save_name) for save_name in save_names]
        model_names = [str(tmp_path / save_name) for save_name in save_names]

        trainer, _ = build_ensemble(trainers[0].args, pt, model_names, foundation_cache=None, device=None)

        ensemble_name = "ensemble.pt"
        trainer.save(ensemble_name)

        # this was created by run_training
        dev_path = str(tmp_path / "dev.conllu")

        args = ["--wordvec_pretrain_file", wordvec_pretrain_file,
                "--eval_file", dev_path,
                "--save_dir", str(tmp_path),
                "--save_name", ensemble_name,
                "--mode", "predict"]
        # this should function... we don't care about the results
        parser.main(args)
        return trainer

    def test_graph_ensemble(self, tmp_path, wordvec_pretrain_file):
        trainer = self.run_ensemble_test(tmp_path, wordvec_pretrain_file, model_type="graph")
        assert isinstance(trainer.model, EnsembleGraphParser)

    def test_transition_ensemble(self, tmp_path, wordvec_pretrain_file):
        trainer = self.run_ensemble_test(tmp_path, wordvec_pretrain_file, model_type="transition")
        assert isinstance(trainer.model, EnsembleTransitionParser)
