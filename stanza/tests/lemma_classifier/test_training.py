import glob
import os

import pytest

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

from stanza.models.lemma_classifier import train_model
from stanza.models.lemma_classifier.transformer_baseline import baseline_trainer

from stanza.tests import TEST_WORKING_DIR
from stanza.tests.lemma_classifier.test_data_preparation import convert_english_dataset

@pytest.fixture(scope="module")
def pretrain_file():
    return f'{TEST_WORKING_DIR}/in/tiny_emb.pt'

def test_train_lstm(tmp_path, pretrain_file):
    converted_files = convert_english_dataset(tmp_path)

    save_name = str(tmp_path / 'lemma.pt')

    train_args = ['--wordvec_pretrain_file', pretrain_file,
                  '--save_name', save_name,
                  '--train_file', converted_files[0],
                  '--eval_file', converted_files[1]]
    train_model.main(train_args)

def test_train_transformer(tmp_path, pretrain_file):
    converted_files = convert_english_dataset(tmp_path)

    save_name = str(tmp_path / 'lemma.pt')

    train_args = ['--bert_model', 'hf-internal-testing/tiny-bert',
                  '--save_name', save_name,
                  '--train_file', converted_files[0],
                  '--eval_file', converted_files[1]]
    baseline_trainer.main(train_args)

