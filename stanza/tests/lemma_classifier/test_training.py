import glob
import os

import pytest

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

from stanza.models.lemma_classifier import train_lstm_model
from stanza.models.lemma_classifier import train_transformer_model
from stanza.models.lemma_classifier.base_model import LemmaClassifier
from stanza.models.lemma_classifier.evaluate_models import evaluate_model

from stanza.tests import TEST_WORKING_DIR
from stanza.tests.lemma_classifier.test_data_preparation import convert_english_dataset

@pytest.fixture(scope="module")
def pretrain_file():
    return f'{TEST_WORKING_DIR}/in/tiny_emb.pt'

def test_train_lstm(tmp_path, pretrain_file):
    converted_files = convert_english_dataset(tmp_path)

    save_name = str(tmp_path / 'lemma.pt')

    train_file = converted_files[0]
    eval_file = converted_files[1]
    train_args = ['--wordvec_pretrain_file', pretrain_file,
                  '--save_name', save_name,
                  '--train_file', train_file,
                  '--eval_file', eval_file]
    trainer = train_lstm_model.main(train_args)

    evaluate_model(trainer.model, eval_file)
    # test that loading the model works
    model = LemmaClassifier.load(save_name, None)

def test_train_transformer(tmp_path, pretrain_file):
    converted_files = convert_english_dataset(tmp_path)

    save_name = str(tmp_path / 'lemma.pt')

    train_file = converted_files[0]
    eval_file = converted_files[1]
    train_args = ['--bert_model', 'hf-internal-testing/tiny-bert',
                  '--save_name', save_name,
                  '--train_file', train_file,
                  '--eval_file', eval_file]
    trainer = train_transformer_model.main(train_args)

    evaluate_model(trainer.model, eval_file)

    # test that loading the model works
    model = LemmaClassifier.load(save_name, None)
