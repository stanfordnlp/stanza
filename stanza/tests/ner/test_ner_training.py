import logging
import os
import warnings

import pytest
import torch

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

from stanza.models import ner_tagger
from stanza.tests import TEST_WORKING_DIR
from stanza.utils.datasets.ner.prepare_ner_file import process_dataset

logger = logging.getLogger('stanza')

EN_TRAIN_BIO = """
Chris B-PERSON
Manning E-PERSON
is O
a O
good O
man O
. O

He O
works O
in O
Stanford B-ORG
University E-ORG
. O
""".lstrip().replace(" ", "\t")

EN_DEV_BIO = """
Chris B-PERSON
Manning E-PERSON
is O
part O
of O
Computer B-ORG
Science E-ORG
""".lstrip().replace(" ", "\t")

@pytest.fixture(scope="module")
def pretrain_file():
    return f'{TEST_WORKING_DIR}/in/tiny_emb.pt'

def write_temp_file(filename, bio_data):
    bio_filename = os.path.splitext(filename)[0] + ".bio"
    with open(bio_filename, "w", encoding="utf-8") as fout:
        fout.write(bio_data)
    process_dataset(bio_filename, filename)

def run_training(pretrain_file, tmp_path, *extra_args):
    train_json = tmp_path / "en_test.train.json"
    write_temp_file(train_json, EN_TRAIN_BIO)

    dev_json = tmp_path / "en_test.dev.json"
    write_temp_file(dev_json, EN_DEV_BIO)

    save_dir = tmp_path / "models"

    logger.error("ERROR")
    logger.warning("WARNING")
    args = ["--data_dir", str(tmp_path),
            "--wordvec_pretrain_file", pretrain_file,
            "--train_file", str(train_json),
            "--eval_file", str(dev_json),
            "--lang", "en",
            "--shorthand", "en_test",
            "--max_steps", "100",
            "--save_dir", str(save_dir)]
    args = args + list(extra_args)

    return ner_tagger.main(args)


def test_train_model_gpu(pretrain_file, tmp_path):
    """
    Briefly train an NER model (no expectation of correctness) and check that it is on the GPU
    """
    trainer = run_training(pretrain_file, tmp_path)
    if not torch.cuda.is_available():
        warnings.warn("Cannot check that the NER model is on the GPU, since GPU is not available")
        return

    model = trainer.model
    device = next(model.parameters()).device
    assert str(device).startswith("cuda")


def test_train_model_cpu(pretrain_file, tmp_path):
    """
    Briefly train an NER model (no expectation of correctness) and check that it is on the GPU
    """
    trainer = run_training(pretrain_file, tmp_path, "--cpu")

    model = trainer.model
    device = next(model.parameters()).device
    assert str(device).startswith("cpu")


def test_with_bert(pretrain_file, tmp_path):
    run_training(pretrain_file, tmp_path, '--bert_model', 'hf-internal-testing/tiny-bert')

