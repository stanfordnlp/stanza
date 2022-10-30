import os
import torch

import pytest

from stanza.models.constituency import trans_lm

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

TRAIN_TEXT = "This is a test"
TEST_TEXT = "This is a zz zz"

def test_small_end_to_end(tmp_path):
    train_filename = tmp_path / "train.lm"
    dev_filename = tmp_path / "dev.lm"
    test_filename = tmp_path / "test.lm"

    with open(train_filename, "w") as fout:
        for i in range(25):
            fout.write(TRAIN_TEXT)
            fout.write("\n")
    with open(dev_filename, "w") as fout:
        for i in range(15):
            fout.write(TRAIN_TEXT)
            fout.write("\n")
    with open(test_filename, "w") as fout:
        for i in range(5):
            fout.write(TEST_TEXT)
            fout.write("\n")

    args = ["--train_file", str(train_filename),
            "--dev_file", str(dev_filename),
            "--test_file", str(test_filename),
            "--lang", "test",
            "--d_embedding", "20",
            "--d_hid", "20",
            "--n_heads", "4",
            "--n_layers", "2",
            "--save_dir", str(tmp_path),
            "--save_name", "model.pt"]

    model = trans_lm.main(args)
    model.eval()
    score_test = model.score([TEST_TEXT])
    score_train = model.score([TRAIN_TEXT])

    score_both = model.score([TRAIN_TEXT, TEST_TEXT])
    score_reversed = model.score([TEST_TEXT, TRAIN_TEXT])

    assert torch.allclose(score_test, score_both[1])
    assert torch.allclose(score_test, score_reversed[0])
    assert torch.allclose(score_train, score_both[0])
    assert torch.allclose(score_train, score_reversed[1])

    model_filename = tmp_path / "model.pt"
    assert os.path.exists(model_filename)

    loaded = trans_lm.TransformerModel.load(model_filename).to(model.device())
    loaded.eval()

    assert model.vocab.get_itos() == loaded.vocab.get_itos()
    for (model_name, model_param), (loaded_name, loaded_param) in zip(model.named_parameters(), loaded.named_parameters()):
        assert model_name == loaded_name
        assert torch.allclose(model_param, loaded_param)

    loaded_score_both = loaded.score([TRAIN_TEXT, TEST_TEXT])
    assert torch.allclose(score_both, loaded_score_both)
