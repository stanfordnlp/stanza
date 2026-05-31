import json
import logging
import os
import warnings

import pytest
import torch

from peft import get_peft_model_state_dict

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

from stanza.models import ner_tagger
from stanza.models.ner.trainer import Trainer
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

EN_TRAIN_2TAG = """
Chris B-PERSON B-PER
Manning E-PERSON E-PER
is O O
a O O
good O O
man O O
. O O

He O O
works O O
in O O
Stanford B-ORG B-ORG
University E-ORG B-ORG
. O O
""".strip().replace(" ", "\t")

EN_TRAIN_2TAG_EMPTY2 = """
Chris B-PERSON -
Manning E-PERSON -
is O -
a O -
good O -
man O -
. O -

He O -
works O -
in O -
Stanford B-ORG -
University E-ORG -
. O -
""".strip().replace(" ", "\t")

EN_DEV_2TAG = """
Chris B-PERSON B-PER
Manning E-PERSON E-PER
is O O
part O O
of O O
Computer B-ORG B-ORG
Science E-ORG E-ORG
""".strip().replace(" ", "\t")

@pytest.fixture(scope="module")
def pretrain_file():
    return f'{TEST_WORKING_DIR}/in/tiny_emb.pt'

def write_temp_file(filename, bio_data):
    bio_filename = os.path.splitext(filename)[0] + ".bio"
    with open(bio_filename, "w", encoding="utf-8") as fout:
        fout.write(bio_data)
    process_dataset(bio_filename, filename)

def write_temp_2tag(filename, bio_data):
    doc = []
    sentences = bio_data.split("\n\n")
    for sentence in sentences:
        doc.append([])
        for word in sentence.split("\n"):
            text, tags = word.split("\t", maxsplit=1)
            doc[-1].append({
                "text": text,
                "multi_ner": tags.split()
            })

    with open(filename, "w", encoding="utf-8") as fout:
        json.dump(doc, fout)

def get_args(tmp_path, pretrain_file, train_json, dev_json, *extra_args):
    save_dir = tmp_path / "models"
    args = ["--data_dir", str(tmp_path),
            "--wordvec_pretrain_file", pretrain_file,
            "--train_file", str(train_json),
            "--eval_file", str(dev_json),
            "--shorthand", "en_test",
            "--max_steps", "100",
            "--eval_interval", "40",
            "--save_dir", str(save_dir)]

    args = args + list(extra_args)
    return args

def run_two_tag_training(pretrain_file, tmp_path, *extra_args, train_data=EN_TRAIN_2TAG):
    train_json = tmp_path / "en_test.train.json"
    write_temp_2tag(train_json, train_data)

    dev_json = tmp_path / "en_test.dev.json"
    write_temp_2tag(dev_json, EN_DEV_2TAG)

    args = get_args(tmp_path, pretrain_file, train_json, dev_json, *extra_args)
    return ner_tagger.main(args)

def test_basic_two_tag_training(pretrain_file, tmp_path):
    trainer = run_two_tag_training(pretrain_file, tmp_path)
    assert len(trainer.model.tag_clfs) == 2
    assert len(trainer.model.crits) == 2
    assert len(trainer.vocab['tag'].lens()) == 2

def test_two_tag_training_backprop(pretrain_file, tmp_path):
    """
    Test that the training is backproping both tags

    We can do this by using the "finetune" mechanism and verifying
    that the output tensors are different
    """
    trainer = run_two_tag_training(pretrain_file, tmp_path)

    # first, need to save the final model before restarting
    # (alternatively, could reload the final checkpoint)
    trainer.save(os.path.join(trainer.args['save_dir'], trainer.args['save_name']))
    new_trainer = run_two_tag_training(pretrain_file, tmp_path, "--finetune")

    assert len(trainer.model.tag_clfs) == 2
    assert len(new_trainer.model.tag_clfs) == 2
    for old_clf, new_clf in zip(trainer.model.tag_clfs, new_trainer.model.tag_clfs):
        assert not torch.allclose(old_clf.weight, new_clf.weight)

def test_two_tag_training_c2_backprop(pretrain_file, tmp_path):
    """
    Test that the training is backproping only one tag if one column is blank

    We can do this by using the "finetune" mechanism and verifying
    that the output tensors are different in just the first column
    """
    trainer = run_two_tag_training(pretrain_file, tmp_path)

    # first, need to save the final model before restarting
    # (alternatively, could reload the final checkpoint)
    trainer.save(os.path.join(trainer.args['save_dir'], trainer.args['save_name']))
    new_trainer = run_two_tag_training(pretrain_file, tmp_path, "--finetune", train_data=EN_TRAIN_2TAG_EMPTY2)

    assert len(trainer.model.tag_clfs) == 2
    assert len(new_trainer.model.tag_clfs) == 2
    assert not torch.allclose(trainer.model.tag_clfs[0].weight, new_trainer.model.tag_clfs[0].weight)
    assert torch.allclose(trainer.model.tag_clfs[1].weight, new_trainer.model.tag_clfs[1].weight)

def test_connected_two_tag_training(pretrain_file, tmp_path):
    trainer = run_two_tag_training(pretrain_file, tmp_path, "--connect_output_layers")
    assert len(trainer.model.tag_clfs) == 2
    assert len(trainer.model.crits) == 2
    assert len(trainer.vocab['tag'].lens()) == 2

    # this checks that with the connected output layers,
    # the second output layer has its size increased
    # by the number of tags known to the first output layer
    assert trainer.model.tag_clfs[1].weight.shape[1] == trainer.vocab['tag'].lens()[0] + trainer.model.tag_clfs[0].weight.shape[1]

def run_training(pretrain_file, tmp_path, *extra_args):
    train_json = tmp_path / "en_test.train.json"
    write_temp_file(train_json, EN_TRAIN_BIO)

    dev_json = tmp_path / "en_test.dev.json"
    write_temp_file(dev_json, EN_DEV_BIO)

    args = get_args(tmp_path, pretrain_file, train_json, dev_json, *extra_args)
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

def model_file_has_bert(filename):
    checkpoint = torch.load(filename, lambda storage, loc: storage, weights_only=True)
    return any(x.startswith("bert_model.") for x in checkpoint['model'].keys())

def test_with_bert(pretrain_file, tmp_path):
    trainer = run_training(pretrain_file, tmp_path, '--bert_model', 'hf-internal-testing/tiny-bert')
    model_file = os.path.join(trainer.args['save_dir'], trainer.args['save_name'])
    assert not model_file_has_bert(model_file)

def test_with_bert_finetune(pretrain_file, tmp_path):
    trainer = run_training(pretrain_file, tmp_path, '--bert_model', 'hf-internal-testing/tiny-bert', '--bert_finetune')
    model_file = os.path.join(trainer.args['save_dir'], trainer.args['save_name'])
    assert model_file_has_bert(model_file)

    foo_save_filename = os.path.join(tmp_path, "foo_" + trainer.args['save_name'])
    bar_save_filename = os.path.join(tmp_path, "bar_" + trainer.args['save_name'])
    trainer.save(foo_save_filename)
    assert model_file_has_bert(foo_save_filename)

    # TODO: technically this should still work if we turn off bert finetuning when reloading
    reloaded_trainer = Trainer(args=trainer.args, model_file=foo_save_filename)
    reloaded_trainer.save(bar_save_filename)
    assert model_file_has_bert(bar_save_filename)

def test_with_peft_finetune(pretrain_file, tmp_path, monkeypatch):
    """
    Verify that PEFT LoRA training moves the adapter weights but not the
    frozen base transformer weights.

    We monkeypatch Trainer.__init__ to capture the model's state immediately
    after construction, before any training steps.  This avoids any dependency
    on random seeds producing identical initialization across two separate
    Trainer constructions.
    """
    initial_lora = {}
    initial_frozen_weights = {}
    original_init = Trainer.__init__

    def capturing_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # only capture on the first construction (the training run),
        # not on any subsequent loads
        if not initial_lora and hasattr(self.model, 'bert_model') and self.model.peft_name:
            for k, v in get_peft_model_state_dict(self.model.bert_model, adapter_name=self.model.peft_name).items():
                initial_lora[k] = v.clone()
            for name, param in self.model.bert_model.named_parameters():
                if not param.requires_grad:
                    initial_frozen_weights[name] = param.data.clone()
                    break  # one frozen param is sufficient

    monkeypatch.setattr(Trainer, '__init__', capturing_init)

    trainer = run_training(pretrain_file, tmp_path, '--bert_model', 'hf-internal-testing/tiny-bert', '--use_peft')
    model_file = os.path.join(trainer.args['save_dir'], trainer.args['save_name'])

    assert initial_lora, "Monkeypatch did not capture initial LoRA weights — check that Trainer.__init__ was called"
    assert initial_frozen_weights, "Monkeypatch did not capture any frozen base weights"

    # --- Saved checkpoint checks ---
    checkpoint = torch.load(model_file, lambda storage, loc: storage, weights_only=True)
    assert 'bert_lora' in checkpoint
    assert not any(x.startswith("bert_model.") for x in checkpoint['model'].keys())

    # --- LoRA weights should have moved ---
    trained_lora = get_peft_model_state_dict(trainer.model.bert_model, adapter_name=trainer.model.peft_name)
    assert any(
        not torch.allclose(initial_lora[k], trained_lora[k])
        for k in initial_lora
    ), "Expected at least one LoRA weight to change during training, but all were identical"

    # --- Frozen base weights should be unchanged ---
    frozen_name, initial_frozen = next(iter(initial_frozen_weights.items()))
    trained_frozen = trainer.model.bert_model.state_dict()[frozen_name]
    assert torch.allclose(initial_frozen, trained_frozen), \
        f"Base transformer weight '{frozen_name}' changed during PEFT training, but it should be frozen"

    # --- Test loading ---
    reloaded_trainer = Trainer(args=trainer.args, model_file=model_file)
