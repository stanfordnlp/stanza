"""
Refactored method to run a short training
"""

import os
import zipfile

from stanza.models import parser
from stanza.models.common import pretrain
from stanza.models.depparse.trainer import Trainer

def run_training(tmp_path, wordvec_pretrain_file, train_text, dev_text, augment_nopunct=False, extra_args=None, zip_train_data=False, model_type="graph", save_name="test_parser.pt"):
    """
    Run the training for a few iterations, load & return the model
    """
    train_file = str(tmp_path / "train.zip") if zip_train_data else str(tmp_path / "train.conllu")
    dev_file = str(tmp_path / "dev.conllu")
    pred_file = str(tmp_path / "pred.conllu")

    save_file = str(tmp_path / save_name)

    if zip_train_data:
        with zipfile.ZipFile(train_file, "w") as zout:
            with zout.open('train.conllu', 'w') as fout:
                fout.write(train_text.encode())
    else:
        with open(train_file, "w", encoding="utf-8") as fout:
            fout.write(train_text)

    with open(dev_file, "w", encoding="utf-8") as fout:
        fout.write(dev_text)

    args = ["--wordvec_pretrain_file", wordvec_pretrain_file,
            "--train_file", train_file,
            "--eval_file", dev_file,
            "--output_file", pred_file,
            "--log_step", "3",
            "--eval_interval", "6",
            "--max_steps", "18",
            "--shorthand", "en_test",
            "--save_dir", str(tmp_path),
            "--save_name", save_name,
            # in case we are doing a bert test
            "--bert_start_finetuning", "10",
            "--bert_warmup_steps", "10",
            "--lang", "en",
            "--model_type", model_type]
    if not augment_nopunct:
        args.extend(["--augment_nopunct", "0.0"])
    if extra_args is not None:
        args = args + extra_args
    trainer, _ = parser.main(args)

    assert os.path.exists(save_file)
    pt = pretrain.Pretrain(wordvec_pretrain_file)
    # test loading the saved model
    saved_model = Trainer.load(filename=save_file, pretrain=pt)
    return trainer

