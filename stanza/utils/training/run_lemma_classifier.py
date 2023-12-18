import os

from stanza.models.lemma_classifier import train_model
from stanza.models.lemma_classifier import evaluate_models
from stanza.models.lemma_classifier.transformer_baseline import baseline_trainer

from stanza.utils.training import common
from stanza.utils.training.common import Mode, add_charlm_args, build_lemma_charlm_args, choose_lemma_charlm

def add_lemma_args(parser):
    add_charlm_args(parser)

def build_model_filename(paths, short_name, command_args, extra_args):
    # TODO: a more interesting name
    return os.path.join("saved_models", "lemmaclassifier", short_name + "_lemmaclassifier.pt")

def run_treebank(mode, paths, treebank, short_name,
                 temp_output_file, command_args, extra_args):
    short_language, dataset = short_name.split("_", 1)
    charlm_args = build_lemma_charlm_args(short_language, dataset, command_args.charlm)

    base_args = []
    if '--save_name' not in extra_args:
        base_args = ['--save_name', build_model_filename(paths, short_name, command_args, extra_args)]

    if mode == Mode.TRAIN:
        train_args = []
        if "--train_file" not in extra_args:
            train_file = os.path.join("data", "lemma_classifier", "%s.train.lemma" % short_name)
            train_args += ['--train_file', train_file]
        train_args = base_args + train_args + extra_args
        if '--model_type' in train_args:
            for idx, arg in enumerate(train_args):
                if arg == '--model_type' and idx + 1 < len(train_args):
                    model_type = train_args[idx + 1]
        if model_type == 'lstm':
            train_args = charlm_args + train_args
            train_model.main(train_args)
        else:
            baseline_trainer.main(train_args)

    if mode == Mode.SCORE_DEV:
        eval_args = []
        if "--eval_file" not in extra_args:
            eval_file = os.path.join("data", "lemma_classifier", "%s.dev.lemma" % short_name)
            eval_args += ['--eval_file', eval_file]
        eval_args = base_args + eval_args + charlm_args + extra_args
        evaluate_models.main(eval_args)

    if mode == Mode.SCORE_TEST:
        eval_args = []
        if "--eval_file" not in extra_args:
            eval_file = os.path.join("data", "lemma_classifier", "%s.test.lemma" % short_name)
            eval_args += ['--eval_file', eval_file]
        eval_args = base_args + eval_args + charlm_args + extra_args
        evaluate_models.main(eval_args)

def main():
    common.main(run_treebank, "lemma_classifier", "lemma_classifier", add_lemma_args, sub_argparse=train_model.build_argparse(), build_model_filename=build_model_filename, choose_charlm_method=choose_lemma_charlm)


if __name__ == '__main__':
    main()
