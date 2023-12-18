import os

from stanza.models.lemma_classifier import train_model
from stanza.models.lemma_classifier import evaluate_models
from stanza.models.lemma_classifier.constants import ModelType
from stanza.models.lemma_classifier.transformer_baseline import baseline_trainer

from stanza.utils.training import common
from stanza.utils.training.common import Mode, add_charlm_args, build_lemma_charlm_args, choose_lemma_charlm

def add_lemma_args(parser):
    add_charlm_args(parser)

    parser.add_argument('--model_type', default=ModelType.LSTM, type=lambda x: ModelType[x.upper()],
                        help='Model type to use.  {}'.format(", ".join(x.name for x in ModelType)))

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
        if command_args.model_type == ModelType.LSTM:
            train_args = charlm_args + train_args
            train_model.main(train_args)
        else:
            model_type_args = ["--model_type", command_args.model_type.name.lower()]
            train_args = model_type_args + train_args
            baseline_trainer.main(train_args)

    if mode == Mode.SCORE_DEV:
        eval_args = []
        if "--eval_file" not in extra_args:
            eval_file = os.path.join("data", "lemma_classifier", "%s.dev.lemma" % short_name)
            eval_args += ['--eval_file', eval_file]
        model_type_args = ["--model_type", command_args.model_type.name.lower()]
        eval_args = model_type_args + base_args + eval_args + charlm_args + extra_args
        evaluate_models.main(eval_args)

    if mode == Mode.SCORE_TEST:
        eval_args = []
        if "--eval_file" not in extra_args:
            eval_file = os.path.join("data", "lemma_classifier", "%s.test.lemma" % short_name)
            eval_args += ['--eval_file', eval_file]
        model_type_args = ["--model_type", command_args.model_type.name.lower()]
        eval_args = model_type_args + base_args + eval_args + charlm_args + extra_args
        evaluate_models.main(eval_args)

def main():
    common.main(run_treebank, "lemma_classifier", "lemma_classifier", add_lemma_args, sub_argparse=train_model.build_argparse(), build_model_filename=build_model_filename, choose_charlm_method=choose_lemma_charlm)


if __name__ == '__main__':
    main()
