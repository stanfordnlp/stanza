import os

from stanza.models.lemma_classifier import evaluate_models
from stanza.models.lemma_classifier import train_lstm_model
from stanza.models.lemma_classifier import train_transformer_model
from stanza.models.lemma_classifier.constants import ModelType

from stanza.resources.default_packages import default_pretrains, TRANSFORMERS
from stanza.utils.training import common
from stanza.utils.training.common import Mode, add_charlm_args, build_lemma_charlm_args, choose_lemma_charlm, find_wordvec_pretrain

def add_lemma_args(parser):
    add_charlm_args(parser)

    parser.add_argument('--model_type', default=ModelType.LSTM, type=lambda x: ModelType[x.upper()],
                        help='Model type to use.  {}'.format(", ".join(x.name for x in ModelType)))

def build_model_filename(paths, short_name, command_args, extra_args):
    return os.path.join("saved_models", "lemma_classifier", short_name + "_lemma_classifier.pt")

def run_treebank(mode, paths, treebank, short_name,
                 temp_output_file, command_args, extra_args):
    short_language, dataset = short_name.split("_", 1)

    base_args = []
    if '--save_name' not in extra_args:
        base_args += ['--save_name', build_model_filename(paths, short_name, command_args, extra_args)]

    embedding_args = build_lemma_charlm_args(short_language, dataset, command_args.charlm)
    if '--wordvec_pretrain_file' not in extra_args:
        wordvec_pretrain = find_wordvec_pretrain(short_language, default_pretrains, {}, dataset)
        embedding_args += ["--wordvec_pretrain_file", wordvec_pretrain]

    bert_args = []
    if command_args.model_type is ModelType.TRANSFORMER:
        if '--bert_model' not in extra_args:
            if short_language in TRANSFORMERS:
                bert_args = ['--bert_model', TRANSFORMERS.get(short_language)]
            else:
                raise ValueError("--bert_model not specified, so cannot figure out which transformer to use for language %s" % short_language)

    extra_train_args = []
    if command_args.force:
        extra_train_args.append('--force')

    if mode == Mode.TRAIN:
        train_args = []
        if "--train_file" not in extra_args:
            train_file = os.path.join("data", "lemma_classifier", "%s.train.lemma" % short_name)
            train_args += ['--train_file', train_file]
        if "--eval_file" not in extra_args:
            eval_file = os.path.join("data", "lemma_classifier", "%s.dev.lemma" % short_name)
            train_args += ['--eval_file', eval_file]
        train_args = base_args + train_args + extra_args + extra_train_args

        if command_args.model_type == ModelType.LSTM:
            train_args = embedding_args + train_args
            train_lstm_model.main(train_args)
        else:
            model_type_args = ["--model_type", command_args.model_type.name.lower()]
            train_args = bert_args + model_type_args + train_args
            train_transformer_model.main(train_args)

    if mode == Mode.SCORE_DEV or mode == Mode.TRAIN:
        eval_args = []
        if "--eval_file" not in extra_args:
            eval_file = os.path.join("data", "lemma_classifier", "%s.dev.lemma" % short_name)
            eval_args += ['--eval_file', eval_file]
        model_type_args = ["--model_type", command_args.model_type.name.lower()]
        eval_args = bert_args + model_type_args + base_args + eval_args + embedding_args + extra_args
        evaluate_models.main(eval_args)

    if mode == Mode.SCORE_TEST or mode == Mode.TRAIN:
        eval_args = []
        if "--eval_file" not in extra_args:
            eval_file = os.path.join("data", "lemma_classifier", "%s.test.lemma" % short_name)
            eval_args += ['--eval_file', eval_file]
        model_type_args = ["--model_type", command_args.model_type.name.lower()]
        eval_args = bert_args + model_type_args + base_args + eval_args + embedding_args + extra_args
        evaluate_models.main(eval_args)

def main(args=None):
    common.main(run_treebank, "lemma_classifier", "lemma_classifier", add_lemma_args, sub_argparse=train_lstm_model.build_argparse(), build_model_filename=build_model_filename, choose_charlm_method=choose_lemma_charlm, args=args)


if __name__ == '__main__':
    main()
