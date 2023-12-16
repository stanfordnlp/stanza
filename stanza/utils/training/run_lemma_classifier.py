import os

from stanza.models.lemma_classifier import train_model

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

    train_args = []
    if '--save_name' not in extra_args:
        train_args = ['--save_name', build_model_filename(paths, short_name, command_args, extra_args)]
    if "--train_file" not in extra_args:
        train_file = os.path.join("data", "lemma_classifier", "%s.train.lemma" % short_name)
        train_args += ['--train_file', train_file]
    train_args = train_args + charlm_args + extra_args
    train_model.main(train_args)

def main():
    common.main(run_treebank, "lemma_classifier", "lemma_classifier", add_lemma_args, sub_argparse=train_model.build_argparse(), build_model_filename=build_model_filename, choose_charlm_method=choose_lemma_charlm)


if __name__ == '__main__':
    main()
