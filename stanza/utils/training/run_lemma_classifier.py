

from stanza.models.lemma_classifier import train_model

from stanza.utils.training import common
from stanza.utils.training.common import Mode, add_charlm_args, build_lemma_charlm_args, choose_lemma_charlm

def add_lemma_args(parser):
    add_charlm_args(parser)

def build_model_filename(paths, short_name, command_args, extra_args):
    # TODO: a more interesting name
    return short_name + "_lemmaclassifier.pt"

def run_treebank(mode, paths, treebank, short_name,
                 temp_output_file, command_args, extra_args):
    short_language, dataset = short_name.split("_", 1)
    charlm_args = build_lemma_charlm_args(short_language, dataset, command_args.charlm)

    # TODO: the script to prepare the files could put them in a standard place
    # then this script could automatically point to those files for the dataset in question
    train_args = charlm_args + extra_args
    train_model.main(train_args)

def main():
    common.main(run_treebank, "lemma_classifier", "lemma_classifier", add_lemma_args, sub_argparse=train_model.build_argparse(), build_model_filename=build_model_filename, choose_charlm_method=choose_lemma_charlm)


if __name__ == '__main__':
    main()
