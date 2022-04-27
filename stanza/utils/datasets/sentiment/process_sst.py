import argparse
import os
import subprocess

import stanza.utils.default_paths as default_paths

TREEBANK_FILES = ["train.txt", "dev.txt", "test.txt", "extra-train.txt", "checked-extra-train.txt"]

ARGUMENTS = {
    "fiveclass":  [],
    "root":       ["-root_only"],
    "binary":     ["-ignore_labels", "2", "-remap_labels", "1=0,2=-1,3=1,4=1"],
    "threeclass": ["-remap_labels", "0=0,1=0,2=1,3=2,4=2"],
}


def output_subtrees(input_file, output_file, *args):
    """
    Use the CoreNLP OutputSubtrees tool to convert the input file to output
    """
    # TODO: maybe can convert this to use the python tree?
    cmd = ["java", "edu.stanford.nlp.trees.OutputSubtrees", "-input", input_file, "-output", output_file]
    if len(args) > 0:
        cmd = cmd + list(args)
    print (" ".join(cmd))
    subprocess.run(cmd)

def convert_version(dataset, input_dir, output_dir):
    """
    Convert the fiveclass files to a specific format

    Uses the ARGUMENTS specific for the format wanted
    """
    extra_args = ARGUMENTS[dataset]
    for treebank_file in TREEBANK_FILES:
        input_file = os.path.join(input_dir, "fiveclass", treebank_file)
        if not os.path.exists(input_file):
            raise FileNotFoundError(input_file)
        output_file = os.path.join(output_dir, "en_sst.%s.%s" % (dataset, treebank_file))
        output_subtrees(input_file, output_file, *extra_args)

def parse_args():
    """
    Actually, the only argument used right now is the formats to convert
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('sections', type=str, nargs='*', help='Which transformations to use: {}'.format(" ".join(ARGUMENTS.keys())))
    args = parser.parse_args()
    if not args.sections:
        args.sections = list(ARGUMENTS.keys())
    return args

def main():
    args = parse_args()
    paths = default_paths.get_default_paths()
    input_dir = os.path.join(paths["SENTIMENT_BASE"], "sentiment-treebank")
    output_dir = paths["SENTIMENT_DATA_DIR"]

    os.makedirs(output_dir, exist_ok=True)
    for section in args.sections:
        convert_version(section, input_dir, output_dir)

if __name__ == '__main__':
    main()
