import argparse
import os
import subprocess

from stanza.utils.datasets.sentiment.process_utils import SentimentDatum
import stanza.utils.datasets.sentiment.process_utils as process_utils

import stanza.utils.default_paths as default_paths

TREEBANK_FILES = ["train.txt", "dev.txt", "test.txt", "extra-train.txt", "checked-extra-train.txt"]

ARGUMENTS = {
    "fiveclass":      [],
    "root":           ["-root_only"],
    "binary":         ["-ignore_labels", "2", "-remap_labels", "1=0,2=-1,3=1,4=1"],
    "binaryroot":     ["-root_only", "-ignore_labels", "2", "-remap_labels", "1=0,2=-1,3=1,4=1"],
    "threeclass":     ["-remap_labels", "0=0,1=0,2=1,3=2,4=2"],
    "threeclassroot": ["-root_only", "-remap_labels", "0=0,1=0,2=1,3=2,4=2"],
}


def get_subtrees(input_file, *args):
    """
    Use the CoreNLP OutputSubtrees tool to convert the input file to a bunch of phrases

    Returns a list of the SentimentDatum namedtuple
    """
    # TODO: maybe can convert this to use the python tree?
    cmd = ["java", "edu.stanford.nlp.trees.OutputSubtrees", "-input", input_file]
    if len(args) > 0:
        cmd = cmd + list(args)
    print (" ".join(cmd))
    results = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    lines = results.stdout.split("\n")
    lines = [x.strip() for x in lines]
    lines = [x for x in lines if x]
    lines = [x.split(maxsplit=1) for x in lines]
    phrases = [SentimentDatum(x[0], x[1].split()) for x in lines]
    return phrases

def get_phrases(dataset, treebank_file, input_dir):
    extra_args = ARGUMENTS[dataset]

    input_file = os.path.join(input_dir, "fiveclass", treebank_file)
    if not os.path.exists(input_file):
        raise FileNotFoundError(input_file)
    phrases = get_subtrees(input_file, *extra_args)
    print("Found {} phrases in SST {} {}".format(len(phrases), treebank_file, dataset))
    return phrases

def convert_version(dataset, treebank_file, input_dir, output_dir):
    """
    Convert the fiveclass files to a specific format

    Uses the ARGUMENTS specific for the format wanted
    """
    phrases = get_phrases(dataset, treebank_file, input_dir)
    output_file = os.path.join(output_dir, "en_sst.%s.%s.json" % (dataset, treebank_file.split(".")[0]))
    process_utils.write_list(output_file, phrases)

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
        for treebank_file in TREEBANK_FILES:
            convert_version(section, treebank_file, input_dir, output_dir)

if __name__ == '__main__':
    main()
