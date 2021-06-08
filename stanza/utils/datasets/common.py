
import argparse
import glob
import logging
import os
import re
import subprocess
import sys

import stanza.utils.default_paths as default_paths
from stanza.models.common.constant import treebank_to_short_name

logger = logging.getLogger('stanza')

SHORTNAME_RE = re.compile("[a-z-]+_[a-z0-9]+")

def project_to_short_name(treebank):
    """
    Project either a treebank or a short name to a short name

    TODO: see if treebank_to_short_name can incorporate this
    """
    if SHORTNAME_RE.match(treebank):
        return treebank
    else:
        return treebank_to_short_name(treebank)

def find_treebank_dataset_file(treebank, udbase_dir, dataset, extension, fail=False):
    """
    For a given treebank, dataset, extension, look for the exact filename to use.

    Sometimes the short name we use is different from the short name
    used by UD.  For example, Norwegian or Chinese.  Hence the reason
    to not hardcode it based on treebank

    set fail=True to fail if the file is not found
    """
    if treebank.startswith("UD_Korean") and treebank.endswith("_seg"):
        treebank = treebank[:-4]
    filename = os.path.join(udbase_dir, treebank, f"*-ud-{dataset}.{extension}")
    files = glob.glob(filename)
    if len(files) == 0:
        if fail:
            raise FileNotFoundError("Could not find any treebank files which matched {}".format(filename))
        else:
            return None
    elif len(files) == 1:
        return files[0]
    else:
        raise RuntimeError(f"Unexpected number of files matched '{udbase_dir}/{treebank}/*-ud-{dataset}.{extension}'")

def mostly_underscores(filename):
    """
    Certain treebanks have proprietary data, so the text is hidden

    For example:
      UD_Arabic-NYUAD
      UD_English-ESL
      UD_English-GUMReddit
      UD_Hindi_English-HIENCS
      UD_Japanese-BCCWJ
    """
    underscore_count = 0
    total_count = 0
    for line in open(filename).readlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        total_count = total_count + 1
        pieces = line.split("\t")
        if pieces[1] in ("_", "-"):
            underscore_count = underscore_count + 1
    return underscore_count / total_count > 0.5

def num_words_in_file(conllu_file):
    """
    Count the number of non-blank lines in a conllu file
    """
    count = 0
    with open(conllu_file) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            count = count + 1
    return count


def get_ud_treebanks(udbase_dir, filtered=True):
    """
    Looks in udbase_dir for all the treebanks which have both train, dev, and test
    """
    treebanks = sorted(glob.glob(udbase_dir + "/UD_*"))
    # skip UD_English-GUMReddit as it is usually incorporated into UD_English-GUM
    treebanks = [os.path.split(t)[1] for t in treebanks]
    treebanks = [t for t in treebanks if t != "UD_English-GUMReddit"]
    if filtered:
        treebanks = [t for t in treebanks
                     if (find_treebank_dataset_file(t, udbase_dir, "train", "conllu") and
                         # this will be fixed using XV
                         #find_treebank_dataset_file(t, udbase_dir, "dev", "conllu") and
                         find_treebank_dataset_file(t, udbase_dir, "test", "conllu"))]
        treebanks = [t for t in treebanks
                     if not mostly_underscores(find_treebank_dataset_file(t, udbase_dir, "train", "conllu"))]
        # eliminate partial treebanks (fixed with XV) for which we only have 1000 words or less
        treebanks = [t for t in treebanks
                     if (find_treebank_dataset_file(t, udbase_dir, "dev", "conllu") or
                         num_words_in_file(find_treebank_dataset_file(t, udbase_dir, "train", "conllu")) > 1000)]
    return treebanks

def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('treebanks', type=str, nargs='+', help='Which treebanks to run on.  Use all_ud or ud_all for all UD treebanks')
    return parser


def main(process_treebank, add_specific_args=None):
    logger.info("Datasets program called with:\n" + " ".join(sys.argv))

    parser = build_argparse()
    if add_specific_args is not None:
        add_specific_args(parser)
    args = parser.parse_args()

    paths = default_paths.get_default_paths()

    treebanks = []
    for treebank in args.treebanks:
        if treebank.lower() in ('ud_all', 'all_ud'):
            ud_treebanks = get_ud_treebanks(paths["UDBASE"])
            treebanks.extend(ud_treebanks)
        else:
            treebanks.append(treebank)

    for treebank in treebanks:
        process_treebank(treebank, paths, args)
