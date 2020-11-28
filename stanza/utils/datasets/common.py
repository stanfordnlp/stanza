
import glob
import os
import sys

import stanza.utils.default_paths as default_paths

def find_treebank_dataset_file(treebank, udbase_dir, dataset, extension):
    """
    For a given treebank, dataset, extension, look for the exact filename to use.

    Sometimes the short name we use is different from the short name
    used by UD.  For example, Norwegian or Chinese.  Hence the reason
    to not hardcode it based on treebank
    """
    files = glob.glob(f"{udbase_dir}/{treebank}/*-ud-{dataset}.{extension}")
    if len(files) == 0:
        return None
    elif len(files) == 1:
        return files[0]
    else:
        raise RuntimeError(f"Unexpected number of files matched '{udbase_dir}/{treebank}/*-ud-{dataset}.{extension}'")

def all_underscores(filename):
    """
    Certain treebanks have proprietary data, so the text is hidden

    For example:
      UD_Arabic-NYUAD
      UD_English-ESL
      UD_English-GUMReddit
      UD_Hindi_English-HIENCS
      UD_Japanese-BCCWJ
    """
    for line in open(filename).readlines():
        line = line.strip()
        if not line:
            continue
        line = line.replace("_", "")
        line = line.replace("-", "")
        line = line.replace(" ", "")
        if line:
            return False
    return True

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
    treebanks = [os.path.split(t)[1] for t in treebanks]
    if filtered:
        treebanks = [t for t in treebanks
                     if (find_treebank_dataset_file(t, udbase_dir, "train", "txt") and
                         # this will be fixed using XV
                         #find_treebank_dataset_file(t, udbase_dir, "dev", "txt") and
                         find_treebank_dataset_file(t, udbase_dir, "test", "txt"))]
        treebanks = [t for t in treebanks
                     if not all_underscores(find_treebank_dataset_file(t, udbase_dir, "train", "txt"))]
        # eliminate partial treebanks (fixed with XV) for which we only have 1000 words or less
        treebanks = [t for t in treebanks
                     if (find_treebank_dataset_file(t, udbase_dir, "dev", "conllu") or
                         num_words_in_file(find_treebank_dataset_file(t, udbase_dir, "train", "conllu")) > 1000)]
    return treebanks

def main(process_treebank):
    if len(sys.argv) == 1:
        raise ValueError("Need to provide a treebank name")

    treebank = sys.argv[1]
    paths = default_paths.get_default_paths()
    if treebank.lower() in ('ud_all', 'all_ud'):
        treebanks = get_ud_treebanks(paths["UDBASE"])
        for t in treebanks:
            process_treebank(t, paths)
    else:
        process_treebank(treebank, paths)

