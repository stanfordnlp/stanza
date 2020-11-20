"""
Prepares train, dev, test for a treebank

For example, do
  python -m stanza.utils.prepare_tokenizer_treebank TREEBANK
such as
  python -m stanza.utils.prepare_tokenizer_treebank UD_English-EWT

and it will prepare each of train, dev, test
"""

import glob
import os
import shutil
import sys

import stanza.utils.default_paths as default_paths
import stanza.utils.prepare_tokenizer_data as prepare_tokenizer_data
import stanza.utils.postprocess_vietnamese_tokenizer_data as postprocess_vietnamese_tokenizer_data
import stanza.utils.preprocess_ssj_data as preprocess_ssj_data

from stanza.models.common.constant import treebank_to_short_name

def find_treebank_dataset_file(treebank, udbase_dir, dataset, extension):
    # sometimes the short name we use is different from the short name
    # used by UD.  For example, Norwegian or Chinese
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

def get_ud_treebanks(udbase_dir, filtered=True):
    """
    Looks in udbase_dir for all the treebanks which have both train, dev, and test
    """
    treebanks = sorted(glob.glob(udbase_dir + "/UD_*"))
    treebanks = [os.path.split(t)[1] for t in treebanks]
    if filtered:
        treebanks = [t for t in treebanks
                     if (find_treebank_dataset_file(t, udbase_dir, "train", "txt") and
                         find_treebank_dataset_file(t, udbase_dir, "dev", "txt") and
                         find_treebank_dataset_file(t, udbase_dir, "test", "txt"))]
        treebanks = [t for t in treebanks
                     if not all_underscores(find_treebank_dataset_file(t, udbase_dir, "train", "txt"))]
    return treebanks


def prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, dataset):
    input_txt = find_treebank_dataset_file(treebank, udbase_dir, dataset, "txt")
    input_txt_copy = f"{tokenizer_dir}/{short_name}.{dataset}.txt"

    input_conllu = find_treebank_dataset_file(treebank, udbase_dir, dataset, "conllu")
    input_conllu_copy = f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu"

    if short_name == "sl_ssj":
        preprocess_ssj_data.process(input_txt, input_conllu, input_txt_copy, input_conllu_copy)
    else:
        os.makedirs(tokenizer_dir, exist_ok=True)
        shutil.copyfile(input_txt, input_txt_copy)
        shutil.copyfile(input_conllu, input_conllu_copy)

    prepare_tokenizer_data.main([input_txt_copy,
                                 input_conllu_copy,
                                 "-o", f"{tokenizer_dir}/{short_name}-ud-{dataset}.toklabels",
                                 "-m", f"{tokenizer_dir}/{short_name}-ud-{dataset}-mwt.json"])

    if short_language == "vi":
        postprocess_vietnamese_tokenizer_data.main([input_txt_copy,
                                                    "--char_level_pred", f"{tokenizer_dir}/{short_name}-ud-{dataset}.toklabels",
                                                    "-o", f"{tokenizer_dir}/{short_name}-ud-{dataset}.json"])

def process_treebank(treebank, paths):
    """
    Processes a single treebank into train, dev, test parts

    TODO
    Currently assumes it is always a UD treebank.  There are Thai
    treebanks which are not included in UD.

    Also, there is no specific mechanism for UD_Arabic-NYUAD or
    similar treebanks, which need integration with LDC datsets
    """
    udbase_dir = paths["UDBASE"]
    tokenizer_dir = paths["TOKENIZE_DATA_DIR"]

    short_name = treebank_to_short_name(treebank)
    short_language = short_name.split("_")[0]

    print("Preparing data for %s: %s, %s" % (treebank, short_name, short_language))

    prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, "train")
    prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, "dev")
    prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, "test")

def main():
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

if __name__ == '__main__':
    main()

