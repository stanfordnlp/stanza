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
import random
import shutil
import subprocess
import sys

import stanza.utils.default_paths as default_paths
import stanza.utils.prepare_tokenizer_data as prepare_tokenizer_data
import stanza.utils.postprocess_vietnamese_tokenizer_data as postprocess_vietnamese_tokenizer_data
import stanza.utils.preprocess_ssj_data as preprocess_ssj_data

from stanza.models.common.constant import treebank_to_short_name

CONLLU_TO_TXT_PERL = os.path.join(os.path.split(__file__)[0], "conllu_to_text.pl")

def read_sentences_from_conllu(filename):
    sents = []
    cache = []
    with open(filename) as infile:
        for line in infile:
            line = line.strip()
            if len(line) == 0:
                if len(cache) > 0:
                    sents += [cache]
                    cache = []
                continue
            cache += [line]
        if len(cache) > 0:
            sents += [cache]
    return sents

def write_sentences_to_conllu(filename, sents):
    with open(filename, 'w') as outfile:
        for lines in sents:
            for line in lines:
                print(line, file=outfile)
            print("", file=outfile)

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

def split_train_file(treebank, train_input_conllu,
                     train_output_conllu, train_output_txt,
                     dev_output_conllu, dev_output_txt):
    random.seed(1234)

    # read and shuffle conllu data
    sents = read_sentences_from_conllu(train_input_conllu)
    random.shuffle(sents)
    if len(sents) < 100:
        print("Only %d sentences in %s.  Skipping" % (len(sents), treebank))
        return False
    n_dev = int(len(sents) * XV_RATIO)
    assert n_dev >= 1, "Dev sentence number less than one."
    n_train = len(sents) - n_dev

    # split conllu data
    dev_sents = sents[:n_dev]
    train_sents = sents[n_dev:]
    print("Train/dev split not present.  Randomly splitting train file")
    print(f"{len(sents)} total sentences found: {n_train} in train, {n_dev} in dev.")

    # write conllu
    write_sentences_to_conllu(train_output_conllu, train_sents)
    write_sentences_to_conllu(dev_output_conllu, dev_sents)

    # use an external script to produce the txt files
    subprocess.run(f"perl {CONLLU_TO_TXT_PERL} {train_output_conllu} > {train_output_txt}", shell=True)
    subprocess.run(f"perl {CONLLU_TO_TXT_PERL} {dev_output_conllu} > {dev_output_txt}", shell=True)

    return True

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
                         # this will be fixed using XV
                         #find_treebank_dataset_file(t, udbase_dir, "dev", "txt") and
                         find_treebank_dataset_file(t, udbase_dir, "test", "txt"))]
        treebanks = [t for t in treebanks
                     if not all_underscores(find_treebank_dataset_file(t, udbase_dir, "train", "txt"))]
    return treebanks

def prepare_labels(input_txt_copy, input_conllu_copy, tokenizer_dir, short_name, short_language, dataset):
    prepare_tokenizer_data.main([input_txt_copy,
                                 input_conllu_copy,
                                 "-o", f"{tokenizer_dir}/{short_name}-ud-{dataset}.toklabels",
                                 "-m", f"{tokenizer_dir}/{short_name}-ud-{dataset}-mwt.json"])

    if short_language == "vi":
        postprocess_vietnamese_tokenizer_data.main([input_txt_copy,
                                                    "--char_level_pred", f"{tokenizer_dir}/{short_name}-ud-{dataset}.toklabels",
                                                    "-o", f"{tokenizer_dir}/{short_name}-ud-{dataset}.json"])

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

    prepare_labels(input_txt_copy, input_conllu_copy, tokenizer_dir, short_name, short_language, dataset)

def process_ud_treebank(treebank, udbase_dir, tokenizer_dir, short_name, short_language):
    """
    Process a normal UD treebank with train/dev/test splits

    SL-SSJ and Vietnamese both use this code path as well.
    """
    prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, "train")
    prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, "dev")
    prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, "test")


XV_RATIO = 0.2

def process_partial_ud_treebank(treebank, udbase_dir, tokenizer_dir, short_name, short_language):
    """
    Process a UD treebank with only train/test splits

    For example, in UD 2.7:
      UD_Buryat-BDT
      UD_Galician-TreeGal
      UD_Indonesian-CSUI
      UD_Kazakh-KTB
      UD_Kurmanji-MG
      UD_Latin-Perseus
      UD_Livvi-KKPP
      UD_North_Sami-Giella
      UD_Old_Russian-RNC
      UD_Sanskrit-Vedic
      UD_Slovenian-SST
      UD_Upper_Sorbian-UFAL
      UD_Welsh-CCG
    """
    train_input_conllu = find_treebank_dataset_file(treebank, udbase_dir, "train", "conllu")
    train_output_conllu = f"{tokenizer_dir}/{short_name}.train.gold.conllu"
    train_output_txt = f"{tokenizer_dir}/{short_name}.train.gold.txt"
    dev_output_conllu = f"{tokenizer_dir}/{short_name}.dev.gold.conllu"
    dev_output_txt = f"{tokenizer_dir}/{short_name}.dev.gold.txt"

    if not split_train_file(treebank=treebank,
                            train_input_conllu=train_input_conllu,
                            train_output_conllu=train_output_conllu,
                            train_output_txt=train_output_txt,
                            dev_output_conllu=dev_output_conllu,
                            dev_output_txt=dev_output_txt):
        return

    prepare_labels(train_output_txt, train_output_conllu, tokenizer_dir, short_name, short_language, "train")
    prepare_labels(dev_output_txt, dev_output_conllu, tokenizer_dir, short_name, short_language, "dev")

    # the test set is already fine
    prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, "test")


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

    train_txt_file = find_treebank_dataset_file(treebank, udbase_dir, "train", "txt")
    if not train_txt_file:
        raise ValueError("Cannot find train file for treebank %s" % treebank)

    short_name = treebank_to_short_name(treebank)
    short_language = short_name.split("_")[0]

    print("Preparing data for %s: %s, %s" % (treebank, short_name, short_language))

    if not find_treebank_dataset_file(treebank, udbase_dir, "dev", "txt"):
        process_partial_ud_treebank(treebank, udbase_dir, tokenizer_dir, short_name, short_language)
    else:
        process_ud_treebank(treebank, udbase_dir, tokenizer_dir, short_name, short_language)


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

