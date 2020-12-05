"""
Prepares train, dev, test for a treebank

For example, do
  python -m stanza.utils.datasets.prepare_tokenizer_treebank TREEBANK
such as
  python -m stanza.utils.datasets.prepare_tokenizer_treebank UD_English-EWT

and it will prepare each of train, dev, test

There are macros for preparing all of the UD treebanks at once:
  python -m stanza.utils.datasets.prepare_tokenizer_treebank ud_all
  python -m stanza.utils.datasets.prepare_tokenizer_treebank all_ud
Both are present because I kept forgetting which was the correct one

There are a few special case handlings of treebanks in this file:
  - UD_English-EWT has the MWTs stripped
  - all Vietnamese treebanks have special post-processing to handle
    some of the difficult spacing issues in Vietnamese text
  - treebanks with train and test but no dev split have the
    train data randomly split into two pieces
  - however, instead of splitting very tiny treebanks, we skip those
"""

import glob
import os
import random
import re
import shutil
import subprocess

import stanza.utils.datasets.common as common
import stanza.utils.datasets.postprocess_vietnamese_tokenizer_data as postprocess_vietnamese_tokenizer_data
import stanza.utils.datasets.prepare_tokenizer_data as prepare_tokenizer_data
import stanza.utils.datasets.preprocess_ssj_data as preprocess_ssj_data

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

def convert_conllu_to_txt(conllu, txt):
    # use an external script to produce the txt files
    subprocess.check_output(f"perl {CONLLU_TO_TXT_PERL} {conllu} > {txt}", shell=True)

def split_train_file(treebank, train_input_conllu,
                     train_output_conllu, train_output_txt,
                     dev_output_conllu, dev_output_txt):
    # set the seed for each data file so that the results are the same
    # regardless of how many treebanks are processed at once
    random.seed(1234)

    # read and shuffle conllu data
    sents = read_sentences_from_conllu(train_input_conllu)
    random.shuffle(sents)
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

    convert_conllu_to_txt(train_output_conllu, train_output_txt)
    convert_conllu_to_txt(dev_output_conllu, dev_output_txt)

    return True

def mwt_name(base_dir, short_name, dataset):
    return f"{base_dir}/{short_name}-ud-{dataset}-mwt.json"

def prepare_labels(input_txt, input_conllu, tokenizer_dir, short_name, short_language, dataset):
    prepare_tokenizer_data.main([input_txt,
                                 input_conllu,
                                 "-o", f"{tokenizer_dir}/{short_name}-ud-{dataset}.toklabels",
                                 "-m", mwt_name(tokenizer_dir, short_name, dataset)])

    if short_language == "vi":
        postprocess_vietnamese_tokenizer_data.main([input_txt,
                                                    "--char_level_pred", f"{tokenizer_dir}/{short_name}-ud-{dataset}.toklabels",
                                                    "-o", f"{tokenizer_dir}/{short_name}-ud-{dataset}.json"])

MWT_RE = re.compile("^[0-9]+[-][0-9]+")

def strip_mwt_from_conll(input_conllu, output_conllu):
    with open(input_conllu) as fin:
        with open(output_conllu, "w") as fout:
            for line in fin:
                if not MWT_RE.match(line):
                    fout.write(line)

def augment_arabic_padt(sents):
    """
    Basic Arabic tokenizer gets the trailing punctuation wrong if there is a blank space.

    Reason seems to be that there are almost no examples of "text ." in the dataset.
    This function augments the Arabic-PADT dataset with a few such examples.
    Note: it may very well be that a lot of tokeners have this problem.

    Also, there are a few examples in UD2.7 which are apparently
    headlines where there is a ' . ' in the middle of the text.
    According to an Arabic speaking labmate, the sentences are
    headlines which could be reasonably split into two items.  Having
    them as one item is quite confusing and possibly incorrect, but
    such is life.
    """
    new_sents = []
    for sentence in sents:
        if len(sentence) < 4:
            raise ValueError("Read a surprisingly short sentence")
        text_line = None
        if sentence[0].startswith("# newdoc") and sentence[3].startswith("# text"):
            text_line = 3
        elif sentence[0].startswith("# newpar") and sentence[2].startswith("# text"):
            text_line = 2
        elif sentence[0].startswith("# sent_id") and sentence[1].startswith("# text"):
            text_line = 1
        else:
            raise ValueError("Could not find text line in %s" % sentence[0].split()[-1])

        # for some reason performance starts dropping quickly at higher numbers
        if (sentence[text_line][-1] in ('.', '؟', '?', '!') and
            sentence[text_line][-2] not in ('.', '؟', '?', '!', ' ') and
            sentence[-2].split()[-1].find("SpaceAfter=No") >= 0 and
            len(sentence[-1].split()[1]) == 1 and
            random.random() < 0.05):
            new_sent = list(sentence)
            new_sent[text_line] = new_sent[text_line][:-1] + ' ' + new_sent[text_line][-1]
            pieces = sentence[-2].split("\t")
            if pieces[-1] == "SpaceAfter=No":
                pieces[-1] = "_"
            elif pieces[-1].startswith("SpaceAfter=No|"):
                pieces[-1] = pieces[-1].replace("SpaceAfter=No|", "")
            elif pieces[-1].find("|SpaceAfter=No") > 0:
                pieces[-1] = piecse[-1].replace("|SpaceAfter=No", "")
            else:
                raise ValueError("WTF")
            new_sent[-2] = "\t".join(pieces)
            assert new_sent != sentence
            new_sents.append(new_sent)
    return new_sents


def augment_telugu(sents):
    """
    Add a few sentences with modified punctuation to Telugu_MTG

    The Telugu-MTG dataset has punctuation separated from the text in
    almost all cases, which makes the tokenizer not learn how to
    process that correctly.

    All of the Telugu sentences end with their sentence final
    punctuation being separated.  Furthermore, all commas are
    separated.  We change that on some subset of the sentences to
    make the tools more generalizable on wild text.
    """
    new_sents = []
    for sentence in sents:
        if not sentence[1].startswith("# text"):
            raise ValueError("Expected the second line of %s to start with # text" % sentence[0])
        if not sentence[2].startswith("# translit"):
            raise ValueError("Expected the second line of %s to start with # translit" % sentence[0])
        if sentence[1].endswith(". . .") or sentence[1][-1] not in ('.', '?', '!'):
            continue
        if sentence[1][-1] in ('.', '?', '!') and sentence[1][-2] != ' ' and sentence[1][-3:] != ' ..' and sentence[1][-4:] != ' ...':
            raise ValueError("Sentence %s does not end with space-punctuation, which is against our assumptions for the te_mtg treebank.  Please check the augment method to see if it is still needed" % sentence[0])
        if random.random() < 0.1:
            new_sentence = list(sentence)
            new_sentence[1] = new_sentence[1][:-2] + new_sentence[1][-1]
            new_sentence[2] = new_sentence[2][:-2] + new_sentence[2][-1]
            new_sentence[-2] = new_sentence[-2] + "|SpaceAfter=No"
            new_sents.append(new_sentence)
        if sentence[1].find(",") > 1 and random.random() < 0.1:
            new_sentence = list(sentence)
            index = sentence[1].find(",")
            new_sentence[1] = sentence[1][:index-1] + sentence[1][index:]
            index = sentence[1].find(",")
            new_sentence[2] = sentence[2][:index-1] + sentence[2][index:]
            for idx, word in enumerate(new_sentence):
                if idx < 4:
                    # skip sent_id, text, transliteration, and the first word
                    continue
                if word.split("\t")[1] == ',':
                    new_sentence[idx-1] = new_sentence[idx-1] + "|SpaceAfter=No"
                    break
            new_sents.append(new_sentence)
    return new_sents

def write_augmented_dataset(input_conllu, output_conllu, output_txt, augment_function):
    # set the seed for each data file so that the results are the same
    # regardless of how many treebanks are processed at once
    random.seed(1234)

    # read and shuffle conllu data
    sents = read_sentences_from_conllu(input_conllu)

    # the actual meat of the function - produce new sentences
    new_sents = augment_function(sents)

    write_sentences_to_conllu(output_conllu, sents + new_sents)
    convert_conllu_to_txt(output_conllu, output_txt)


def prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, dataset, augment=True):
    os.makedirs(tokenizer_dir, exist_ok=True)

    input_txt = common.find_treebank_dataset_file(treebank, udbase_dir, dataset, "txt")
    input_txt_copy = f"{tokenizer_dir}/{short_name}.{dataset}.txt"

    input_conllu = common.find_treebank_dataset_file(treebank, udbase_dir, dataset, "conllu")
    input_conllu_copy = f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu"

    if short_name == "sl_ssj":
        preprocess_ssj_data.process(input_txt, input_conllu, input_txt_copy, input_conllu_copy)
    elif short_name == "te_mtg" and dataset == 'train' and augment:
        write_augmented_dataset(input_conllu, input_conllu_copy, input_txt_copy, augment_telugu)
    elif short_name == "ar_padt" and dataset == 'train' and augment:
        write_augmented_dataset(input_conllu, input_conllu_copy, input_txt_copy, augment_arabic_padt)
    elif short_name == "en_ewt":
        # For a variety of reasons we want to strip the MWT from English
        # One reason in particular is that other English datasets do not
        # have MWT, so if we have the eventual goal of mixing datasets,
        # it will be impossible to do while keeping MWT.
        # Another reason is even if we kept MWT in EWT when mixing datasets,
        # it would be very difficult for users to switch between the two
        strip_mwt_from_conll(input_conllu, input_conllu_copy)
        shutil.copyfile(input_txt, input_txt_copy)
    else:
        shutil.copyfile(input_txt, input_txt_copy)
        shutil.copyfile(input_conllu, input_conllu_copy)

    prepare_labels(input_txt_copy, input_conllu_copy, tokenizer_dir, short_name, short_language, dataset)

def process_ud_treebank(treebank, udbase_dir, tokenizer_dir, short_name, short_language, augment=True):
    """
    Process a normal UD treebank with train/dev/test splits

    SL-SSJ and Vietnamese both use this code path as well.
    """
    prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, "train", augment)
    prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, "dev", augment)
    prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, "test", augment)


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
    train_input_conllu = common.find_treebank_dataset_file(treebank, udbase_dir, "train", "conllu")
    train_output_conllu = f"{tokenizer_dir}/{short_name}.train.gold.conllu"
    train_output_txt = f"{tokenizer_dir}/{short_name}.train.txt"
    dev_output_conllu = f"{tokenizer_dir}/{short_name}.dev.gold.conllu"
    dev_output_txt = f"{tokenizer_dir}/{short_name}.dev.txt"

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
    # currently we do not do any augmentation of these partial treebanks
    prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, "test", augment=False)


def process_treebank(treebank, paths, augment=True):
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

    train_txt_file = common.find_treebank_dataset_file(treebank, udbase_dir, "train", "txt")
    if not train_txt_file:
        raise ValueError("Cannot find train file for treebank %s" % treebank)

    short_name = treebank_to_short_name(treebank)
    short_language = short_name.split("_")[0]

    print("Preparing data for %s: %s, %s" % (treebank, short_name, short_language))

    if not common.find_treebank_dataset_file(treebank, udbase_dir, "dev", "txt"):
        process_partial_ud_treebank(treebank, udbase_dir, tokenizer_dir, short_name, short_language)
    else:
        process_ud_treebank(treebank, udbase_dir, tokenizer_dir, short_name, short_language, augment)


def main():
    common.main(process_treebank)

if __name__ == '__main__':
    main()

