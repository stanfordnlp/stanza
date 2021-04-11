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
  - all Vietnamese treebanks have special post-processing to handle
    some of the difficult spacing issues in Vietnamese text
  - treebanks with train and test but no dev split have the
    train data randomly split into two pieces
  - however, instead of splitting very tiny treebanks, we skip those
"""

import argparse
import glob
import os
import random
import re
import shutil
import tempfile

import stanza.utils.datasets.common as common
import stanza.utils.datasets.prepare_tokenizer_data as prepare_tokenizer_data
import stanza.utils.datasets.preprocess_ssj_data as preprocess_ssj_data


def copy_conllu_file(tokenizer_dir, tokenizer_file, dest_dir, dest_file, short_name):
    original = f"{tokenizer_dir}/{short_name}.{tokenizer_file}.conllu"
    copied = f"{dest_dir}/{short_name}.{dest_file}.conllu"

    shutil.copyfile(original, copied)

def copy_conllu_treebank(treebank, paths, dest_dir, postprocess=None):
    """
    This utility method copies only the conllu files to the given destination directory.

    Both POS and lemma annotators need this.
    """
    os.makedirs(dest_dir, exist_ok=True)

    short_name = common.project_to_short_name(treebank)
    short_language = short_name.split("_")[0]

    with tempfile.TemporaryDirectory() as tokenizer_dir:
        paths = dict(paths)
        paths["TOKENIZE_DATA_DIR"] = tokenizer_dir

        # first we process the tokenization data
        args = argparse.Namespace()
        args.augment = False
        args.prepare_labels = False
        process_treebank(treebank, paths, args)

        os.makedirs(dest_dir, exist_ok=True)

        if postprocess is None:
            postprocess = copy_conllu_file

        # now we copy the processed conllu data files
        postprocess(tokenizer_dir, "train.gold", dest_dir, "train.in", short_name)
        postprocess(tokenizer_dir, "dev.gold", dest_dir, "dev.gold", short_name)
        copy_conllu_file(dest_dir, "dev.gold", dest_dir, "dev.in", short_name)
        postprocess(tokenizer_dir, "test.gold", dest_dir, "test.gold", short_name)
        copy_conllu_file(dest_dir, "test.gold", dest_dir, "test.in", short_name)

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

    common.convert_conllu_to_txt(train_output_conllu, train_output_txt)
    common.convert_conllu_to_txt(dev_output_conllu, dev_output_txt)

    return True

def mwt_name(base_dir, short_name, dataset):
    return f"{base_dir}/{short_name}-ud-{dataset}-mwt.json"

def prepare_dataset_labels(input_txt, input_conllu, tokenizer_dir, short_name, short_language, dataset):
    prepare_tokenizer_data.main([input_txt,
                                 input_conllu,
                                 "-o", f"{tokenizer_dir}/{short_name}-ud-{dataset}.toklabels",
                                 "-m", mwt_name(tokenizer_dir, short_name, dataset)])

MWT_RE = re.compile("^[0-9]+[-][0-9]+")

def strip_mwt_from_sentences(sents):
    """
    Removes all mwt lines from the given list of sentences

    Useful for mixing MWT and non-MWT treebanks together (especially English)
    """
    new_sents = []
    for sentence in sents:
        new_sentence = [line for line in sentence if not MWT_RE.match(line)]
        new_sents.append(new_sentence)
    return new_sents


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
    return sents + new_sents


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
    return sents + new_sents

COMMA_SEPARATED_RE = re.compile(" ([a-zA-Z]+)[,] ([a-zA-Z]+) ")
def augment_ancora(sents):
    """
    Find some fraction of the sentences which match "asdf, zzzz" and squish them to "asdf,zzzz"

    This leaves the tokens and all of the other data the same.  The
    only change made is to change SpaceAfter=No for the "," token and
    adjust the #text line, with the assumption that the conllu->txt
    conversion will correctly handle this change.
    """
    new_sents = []
    for sentences in sents:
        if not sentences[1].startswith("# text"):
            raise ValueError("UD_Spanish-AnCora not in the expected format")

    for sentence in sents:
        match = COMMA_SEPARATED_RE.search(sentence[1])
        if match and random.random() < 0.03:
            for idx, word in enumerate(sentence):
                if word.startswith("#"):
                    continue
                # find() doesn't work because we wind up finding substrings
                if word.split("\t")[1] != match.group(1):
                    continue
                if sentence[idx+1].split("\t")[1] != ',':
                    continue
                if sentence[idx+2].split("\t")[2] != match.group(2):
                    continue
                break
            if idx == len(sentence) - 1:
                # this can happen with MWTs.  we may actually just
                # want to skip MWTs anyway, so no big deal
                continue
            # now idx+1 should be the line with the comma in it
            comma = sentence[idx+1]
            pieces = comma.split("\t")
            assert pieces[1] == ','
            if pieces[-1] == '_':
                pieces[-1] = "SpaceAfter=No"
            else:
                pieces[-1] = pieces[-1] + "|SpaceAfter=No"
            comma = "\t".join(pieces)
            new_sent = sentence[:idx+1] + [comma] + sentence[idx+2:]

            text_offset = sentence[1].find(match.group(1) + ", " + match.group(2))
            text_len = len(match.group(1) + ", " + match.group(2))
            new_text = sentence[1][:text_offset] + match.group(1) + "," + match.group(2) + sentence[1][text_offset+text_len:]
            new_sent[1] = new_text

            new_sents.append(new_sent)

    return sents + new_sents

def fix_spanish_ancora(input_conllu, output_conllu, output_txt, augment):
    """
    The basic Spanish tokenizer has an issue where "asdf,zzzz" does not get tokenized.

    One possible problem is with this sentence:
    # orig_file_sentence 143#5
    In this sentence, there is a comma smashed next to a token.  Seems incorrect.

    Fixing just this one sentence is not sufficient to tokenize
    "asdf,zzzz" as desired, so we also augment by some fraction where
    we have squished "asdf, zzzz" into "asdf,zzzz".
    """
    random.seed(1234)
    sents = read_sentences_from_conllu(input_conllu)

    ORIGINAL_BAD = "29	,Comerç	,Comerç	PROPN	PROPN	_	28	flat	_	_"
    NEW_FIXED = ["29	,	,	PUNCT	PUNCT	PunctType=Comm	32	punct	_	SpaceAfter=No",   # TODO dunno about the head
                 "30	Comerç	Comerç	PROPN	PROPN	_	26	flat	_	_"]
    new_sentences = []
    found = False
    for sentence in sents:
        if sentence[0].strip() != '# sent_id = train-s14205':
            new_sentences.append(sentence)
            continue
        assert not found, "WTF"
        found = True

        for idx, word in enumerate(sentence):
            if word.strip() == ORIGINAL_BAD:
                break
        assert idx == 31, "Could not find ,Comerç at the expected line number.  Perhaps the treebank has been fixed?"
        for word in sentence[3:idx]:
            assert int(sentence[idx].strip().split("\t")[6]) < idx
        new_sentence = sentence[:idx] + NEW_FIXED
        # increase the token idx and the dep of each word as appropriate
        for word in sentence[idx+1:]:
            pieces = word.strip().split("\t")
            pieces[0] = str(int(pieces[0]) + 1)
            dep = int(pieces[6])
            if dep > 29:
                pieces[6] = str(dep + 1)
            new_sentence.append("\t".join(pieces))

        new_sentences.append(new_sentence)

    assert found, "Could not find sentence train-s14205 in Spanish Ancora"

    if augment:
        new_sentences = augment_ancora(new_sentences)

    write_sentences_to_conllu(output_conllu, new_sentences)
    common.convert_conllu_to_txt(output_conllu, output_txt)

def augment_apos(sents):
    """
    If there are no instances of ’ in the dataset, but there are instances of ',
    we replace some fraction of ' with ’ so that the tokenizer will recognize it.
    """
    has_unicode_apos = False
    has_ascii_apos = False
    for sent in sents:
        for line in sent:
            if line.startswith("# text"):
                if line.find("'") >= 0:
                    has_ascii_apos = True
                if line.find("’") >= 0:
                    has_unicode_apos = True
                break
        else:
            raise ValueError("Cannot find '# text'")

    if has_unicode_apos or not has_ascii_apos:
        return sents

    new_sents = []
    for sent in sents:
        if random.random() > 0.05:
            new_sents.append(sent)
            continue
        new_sent = []
        for line in sent:
            if line.startswith("# text"):
                new_sent.append(line.replace("'", "’"))
            elif line.startswith("#"):
                new_sent.append(line)
            else:
                pieces = line.split("\t")
                pieces[1] = pieces[1].replace("'", "’")
                new_sent.append("\t".join(pieces))
        new_sents.append(new_sent)

    return new_sents

def augment_ellipses(sents):
    """
    Replaces a fraction of '...' with '…'
    """
    has_ellipses = False
    has_unicode_ellipses = False
    for sent in sents:
        for line in sent:
            if line.startswith("#"):
                continue
            pieces = line.split("\t")
            if pieces[1] == '...':
                has_ellipses = True
            elif pieces[1] == '…':
                has_unicode_ellipses = True

    if has_unicode_ellipses or not has_ellipses:
        return sents

    new_sents = []

    for sent in sents:
        if random.random() > 0.05:
            new_sents.append(sent)
            continue
        new_sent = []
        for line in sent:
            if line.startswith("#"):
                new_sent.append(line)
            else:
                pieces = line.split("\t")
                if pieces[1] == '...':
                    pieces[1] = '…'
                new_sent.append("\t".join(pieces))
        new_sents.append(new_sent)

    return new_sents

def augment_punct(sents):
    """
    If there are no instances of ’ in the dataset, but there are instances of ',
    we replace some fraction of ' with ’ so that the tokenizer will recognize it.

    Also augments with ... / …

    TODO: handle the wide variety of quotes which are possible
    """
    new_sents = augment_apos(sents)
    new_sents = augment_ellipses(new_sents)

    return new_sents



def write_augmented_dataset(input_conllu, output_conllu, output_txt, augment_function):
    # set the seed for each data file so that the results are the same
    # regardless of how many treebanks are processed at once
    random.seed(1234)

    # read and shuffle conllu data
    sents = read_sentences_from_conllu(input_conllu)

    # the actual meat of the function - produce new sentences
    new_sents = augment_function(sents)

    write_sentences_to_conllu(output_conllu, new_sents)
    common.convert_conllu_to_txt(output_conllu, output_txt)

def remove_spaces_from_sentences(sents):
    """
    Makes sure every word in the list of sentences has SpaceAfter=No.

    Returns a new list of sentences
    """
    new_sents = []
    for sentence in sents:
        new_sentence = []
        for word in sentence:
            if word.startswith("#"):
                new_sentence.append(word)
                continue
            pieces = word.split("\t")
            if pieces[-1] == "_":
                pieces[-1] = "SpaceAfter=No"
            elif pieces[-1].find("SpaceAfter=No") >= 0:
                pass
            else:
                raise ValueError("oops")
            word = "\t".join(pieces)
            new_sentence.append(word)
        new_sents.append(new_sentence)
    return new_sents

def remove_spaces(input_conllu, output_conllu, output_txt):
    """
    Turns a dataset into something appropriate for building a segmenter.

    For example, this works well on the Korean datasets.
    """
    sents = read_sentences_from_conllu(input_conllu)

    new_sents = remove_spaces_from_sentences(sents)

    write_sentences_to_conllu(output_conllu, new_sents)
    common.convert_conllu_to_txt(output_conllu, output_txt)


def build_combined_korean_dataset(udbase_dir, tokenizer_dir, short_name, dataset, output_txt, output_conllu, prepare_labels=True):
    """
    Builds a combined dataset out of multiple Korean datasets.

    Currently this uses GSD and Kaist.  If a segmenter-appropriate
    dataset was requested, spaces are removed.

    TODO: we need to handle the difference in xpos tags somehow.
    """
    gsd_conllu = common.find_treebank_dataset_file("UD_Korean-GSD", udbase_dir, dataset, "conllu")
    kaist_conllu = common.find_treebank_dataset_file("UD_Korean-Kaist", udbase_dir, dataset, "conllu")
    sents = read_sentences_from_conllu(gsd_conllu) + read_sentences_from_conllu(kaist_conllu)

    segmenter = short_name.endswith("_seg")
    if segmenter:
        sents = remove_spaces_from_sentences(sents)

    write_sentences_to_conllu(output_conllu, sents)
    common.convert_conllu_to_txt(output_conllu, output_txt)

    if prepare_labels:
        prepare_dataset_labels(output_txt, output_conllu, tokenizer_dir, short_name, "ko", dataset)

def build_combined_korean(udbase_dir, tokenizer_dir, short_name, prepare_labels=True):
    for dataset in ("train", "dev", "test"):
        output_txt = f"{tokenizer_dir}/{short_name}.{dataset}.txt"
        output_conllu = f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu"
        build_combined_korean_dataset(udbase_dir, tokenizer_dir, short_name, dataset, output_txt, output_conllu, prepare_labels)

def build_combined_italian_dataset(udbase_dir, tokenizer_dir, handparsed_dir, short_name, dataset, prepare_labels):
    output_txt = f"{tokenizer_dir}/{short_name}.{dataset}.txt"
    output_conllu = f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu"

    if dataset == 'train':
        # could maybe add ParTUT, but that dataset has a slightly different xpos set
        # (no DE or I)
        # and I didn't feel like sorting through the differences
        # Note: currently these each have small changes compared with
        # the UD2.7 release.  See the issues (possibly closed by now)
        # filed by AngledLuffa on each of the treebanks for more info.
        treebanks = ["UD_Italian-ISDT", "UD_Italian-VIT", "UD_Italian-TWITTIRO", "UD_Italian-PoSTWITA"]
        sents = []
        for treebank in treebanks:
            conllu_file = common.find_treebank_dataset_file(treebank, udbase_dir, dataset, "conllu", fail=True)
            sents.extend(read_sentences_from_conllu(conllu_file))
        extra_italian = os.path.join(handparsed_dir, "italian-mwt", "italian.mwt")
        if not os.path.exists(extra_italian):
            raise FileNotFoundError("Cannot find the extra dataset 'italian.mwt' which includes various multi-words retokenized, expected {}".format(extra_italian))
        extra_sents = read_sentences_from_conllu(extra_italian)
        for sentence in extra_sents:
            if not sentence[2].endswith("_") or not MWT_RE.match(sentence[2]):
                raise AssertionError("Unexpected format of the italian.mwt file.  Has it already be modified to have SpaceAfter=No everywhere?")
            sentence[2] = sentence[2][:-1] + "SpaceAfter=No"
        sents = sents + extra_sents

        sents = augment_punct(sents)
    else:
        istd_conllu = common.find_treebank_dataset_file("UD_Italian-ISDT", udbase_dir, dataset, "conllu")
        sents = read_sentences_from_conllu(istd_conllu)

    write_sentences_to_conllu(output_conllu, sents)
    common.convert_conllu_to_txt(output_conllu, output_txt)

    if prepare_labels:
        prepare_dataset_labels(output_txt, output_conllu, tokenizer_dir, short_name, "it", dataset)


def build_combined_italian(udbase_dir, tokenizer_dir, handparsed_dir, short_name, prepare_labels=True):
    for dataset in ("train", "dev", "test"):
        build_combined_italian_dataset(udbase_dir, tokenizer_dir, handparsed_dir, short_name, dataset, prepare_labels)

def build_combined_english_dataset(udbase_dir, tokenizer_dir, handparsed_dir, short_name, dataset, prepare_labels):
    """
    en_combined is currently EWT, GUM, and a fork of Pronouns
    """
    output_txt = f"{tokenizer_dir}/{short_name}.{dataset}.txt"
    output_conllu = f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu"

    if dataset == 'train':
        # TODO: include more UD treebanks, possibly with xpos removed
        #  UD_English-ParTUT - xpos are different
        # also include "external" treebanks such as PTB
        treebanks = ["UD_English-EWT", "UD_English-GUM"]
        sents = []
        for treebank in treebanks:
            conllu_file = common.find_treebank_dataset_file(treebank, udbase_dir, dataset, "conllu", fail=True)
            sents.extend(read_sentences_from_conllu(conllu_file))
        # this fork of Pronouns addresses a few issues with the dataset
        # features, tags, and lemmas were all improved
        pronouns_fork = os.path.join(handparsed_dir, "english-pronouns", "en_pronouns-ud-test-GL.conllu")
        pronouns_sents = read_sentences_from_conllu(pronouns_fork)
        sents.extend(pronouns_sents)

        sents = augment_punct(sents)
    else:
        ewt_conllu = common.find_treebank_dataset_file("UD_English-EWT", udbase_dir, dataset, "conllu")
        sents = read_sentences_from_conllu(ewt_conllu)

    sents = strip_mwt_from_sentences(sents)
    write_sentences_to_conllu(output_conllu, sents)
    common.convert_conllu_to_txt(output_conllu, output_txt)

    if prepare_labels:
        prepare_dataset_labels(output_txt, output_conllu, tokenizer_dir, short_name, "it", dataset)


def build_combined_english(udbase_dir, tokenizer_dir, handparsed_dir, short_name, prepare_labels=True):
    for dataset in ("train", "dev", "test"):
        build_combined_english_dataset(udbase_dir, tokenizer_dir, handparsed_dir, short_name, dataset, prepare_labels)


def prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, dataset, augment=True, prepare_labels=True):
    # TODO: do this higher up
    os.makedirs(tokenizer_dir, exist_ok=True)

    input_txt = common.find_treebank_dataset_file(treebank, udbase_dir, dataset, "txt")
    input_txt_copy = f"{tokenizer_dir}/{short_name}.{dataset}.txt"

    input_conllu = common.find_treebank_dataset_file(treebank, udbase_dir, dataset, "conllu")
    input_conllu_copy = f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu"

    if short_name == "sl_ssj":
        preprocess_ssj_data.process(input_conllu, input_txt_copy, input_conllu_copy)
    elif short_name == "te_mtg" and dataset == 'train' and augment:
        write_augmented_dataset(input_conllu, input_conllu_copy, input_txt_copy, augment_telugu)
    elif short_name == "ar_padt" and dataset == 'train' and augment:
        write_augmented_dataset(input_conllu, input_conllu_copy, input_txt_copy, augment_arabic_padt)
    elif short_name.startswith("es_ancora") and dataset == 'train':
        # note that we always do this for AnCora, since this token is bizarre and confusing
        fix_spanish_ancora(input_conllu, input_conllu_copy, input_txt_copy, augment=augment)
    elif short_name.startswith("ko_") and short_name.endswith("_seg"):
        remove_spaces(input_conllu, input_conllu_copy, input_txt_copy)
    elif dataset == 'train':
        # we treat the additional punct as something that always needs to be there
        # this will teach the tagger & depparse about unicode apos, for example
        write_augmented_dataset(input_conllu, input_conllu_copy, input_txt_copy, augment_punct)
    else:
        shutil.copyfile(input_txt, input_txt_copy)
        shutil.copyfile(input_conllu, input_conllu_copy)

    if prepare_labels:
        prepare_dataset_labels(input_txt_copy, input_conllu_copy, tokenizer_dir, short_name, short_language, dataset)

def process_ud_treebank(treebank, udbase_dir, tokenizer_dir, short_name, short_language, augment=True, prepare_labels=True):
    """
    Process a normal UD treebank with train/dev/test splits

    SL-SSJ and other datasets with inline modifications all use this code path as well.
    """
    prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, "train", augment, prepare_labels)
    prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, "dev", augment, prepare_labels)
    prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, "test", augment, prepare_labels)


XV_RATIO = 0.2

def process_partial_ud_treebank(treebank, udbase_dir, tokenizer_dir, short_name, short_language, prepare_labels=True):
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

    if prepare_labels:
        prepare_dataset_labels(train_output_txt, train_output_conllu, tokenizer_dir, short_name, short_language, "train")
        prepare_dataset_labels(dev_output_txt, dev_output_conllu, tokenizer_dir, short_name, short_language, "dev")

    # the test set is already fine
    # currently we do not do any augmentation of these partial treebanks
    prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, "test", augment=False, prepare_labels=prepare_labels)

def add_specific_args(parser):
    parser.add_argument('--no_augment', action='store_false', dest='augment', default=True,
                        help='Augment the dataset in various ways')
    parser.add_argument('--no_prepare_labels', action='store_false', dest='prepare_labels', default=True,
                        help='Prepare tokenizer and MWT labels.  Expensive, but obviously necessary for training those models.')

def process_treebank(treebank, paths, args):
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
    handparsed_dir = paths["HANDPARSED_DIR"]

    short_name = common.project_to_short_name(treebank)
    short_language = short_name.split("_")[0]

    if short_name.startswith("ko_combined"):
        build_combined_korean(udbase_dir, tokenizer_dir, short_name, args.prepare_labels)
    elif short_name.startswith("it_combined"):
        build_combined_italian(udbase_dir, tokenizer_dir, handparsed_dir, short_name, args.prepare_labels)
    elif short_name.startswith("en_combined"):
        build_combined_english(udbase_dir, tokenizer_dir, handparsed_dir, short_name, args.prepare_labels)
    else:
        # check that we can find the train file where we expect it
        train_txt_file = common.find_treebank_dataset_file(treebank, udbase_dir, "train", "txt", fail=True)

        print("Preparing data for %s: %s, %s" % (treebank, short_name, short_language))

        if not common.find_treebank_dataset_file(treebank, udbase_dir, "dev", "txt", fail=False):
            process_partial_ud_treebank(treebank, udbase_dir, tokenizer_dir, short_name, short_language, args.prepare_labels)
        else:
            process_ud_treebank(treebank, udbase_dir, tokenizer_dir, short_name, short_language, args.augment, args.prepare_labels)


def main():
    common.main(process_treebank, add_specific_args)

if __name__ == '__main__':
    main()

