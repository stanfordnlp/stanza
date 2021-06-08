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
import subprocess
import tempfile

from collections import Counter

import stanza.utils.datasets.common as common
import stanza.utils.datasets.prepare_tokenizer_data as prepare_tokenizer_data


def copy_conllu_file(tokenizer_dir, tokenizer_file, dest_dir, dest_file, short_name):
    original = f"{tokenizer_dir}/{short_name}.{tokenizer_file}.conllu"
    copied = f"{dest_dir}/{short_name}.{dest_file}.conllu"

    shutil.copyfile(original, copied)

def copy_conllu_treebank(treebank, paths, dest_dir, postprocess=None, augment=True):
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
        args.augment = augment
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
                     train_output_conllu, dev_output_conllu):
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

    return True

def mwt_name(base_dir, short_name, dataset):
    return f"{base_dir}/{short_name}-ud-{dataset}-mwt.json"

def prepare_dataset_labels(input_txt, input_conllu, tokenizer_dir, short_name, dataset):
    prepare_tokenizer_data.main([input_txt,
                                 input_conllu,
                                 "-o", f"{tokenizer_dir}/{short_name}-ud-{dataset}.toklabels",
                                 "-m", mwt_name(tokenizer_dir, short_name, dataset)])

def prepare_treebank_labels(tokenizer_dir, short_name):
    for dataset in ("train", "dev", "test"):
        output_txt = f"{tokenizer_dir}/{short_name}.{dataset}.txt"
        output_conllu = f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu"
        prepare_dataset_labels(output_txt, output_conllu, tokenizer_dir, short_name, dataset)

CONLLU_TO_TXT_PERL = os.path.join(os.path.split(__file__)[0], "conllu_to_text.pl")

def convert_conllu_to_txt(tokenizer_dir, short_name):
    for dataset in ("train", "dev", "test"):
        output_conllu = f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu"
        output_txt = f"{tokenizer_dir}/{short_name}.{dataset}.txt"

        # use an external script to produce the txt files
        subprocess.check_output(f"perl {CONLLU_TO_TXT_PERL} {output_conllu} > {output_txt}", shell=True)


# RE to see if the index of a conllu line represents an MWT
MWT_RE = re.compile("^[0-9]+[-][0-9]+")

# RE to see if the index of a conllu line represents an MWT or copy node
MWT_OR_COPY_RE = re.compile("^[0-9]+[-.][0-9]+")

# more restrictive than an actual int as we expect certain formats in the conllu files
INT_RE = re.compile("^[0-9]+$")

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


def has_space_after_no(piece):
    if not piece or piece == "_":
        return False
    if piece == "SpaceAfter=No":
        return True
    tags = piece.split("|")
    return any(t == "SpaceAfter=No" for t in tags)


def remove_space_after_no(piece, fail_if_missing=True):
    """
    Removes a SpaceAfter=No annotation from a single piece of a single word.
    In other words, given a list of conll lines, first call split("\t"), then call this on the -1 column
    """
    # |SpaceAfter is in UD_Romanian-Nonstandard... seems fitting
    if piece == "SpaceAfter=No" or piece == "|SpaceAfter=No":
        piece = "_"
    elif piece.startswith("SpaceAfter=No|"):
        piece = piece.replace("SpaceAfter=No|", "")
    elif piece.find("|SpaceAfter=No") > 0:
        piece = piece.replace("|SpaceAfter=No", "")
    elif fail_if_missing:
        raise ValueError("Could not find SpaceAfter=No in the given notes field")
    return piece

def add_space_after_no(piece, fail_if_found=True):
    if piece == '_':
        return "SpaceAfter=No"
    else:
        if fail_if_found:
            if has_space_after_no(piece):
                raise ValueError("Given notes field already contained SpaceAfter=No")
        return piece + "|SpaceAfter=No"


def augment_arabic_padt(sents, ratio=0.05):
    """
    Basic Arabic tokenizer gets the trailing punctuation wrong if there is a blank space.

    Reason seems to be that there are almost no examples of "text ." in the dataset.
    This function augments the Arabic-PADT dataset with a few such examples.
    TODO: it may very well be that a lot of tokeners have this problem.

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
        if random.random() > ratio:
            continue

        if (sentence[text_line][-1] in ('.', '؟', '?', '!') and
            sentence[text_line][-2] not in ('.', '؟', '?', '!', ' ') and
            has_space_after_no(sentence[-2].split()[-1]) and
            len(sentence[-1].split()[1]) == 1):
            new_sent = list(sentence)
            new_sent[text_line] = new_sent[text_line][:-1] + ' ' + new_sent[text_line][-1]
            pieces = sentence[-2].split("\t")
            pieces[-1] = remove_space_after_no(pieces[-1])
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
def augment_comma_separations(sents):
    """Find some fraction of the sentences which match "asdf, zzzz" and squish them to "asdf,zzzz"

    This leaves the tokens and all of the other data the same.  The
    only change made is to change SpaceAfter=No for the "," token and
    adjust the #text line, with the assumption that the conllu->txt
    conversion will correctly handle this change.

    This was particularly an issue for Spanish-AnCora, but it's
    reasonable to think it could happen to any dataset.  Currently
    this just operates on commas and ascii letters to avoid
    accidentally squishing anything that shouldn't be squished.

    UD_Spanish-AnCora 2.7 had a problem is with this sentence:
    # orig_file_sentence 143#5
    In this sentence, there was a comma smashed next to a token.

    Fixing just this one sentence is not sufficient to tokenize
    "asdf,zzzz" as desired, so we also augment by some fraction where
    we have squished "asdf, zzzz" into "asdf,zzzz".

    This exact example was later fixed in UD 2.8, but it should still
    potentially be useful for compensating for typos.
    """
    new_sents = []
    for sentence in sents:
        for text_idx, text_line in enumerate(sentence):
            # look for the line that starts with "# text".
            # keep going until we find it, or silently ignore it
            # if the dataset isn't in that format
            if text_line.startswith("# text"):
                break
        else:
            continue

        match = COMMA_SEPARATED_RE.search(sentence[text_idx])
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
            pieces[-1] = add_space_after_no(pieces[-1])
            comma = "\t".join(pieces)
            new_sent = sentence[:idx+1] + [comma] + sentence[idx+2:]

            text_offset = sentence[text_idx].find(match.group(1) + ", " + match.group(2))
            text_len = len(match.group(1) + ", " + match.group(2))
            new_text = sentence[text_idx][:text_offset] + match.group(1) + "," + match.group(2) + sentence[text_idx][text_offset+text_len:]
            new_sent[text_idx] = new_text

            new_sents.append(new_sent)

    print("Added %d new sentences with asdf, zzzz -> asdf,zzzz" % len(new_sents))
            
    return sents + new_sents

def augment_move_comma(sents, ratio=0.02):
    """
    Move the comma from after a word to before the next word some fraction of the time

    We looks for this exact pattern:
      w1, w2
    and replace it with
      w1 ,w2

    The idea is that this is a relatively common typo, but the tool
    won't learn how to tokenize it without some help.

    Note that this modification replaces the original text.
    """
    new_sents = []
    num_operations = 0
    for sentence in sents:
        if random.random() > ratio:
            new_sents.append(sentence)
            continue

        found = False
        for word_idx, word in enumerate(sentence):
            if word.startswith("#"):
                continue
            if word_idx == 0 or word_idx >= len(sentence) - 2:
                continue
            pieces = word.split("\t")
            if pieces[1] == ',' and not has_space_after_no(pieces[-1]):
                # found a comma with a space after it
                prev_word = sentence[word_idx-1]
                if not has_space_after_no(prev_word.split("\t")[-1]):
                    # unfortunately, the previous word also had a
                    # space after it.  does not fit what we are
                    # looking for
                    continue
                # also, want to skip instances near MWT or copy nodes,
                # since those are harder to rearrange
                next_word = sentence[word_idx+1]
                if MWT_OR_COPY_RE.match(next_word.split("\t")[0]):
                    continue
                if MWT_OR_COPY_RE.match(prev_word.split("\t")[0]):
                    continue
                # at this point, the previous word has no space and the comma does
                found = True
                break

        if not found:
            new_sents.append(sentence)
            continue

        new_sentence = list(sentence)

        pieces = new_sentence[word_idx].split("\t")
        pieces[-1] = add_space_after_no(pieces[-1])
        new_sentence[word_idx] = "\t".join(pieces)

        pieces = new_sentence[word_idx-1].split("\t")
        prev_word = pieces[1]
        pieces[-1] = remove_space_after_no(pieces[-1])
        new_sentence[word_idx-1] = "\t".join(pieces)

        next_word = new_sentence[word_idx+1].split("\t")[1]

        for text_idx, text_line in enumerate(sentence):
            # look for the line that starts with "# text".
            # keep going until we find it, or silently ignore it
            # if the dataset isn't in that format
            if text_line.startswith("# text"):
                old_chunk = prev_word + ", " + next_word
                new_chunk = prev_word + " ," + next_word
                word_idx = text_line.find(old_chunk)
                if word_idx < 0:
                    raise RuntimeError("Unexpected #text line which did not contain the original text to be modified.  Looking for\n" + old_chunk + "\n" + text_line)
                new_text_line = text_line[:word_idx] + new_chunk + text_line[word_idx+len(old_chunk):]
                new_sentence[text_idx] = new_text_line
                break

        new_sents.append(new_sentence)
        num_operations = num_operations + 1

    print("Swapped 'w1, w2' for 'w1 ,w2' %d times" % num_operations)
    return new_sents

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

# https://en.wikipedia.org/wiki/Quotation_mark
QUOTES = ['"', '“', '”', '«', '»', '「', '」', '《', '》', '„', '″']
QUOTES_RE = re.compile("(.?)[" + "".join(QUOTES) + "](.+)[" + "".join(QUOTES) + "](.?)")
# Danish does '«' the other way around from most European languages
START_QUOTES = ['"', '“', '”', '«', '»', '「', '《', '„', '„', '″']
END_QUOTES   = ['"', '“', '”', '»', '«', '」', '》', '”', '“', '″']

def augment_quotes(sents, ratio=0.15):
    """
    Go through the sentences and replace a fraction of sentences with alternate quotes

    TODO: for certain languages we may want to make some language-specific changes
      eg Danish, don't add «...»
    """
    assert len(START_QUOTES) == len(END_QUOTES)

    counts = Counter()
    new_sents = []
    for sent in sents:
        if random.random() > ratio:
            new_sents.append(sent)
            continue

        # count if there are exactly 2 quotes in this sentence
        # this is for convenience - otherwise we need to figure out which pairs go together
        count_quotes = sum(1 for x in sent
                           if (not x.startswith("#") and
                               x.split("\t")[1] in QUOTES))
        if count_quotes != 2:
            new_sents.append(sent)
            continue

        # choose a pair of quotes from the candidates
        quote_idx = random.choice(range(len(START_QUOTES)))
        start_quote = START_QUOTES[quote_idx]
        end_quote = END_QUOTES[quote_idx]
        counts[start_quote + end_quote] = counts[start_quote + end_quote] + 1

        new_sent = []
        saw_start = False
        for line in sent:
            if line.startswith("#"):
                new_sent.append(line)
                continue
            pieces = line.split("\t")
            if pieces[1] in QUOTES:
                if saw_start:
                    # Note that we don't change the lemma.  Presumably it's
                    # set to the correct lemma for a quote for this treebank
                    pieces[1] = end_quote
                else:
                    pieces[1] = start_quote
                    saw_start = True
                new_sent.append("\t".join(pieces))
            else:
                new_sent.append(line)

        for text_idx, text_line in enumerate(new_sent):
            # look for the line that starts with "# text".
            # keep going until we find it, or silently ignore it
            # if the dataset isn't in that format
            if text_line.startswith("# text"):
                replacement = "\\1%s\\2%s\\3" % (start_quote, end_quote)
                new_text_line = QUOTES_RE.sub(replacement, text_line)
                new_sent[text_idx] = new_text_line

        new_sents.append(new_sent)

    print("Augmented {} quotes: {}".format(sum(counts.values()), counts))
    return new_sents

def find_text_idx(sentence):
    """
    Return the index of the # text line or -1
    """
    for idx, line in enumerate(sentence):
        if line.startswith("# text"):
            return idx
    return -1

def change_indices(line, delta):
    """
    Adjust all indices in the given sentence by delta.  Useful when removing a word, for example
    """
    if line.startswith("#"):
        return line

    pieces = line.split("\t")
    if MWT_RE.match(pieces[0]):
        indices = pieces[0].split("-")
        pieces[0] = "%d-%d" % (int(indices[0]) + delta, int(indices[1]) + delta)
        line = "\t".join(pieces)
        return line

    if MWT_OR_COPY_RE.match(pieces[0]):
        raise NotImplementedError("Need to implement change_indices for copy nodes")

    if not INT_RE.match(pieces[0]):
        raise NotImplementedError("Unknown index type: %s" % pieces[0])

    pieces[0] = str(int(pieces[0]) + delta)
    dep = int(pieces[6])
    if dep != 0:
        pieces[6] = str(int(dep) + delta)
    if pieces[8] != '_':
        raise NotImplementedError("Need to handle the additional deps field in change_indices")
    line = "\t".join(pieces)
    return line

def augment_initial_punct(sents, ratio=0.20):
    """
    If a sentence starts with certain punct marks, occasionally use the same sentence without the initial punct.

    Currently this just handles ¿
    This helps languages such as CA and ES where the models go awry when the initial ¿ is missing.
    """
    new_sents = []
    for sent in sents:
        if random.random() > ratio:
            continue

        text_idx = find_text_idx(sent)
        text_line = sent[text_idx]
        if text_line.count("¿") != 1:
            # only handle sentences with exactly one ¿
            continue

        # find the first line with actual text
        for idx, line in enumerate(sent):
            if line.startswith("#"):
                continue
            break
        if idx >= len(sent) - 1:
            raise ValueError("Unexpectedly an entire sentence is comments")
        pieces = line.split("\t")
        if pieces[1] != '¿':
            continue
        if has_space_after_no(pieces[-1]):
            replace_text = "¿"
        else:
            replace_text = "¿ "

        new_sent = sent[:idx] + sent[idx+1:]
        new_sent[text_idx] = text_line.replace(replace_text, "")

        # now need to update all indices
        new_sent = [change_indices(x, -1) for x in new_sent]
        new_sents.append(new_sent)

    if len(new_sents) > 0:
        print("Added %d sentences with the leading ¿ removed" % len(new_sents))

    return sents + new_sents


def augment_punct(sents):
    """
    If there are no instances of ’ in the dataset, but there are instances of ',
    we replace some fraction of ' with ’ so that the tokenizer will recognize it.

    Also augments with ... / …
    """
    new_sents = augment_apos(sents)
    new_sents = augment_quotes(new_sents)
    new_sents = augment_move_comma(new_sents)
    new_sents = augment_comma_separations(new_sents)
    new_sents = augment_initial_punct(new_sents)
    new_sents = augment_ellipses(new_sents)

    return new_sents



def write_augmented_dataset(input_conllu, output_conllu, augment_function):
    # set the seed for each data file so that the results are the same
    # regardless of how many treebanks are processed at once
    random.seed(1234)

    # read and shuffle conllu data
    sents = read_sentences_from_conllu(input_conllu)

    # the actual meat of the function - produce new sentences
    new_sents = augment_function(sents)

    write_sentences_to_conllu(output_conllu, new_sents)

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

def remove_spaces(input_conllu, output_conllu):
    """
    Turns a dataset into something appropriate for building a segmenter.

    For example, this works well on the Korean datasets.
    """
    sents = read_sentences_from_conllu(input_conllu)

    new_sents = remove_spaces_from_sentences(sents)

    write_sentences_to_conllu(output_conllu, new_sents)


def build_combined_korean_dataset(udbase_dir, tokenizer_dir, short_name, dataset, output_conllu):
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

def build_combined_korean(udbase_dir, tokenizer_dir, short_name):
    for dataset in ("train", "dev", "test"):
        output_conllu = f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu"
        build_combined_korean_dataset(udbase_dir, tokenizer_dir, short_name, dataset, output_conllu)

def build_combined_italian_dataset(udbase_dir, tokenizer_dir, handparsed_dir, short_name, dataset):
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
    else:
        istd_conllu = common.find_treebank_dataset_file("UD_Italian-ISDT", udbase_dir, dataset, "conllu")
        sents = read_sentences_from_conllu(istd_conllu)

    return sents

def check_gum_ready(udbase_dir):
    gum_conllu = common.find_treebank_dataset_file("UD_English-GUMReddit", udbase_dir, "train", "conllu")
    if common.mostly_underscores(gum_conllu):
        raise ValueError("Cannot process UD_English-GUMReddit in its current form.  There should be a download script available in the directory which will help integrate the missing proprietary values.  Please run that script to update the data, then try again.")

def build_combined_english_dataset(udbase_dir, tokenizer_dir, handparsed_dir, short_name, dataset):
    """
    en_combined is currently EWT, GUM, PUD, and Pronouns

    TODO: use more of the handparsed data
    """
    check_gum_ready(udbase_dir)

    if dataset == 'train':
        # TODO: include more UD treebanks, possibly with xpos removed
        #  UD_English-ParTUT - xpos are different
        # also include "external" treebanks such as PTB
        # NOTE: in order to get the best results, make sure each of these treebanks have the latest edits applied
        train_treebanks = ["UD_English-EWT", "UD_English-GUM", "UD_English-GUMReddit"]
        test_treebanks = ["UD_English-PUD", "UD_English-Pronouns"]
        sents = []
        for treebank in train_treebanks:
            conllu_file = common.find_treebank_dataset_file(treebank, udbase_dir, "train", "conllu", fail=True)
            sents.extend(read_sentences_from_conllu(conllu_file))
        for treebank in test_treebanks:
            conllu_file = common.find_treebank_dataset_file(treebank, udbase_dir, "test", "conllu", fail=True)
            sents.extend(read_sentences_from_conllu(conllu_file))
    else:
        ewt_conllu = common.find_treebank_dataset_file("UD_English-EWT", udbase_dir, dataset, "conllu")
        sents = read_sentences_from_conllu(ewt_conllu)

    sents = strip_mwt_from_sentences(sents)
    return sents


def replace_semicolons(sentences):
    """
    Spanish GSD and AnCora have different standards for semicolons.

    GSD has semicolons at the end of sentences, AnCora has them in the middle as clause separators.
    Consecutive sentences in GSD do not seem to be related, so there is no combining that can be done.
    The easiest solution is to replace sentence final semicolons with "." in GSD
    """
    new_sents = []
    count = 0
    for sentence in sentences:
        for text_idx, text_line in enumerate(sentence):
            if text_line.startswith("# text"):
                break
        else:
            raise ValueError("Expected every sentence in GSD to have a # text field")
        if not text_line.endswith(";"):
            new_sents.append(sentence)
            continue
        count = count + 1
        new_sent = list(sentence)
        new_sent[text_idx] = text_line[:-1] + "."
        new_sent[-1] = new_sent[-1].replace(";", ".")
        count = count + 1
        new_sents.append(new_sent)
    print("Updated %d sentences to replace sentence-final ; with ." % count)
    return new_sents

def build_combined_spanish_dataset(udbase_dir, tokenizer_dir, handparsed_dir, short_name, dataset):
    """
    es_combined is AnCora and GSD put together

    TODO: remove features which aren't shared between datasets
    TODO: consider mixing in PUD?
    """
    if dataset == 'train':
        treebanks = ["UD_Spanish-AnCora", "UD_Spanish-GSD"]
        sents = []
        for treebank in treebanks:
            conllu_file = common.find_treebank_dataset_file(treebank, udbase_dir, dataset, "conllu", fail=True)
            new_sents = read_sentences_from_conllu(conllu_file)
            if treebank.endswith("GSD"):
                new_sents = replace_semicolons(new_sents)
            sents.extend(new_sents)

        extra_spanish = os.path.join(handparsed_dir, "spanish-mwt", "spanish.mwt")
        if not os.path.exists(extra_spanish):
            raise FileNotFoundError("Cannot find the extra dataset 'spanish.mwt' which includes various multi-words retokenized, expected {}".format(extra_italian))
        extra_sents = read_sentences_from_conllu(extra_spanish)
        sents.extend(extra_sents)
    else:
        conllu_file = common.find_treebank_dataset_file("UD_Spanish-AnCora", udbase_dir, dataset, "conllu", fail=True)
        sents = read_sentences_from_conllu(conllu_file)

    return sents


COMBINED_FNS = {
    "en_combined": build_combined_english_dataset,
    "es_combined": build_combined_spanish_dataset,
    "it_combined": build_combined_italian_dataset,
}

def build_combined_dataset(udbase_dir, tokenizer_dir, handparsed_dir, short_name, augment):
    random.seed(1234)
    build_fn = COMBINED_FNS[short_name]
    for dataset in ("train", "dev", "test"):
        output_conllu = f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu"
        sents = build_fn(udbase_dir, tokenizer_dir, handparsed_dir, short_name, dataset)
        if dataset == 'train' and augment:
            sents = augment_punct(sents)
        write_sentences_to_conllu(output_conllu, sents)

def build_combined_english_gum_dataset(udbase_dir, tokenizer_dir, short_name, dataset, augment):
    """
    Build the GUM dataset by combining GUMReddit

    It checks to make sure GUMReddit is filled out using the included script
    """
    check_gum_ready(udbase_dir)
    random.seed(1234)

    output_conllu = f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu"

    treebanks = ["UD_English-GUM", "UD_English-GUMReddit"]
    sents = []
    for treebank in treebanks:
        conllu_file = common.find_treebank_dataset_file(treebank, udbase_dir, dataset, "conllu", fail=True)
        sents.extend(read_sentences_from_conllu(conllu_file))

    if dataset == 'train' and augment:
        sents = augment_punct(sents)

    write_sentences_to_conllu(output_conllu, sents)

def build_combined_english_gum(udbase_dir, tokenizer_dir, short_name, augment):
    for dataset in ("train", "dev", "test"):
        build_combined_english_gum_dataset(udbase_dir, tokenizer_dir, short_name, dataset, augment)

def prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, dataset, augment=True):
    input_conllu = common.find_treebank_dataset_file(treebank, udbase_dir, dataset, "conllu")
    output_conllu = f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu"

    if short_name == "te_mtg" and dataset == 'train' and augment:
        write_augmented_dataset(input_conllu, output_conllu, augment_telugu)
    elif short_name == "ar_padt" and dataset == 'train' and augment:
        write_augmented_dataset(input_conllu, output_conllu, augment_arabic_padt)
    elif short_name.startswith("ko_") and short_name.endswith("_seg"):
        remove_spaces(input_conllu, output_conllu)
    elif dataset == 'train' and augment:
        write_augmented_dataset(input_conllu, output_conllu, augment_punct)
    else:
        shutil.copyfile(input_conllu, output_conllu)

def process_ud_treebank(treebank, udbase_dir, tokenizer_dir, short_name, short_language, augment=True):
    """
    Process a normal UD treebank with train/dev/test splits

    SL-SSJ and other datasets with inline modifications all use this code path as well.
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
    dev_output_conllu = f"{tokenizer_dir}/{short_name}.dev.gold.conllu"

    if not split_train_file(treebank=treebank,
                            train_input_conllu=train_input_conllu,
                            train_output_conllu=train_output_conllu,
                            dev_output_conllu=dev_output_conllu):
        return

    # the test set is already fine
    # currently we do not do any augmentation of these partial treebanks
    prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, "test", augment=False)

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

    os.makedirs(tokenizer_dir, exist_ok=True)

    if short_name.startswith("ko_combined"):
        build_combined_korean(udbase_dir, tokenizer_dir, short_name)
    elif short_name in ("it_combined", "en_combined", "es_combined"):
        build_combined_dataset(udbase_dir, tokenizer_dir, handparsed_dir, short_name, args.augment)
    elif short_name.startswith("en_gum"):
        # we special case GUM because it should include a filled-out GUMReddit
        print("Preparing data for %s: %s, %s" % (treebank, short_name, short_language))
        build_combined_english_gum(udbase_dir, tokenizer_dir, short_name, args.augment)
    else:
        # check that we can find the train file where we expect it
        train_conllu_file = common.find_treebank_dataset_file(treebank, udbase_dir, "train", "conllu", fail=True)

        print("Preparing data for %s: %s, %s" % (treebank, short_name, short_language))

        if not common.find_treebank_dataset_file(treebank, udbase_dir, "dev", "conllu", fail=False):
            process_partial_ud_treebank(treebank, udbase_dir, tokenizer_dir, short_name, short_language)
        else:
            process_ud_treebank(treebank, udbase_dir, tokenizer_dir, short_name, short_language, args.augment)

    convert_conllu_to_txt(tokenizer_dir, short_name)

    if args.prepare_labels:
        prepare_treebank_labels(tokenizer_dir, short_name)


def main():
    common.main(process_treebank, add_specific_args)

if __name__ == '__main__':
    main()

