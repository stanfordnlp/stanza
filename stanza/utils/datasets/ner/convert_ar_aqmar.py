"""
A script to randomly shuffle the input files in the AQMAR dataset and produce train/dev/test for stanza

The sentences themselves are shuffled, not the data files

This script reads the input files directly from the .zip
"""


from collections import Counter
import random
import zipfile

from stanza.utils.datasets.ner.utils import write_dataset

def read_sentences(infile):
    """
    Read sentences from an open file
    """
    sents = []
    cache = []
    for line in infile:
        if isinstance(line, bytes):
            line = line.decode()
        line = line.rstrip()
        if len(line) == 0:
            if len(cache) > 0:
                sents.append(cache)
                cache = []
            continue
        array = line.split()
        assert len(array) == 2
        w, t = array
        cache.append([w, t])
    if len(cache) > 0:
        sents.append(cache)
        cache = []
    return sents


def normalize_tags(sents):
    new_sents = []
    # normalize tags
    for sent in sents:
        new_sentence = []
        for i, pair in enumerate(sent):
            w, t = pair
            if t.startswith('O'):
                new_t = 'O'
            elif t.startswith('I-'):
                type = t[2:]
                if type.startswith('MIS'):
                    new_t = 'I-MISC'
                elif type.startswith('-'): # handle I--ORG
                    new_t = 'I-' + type[1:]
                else:
                    new_t = t
            elif t.startswith('B-'):
                type = t[2:]
                if type.startswith('MIS'):
                    new_t = 'B-MISC'
                elif type.startswith('ENGLISH') or type.startswith('SPANISH'):
                    new_t = 'O'
                else:
                    new_t = t
            else:
                new_t = 'O'
            # modify original tag
            new_sentence.append((sent[i][0], new_t))
        new_sents.append(new_sentence)
    return new_sents


def convert_shuffle(base_input_path, base_output_path, short_name):
    """
    Convert AQMAR to a randomly shuffled dataset

    base_input_path is the zip file.  base_output_path is the output directory
    """
    if not zipfile.is_zipfile(base_input_path):
        raise FileNotFoundError("Expected %s to be the zipfile with AQMAR in it" % base_input_path)

    with zipfile.ZipFile(base_input_path) as zin:
        namelist = zin.namelist()
        annotation_files = [x for x in namelist if x.endswith(".txt") and not "/" in x]
        annotation_files = sorted(annotation_files)

        # although not necessary for good results, this does put
        # things in the same order the shell was alphabetizing files
        # when the original models were created for Stanza
        assert annotation_files[2] == 'Computer.txt'
        assert annotation_files[3] == 'Computer_Software.txt'
        annotation_files[2], annotation_files[3] = annotation_files[3], annotation_files[2]

        if len(annotation_files) != 28:
            raise RuntimeError("Expected exactly 28 labeled .txt files in %s but got %d" % (base_input_path, len(annotation_files)))

        sentences = []
        for in_filename in annotation_files:
            with zin.open(in_filename) as infile:
                new_sentences = read_sentences(infile)
            print(f"{len(new_sentences)} sentences read from {in_filename}")

            new_sentences = normalize_tags(new_sentences)
            sentences.extend(new_sentences)

    all_tags = Counter([p[1] for sent in sentences for p in sent])
    print("All tags after normalization:")
    print(list(all_tags.keys()))

    num = len(sentences)
    train_num = int(num*0.7)
    dev_num = int(num*0.15)

    random.seed(1234)

    random.shuffle(sentences)

    train_sents = sentences[:train_num]
    dev_sents = sentences[train_num:train_num+dev_num]
    test_sents = sentences[train_num+dev_num:]

    shuffled_dataset = [train_sents, dev_sents, test_sents]

    write_dataset(shuffled_dataset, base_output_path, short_name)

