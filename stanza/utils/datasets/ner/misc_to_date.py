# for the Worldwide dataset, automatically switch the Misc tags to Date when Stanza Ontonotes thinks it's a Date
# this keeps our annotation scheme for dates (eg, not "3 months ago") while hopefully switching them all to Date
#
# maybe some got missed
# also, there are a few with some nested entities.  printed out warnings and edited those by hand
#
# just need to run this with the Worldwide dataset in the ner path
# it will automatically convert as many as it can

import os

from tqdm import tqdm

import stanza
from stanza.utils.datasets.ner.utils import read_tsv
from stanza.utils.default_paths import get_default_paths

paths = get_default_paths()
BASE_PATH = os.path.join(paths["NERBASE"], "en_foreign")
input_dir = os.path.join(BASE_PATH, "en-foreign-newswire")

pipe = stanza.Pipeline("en", processors="tokenize,ner", tokenize_pretokenized=True, package={"ner": "ontonotes_bert"})

filenames = []

def ner_tags(pipe, sentence):
    doc = pipe([sentence])
    tags = [token.ner for sentence in doc.sentences for token in sentence.tokens]
    return tags

for root, dirs, files in os.walk(input_dir):
    if root[-6:] == "REVIEW":
        batch_files = os.listdir(root)
        for filename in batch_files:
            file_path = os.path.join(root, filename)
            filenames.append(file_path)

for filename in tqdm(filenames):
    try:
        data = read_tsv(filename, text_column=0, annotation_column=1, skip_comments=False, keep_all_columns=True)

        with open(filename, 'w', encoding='utf-8') as fout:
            warned_file = False
            for sentence in data:  # segments delimited by spaces, effectively sentences
                tokens = [x[0] for x in sentence]
                labels = [x[1] for x in sentence]

                if any(x.endswith("Misc") for x in labels):
                    stanza_tags = ner_tags(pipe, tokens)
                    in_date = False
                    for i, stanza_tag in enumerate(stanza_tags):
                        if stanza_tag[2:] == "DATE" and labels[i] != "O":
                            if len(sentence[i]) > 2:
                                if not warned_file:
                                    print("Warning: file %s has nested tags being altered" % filename)
                                    warned_file = True
                            # put DATE tags where Stanza thinks there are DATEs
                            # as long as we already had a MISC (or something else, I suppose)
                            if in_date and not stanza_tag[0].startswith("B") and not stanza_tag[0].startswith("S"):
                                sentence[i][1] = "I-Date"
                            else:
                                sentence[i][1] = "B-Date"
                            in_date = True
                        elif in_date:
                            # make sure new tags start with B- instead of I-
                            # honestly it's not clear if, in these cases,
                            # we should be switching the following tags to
                            # DATE as well. will have to experiment some
                            in_date = False
                            if labels[i].startswith("I-"):
                                sentence[i][1] = "B-" + labels[i][2:]
                for word in sentence:
                    fout.write("\t".join(word))
                    fout.write("\n")
                fout.write("\n")
    except AssertionError:
        print("Could not process %s" % filename)
