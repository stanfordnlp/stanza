import argparse
import os
import tempfile

import stanza
from stanza.models.common.utils import get_tqdm
from stanza.utils.default_paths import get_default_paths
from stanza.utils.datasets.ner.utils import read_tsv

tqdm = get_tqdm()

PUNCTUATION = """!"#%&'()*+, -./:;<=>?@[\]^_`{|}~"""
MONEY_WORDS = {"million", "billion", "trillion", "millions", "billions", "trillions", "hundred", "hundreds",
               "lakh", "crore", # south asian english
               "tens", "of", "ten", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "couple"}

# Doesn't include Money but this case is handled explicitly for processing
LABEL_TRANSLATION = {
    "Misc":         "MISC",
    "Product":      "MISC",
    "NORP":         "MISC",
    "Facility":     "LOC",
    "Location":     "LOC",
    "Person":       "PER",
    "Organization": "ORG",
}

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def process_label(line, is_start=False):
    """
    Converts our stuff to conll labels

    event, product, work of art, norp -> MISC
    take out dates - can use Stanza to identify them as dates and eliminate them
    money requires some special care
    facility -> location  (there are examples of Bridge and Hospital in the data)
    the version of conll we used to train CoreNLP NER is here:

    Overall plan:
    Collapse Product, NORP, Money (extract only the symbols), into misc.
    Collapse Facilities into LOC
    Deletes Dates

    Rule for currency is that we take out labels for the numbers that return True for isfloat()
    Take out words that categorize money (Million, Billion, Trillion, Thousand, Hundred, Ten, Nine, Eight, Seven, Six, Five,
    Four, Three, Two, One)
    Take out punctuation characters

    If we remove the 'B' tag, then move it to the first remaining tag.

    Replace tags with 'O'
    is_start parameter signals whether or not this current line is the new start of a tag. Needed for when
    the previous line analyzed is the start of a MONEY tag but is removed because it is a non symbol- need to
    set the starting token that is a symbol to the B-MONEY tag when it might have previously been I-MONEY
    """
    if not line:
        return []
    token = line[0]
    outermost_label = line[-1]
    position, label_name = outermost_label[:2], outermost_label[2:]

    if label_name == "Money":
        if token.lower() in MONEY_WORDS or token in PUNCTUATION or isfloat(token):  # remove this tag
            label_name = "O"
            is_start = True
            position = ""
        else:  # keep money tag
            label_name = "MISC"
            if is_start:
                position = "B-"
                is_start = False

    elif not label_name or label_name == "O":
        pass
    elif label_name in LABEL_TRANSLATION:
        label_name = LABEL_TRANSLATION[label_name]
    else:
        raise ValueError("Oops, missed a label: %s" % label_name)
    return [token, position + label_name, is_start]


def write_new_file(save_dir, input_path, old_file):
    starts_b = False
    with open(input_path, "r+", encoding="utf-8") as iob:
        new_filename = os.path.splitext(old_file)[0] + ".4class.tsv"
        with open(os.path.join(save_dir, new_filename), 'w', encoding='utf-8') as fout:
            for i, line in enumerate(iob):
                if i == 0 or i == 1:  # skip over the URL and subsequent space line.
                    continue
                line = line.strip()
                if not line:
                    fout.write("\n")
                    continue
                label = line.split("\t")
                try:
                    edited = process_label(label, is_start=starts_b)  # processed label line labels
                except ValueError as e:
                    raise ValueError("Error in %s at line %d" % (input_path, i)) from e
                assert edited
                starts_b = edited[-1]
                fout.write("\t".join(edited[:-1]))
                fout.write("\n")


def ner_tags(pipe, sentence):
    doc = pipe([sentence])
    tags = [token.ner for sentence in doc.sentences for token in sentence.tokens]
    return tags


def write_file_stanza(pipe, input_dir, output_dir, file_name):
    """
    REMOVES DATES ONLY! To collapse rest of labels use write_new_file
    """
    complete_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name)
    data = read_tsv(complete_path, text_column=0, annotation_column=1, keep_broken_tags=True)
    with open(output_path, 'w', encoding='utf-8') as fout:
        for segment in data:  # segments delimited by spaces
            tokens = [token for token, _ in segment]
            labels = [label for _, label in segment]

            if any(x.endswith("MISC") for x in labels):
                stanza_tags = ner_tags(pipe, tokens)
                just_removed = False
                for i, stanza_tag in enumerate(stanza_tags):
                    if stanza_tag[2:] == "DATE":
                        labels[i] = "O"  # remove date labels replace with empty
                        just_removed = True
                    elif just_removed:
                        # make sure new tags start with B- instead of I-
                        just_removed = False
                        if labels[i].startswith("I-"):
                            labels[i] = "B-" + labels[i][2:]
            for token, tag in zip(tokens, labels):
                fout.write("%s\t%s\n" % (token, tag))
            fout.write("\n")


def main(args=None):
    BASE_PATH = "C:\\Users\\SystemAdmin\\PycharmProjects\\General Code\\stanza source code"
    if not os.path.exists(BASE_PATH):
        paths = get_default_paths()
        BASE_PATH = os.path.join(paths["NERBASE"], "en_foreign")

    # TODO: use a temp dir for the intermediate files
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default=BASE_PATH, help="Where to find the raw data")
    args = parser.parse_args(args=args)

    BASE_PATH = args.base_path

    with tempfile.TemporaryDirectory(dir=BASE_PATH) as tempdir:
        # Condense Labels
        input_dir = os.path.join(BASE_PATH, "en-foreign-newswire")
        final_dir = os.path.join(BASE_PATH, "4class")
        os.makedirs(tempdir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)
        for root, dirs, files in os.walk(input_dir):
            if root[-6:] == "REVIEW":
                batch_files = os.listdir(root)
                for filename in batch_files:
                    file_path = os.path.join(root, filename)
                    write_new_file(tempdir, file_path, filename)

        # REMOVE DATES
        batch_files = os.listdir(tempdir)
        pipe = stanza.Pipeline("en", processors="tokenize,ner", tokenize_pretokenized=True)
        for filename in tqdm(batch_files):
            write_file_stanza(pipe, tempdir, final_dir, filename)

if __name__ == '__main__':
    main()



