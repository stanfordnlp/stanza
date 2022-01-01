"""
Process a conll file into BIO

Includes the ability to process a file from a text file
or a text file within a zip

Main program extracts a piece of the zip file from the Danish DDT dataset
"""

import io
import zipfile
from zipfile import ZipFile
from stanza.utils.conll import CoNLL

def process_conll(input_file, output_file, zip_file=None, conversion=None):
    """
    Process a single file from DDT

    zip_filename: path to ddt.zip
    in_filename: which piece to read
    out_filename: where to write the result
    """
    doc = CoNLL.conll2doc(input_file=input_file, zip_file=zip_file)

    with open(output_file, "w", encoding="utf-8") as fout:
        for sentence_idx, sentence in enumerate(doc.sentences):
            for token_idx, token in enumerate(sentence.tokens):
                misc = token.misc.split("|")
                for attr in misc:
                    if attr.startswith("name="):
                        ner = attr.split("=", 1)[1]
                        break
                else:
                    raise ValueError("Could not find ner tag in document {}, sentence {}, token {}".format(input_file, sentence_idx, token_idx))
                if ner != "O" and conversion is not None:
                    bio, label = ner.split("-", 1)
                    if label in conversion:
                        label = conversion[label]
                    ner = "%s-%s" % (bio, label)
                fout.write("%s\t%s\n" % (token.text, ner))
            fout.write("\n")

def main():
    process_zip("extern_data/ner/da_ddt/ddt.zip", "ddt.train.conllu", "data/ner/da_ddt.train.bio")

if __name__ == '__main__':
    main()
