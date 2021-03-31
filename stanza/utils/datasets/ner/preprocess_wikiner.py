"""
Converts the WikiNER data format to a format usable by our processing tools

python preprocess_wikiner input output
"""

import sys

def preprocess_wikiner(input_file, output_file):
    with open(input_file) as fin:
        with open(output_file, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    fout.write("-DOCSTART- O\n")
                    fout.write("\n")
                    continue

                words = line.split()
                for word in words:
                    pieces = word.split("|")
                    text = pieces[0]
                    tag = pieces[-1]
                    # some words look like Daniel_Bernoulli|I-PER
                    # but the original .pl conversion script didn't take that into account
                    subtext = text.split("_")
                    if tag.startswith("B-") and len(subtext) > 1:
                        fout.write("{} {}\n".format(subtext[0], tag))
                        for chunk in subtext[1:]:
                            fout.write("{} I-{}\n".format(chunk, tag[2:]))
                    else:
                        for chunk in subtext:
                            fout.write("{} {}\n".format(chunk, tag))
                fout.write("\n")

if __name__ == '__main__':
    preprocess_wikiner(sys.argv[1], sys.argv[2])
