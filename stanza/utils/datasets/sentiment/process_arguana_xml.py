from collections import namedtuple
import glob
import os
import sys
import xml.etree.ElementTree as ET

from stanza.utils.datasets.sentiment.process_utils import SentimentDatum
import stanza.utils.datasets.sentiment.process_utils as process_utils

ArguanaSentimentDatum = namedtuple('ArguanaSentimentDatum', ['begin', 'end', 'rating'])

"""
Extracts positive, neutral, and negative phrases from the ArguAna hotel review corpus

Run as follows:

python3 parse_arguana_xml.py split/training data/sentiment

ArguAna can be downloaded here:

http://argumentation.bplaced.net/arguana/data
http://argumentation.bplaced.net/arguana-data/arguana-tripadvisor-annotated-v2.zip
"""

def get_phrases(filename):
    tree = ET.parse(filename)
    fragments = []

    root = tree.getroot()
    body = None
    for child in root:
        if child.tag == '{http:///uima/cas.ecore}Sofa':
            body = child.attrib['sofaString']
        elif child.tag == '{http:///de/aitools/ie/uima/type/arguana.ecore}Fact':
            fragments.append(ArguanaSentimentDatum(begin=int(child.attrib['begin']),
                                                   end=int(child.attrib['end']),
                                                   rating="1"))
        elif child.tag == '{http:///de/aitools/ie/uima/type/arguana.ecore}Opinion':
            if child.attrib['polarity'] == 'negative':
                rating = "0"
            elif child.attrib['polarity'] == 'positive':
                rating = "2"
            else:
                raise ValueError("Unexpected polarity found in {}".format(filename))
            fragments.append(ArguanaSentimentDatum(begin=int(child.attrib['begin']),
                                                   end=int(child.attrib['end']),
                                                   rating=rating))


    phrases = [SentimentDatum(fragment.rating, body[fragment.begin:fragment.end]) for fragment in fragments]
    #phrases = [phrase.replace("\n", " ") for phrase in phrases]
    return phrases

def get_phrases_from_directory(directory):
    phrases = []
    inpath = os.path.join(directory, "arguana-tripadvisor-annotated-v2", "split", "training", "*", "*xmi")
    for filename in glob.glob(inpath):
        phrases.extend(get_phrases(filename))
    return phrases

def get_tokenized_phrases(in_directory):
    phrases = get_phrases_from_directory(in_directory)
    phrases = process_utils.get_ptb_tokenized_phrases(phrases)
    print("Found {} phrases in arguana".format(len(phrases)))
    return phrases

def main(in_directory, out_directory, short_name):
    phrases = get_tokenized_phrases(in_directory)
    process_utils.write_list(os.path.join(out_directory, "%s.train.json" % short_name), phrases)


if __name__ == "__main__":
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    short_name = sys.argv[3]
    main(in_directory, out_directory, short_name)
