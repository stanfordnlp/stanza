import glob
import os
import random
import sys

import xml.etree.ElementTree as ET

from collections import namedtuple

import stanza

from stanza.utils.datasets.sentiment.process_utils import SentimentDatum
import stanza.utils.datasets.sentiment.process_utils as process_utils

"""
This processes a Chinese corpus, hosted here:

http://a1-www.is.tokushima-u.ac.jp/member/ren/Ren-CECps1.0/Ren-CECps1.0.html

The authors want a signed document saying you won't redistribute the corpus.

The corpus format is a bunch of .xml files, with sentences labeled with various emotions and an overall polarity.  Polarity is labeled as follows:

消极: negative
中性: neutral
积极: positive
"""

def get_phrases(filename):
    tree = ET.parse(filename)
    fragments = []

    root = tree.getroot()
    for child in root:
        if child.tag == 'paragraph':
            for subchild in child:
                if subchild.tag == 'sentence':
                    text = subchild.attrib['S'].strip()
                    if len(text) <= 2:
                        continue
                    polarity = None
                    for inner in subchild:
                        if inner.tag == 'Polarity':
                            polarity = inner
                            break
                    if polarity is None:
                        print("Found sentence with no polarity in {}: {}".format(filename, text))
                        continue
                    if polarity.text == '消极':
                        sentiment = "0"
                    elif polarity.text == '中性':
                        sentiment = "1"
                    elif polarity.text == '积极':
                        sentiment = "2"
                    else:
                        raise ValueError("Unknown polarity {} in {}".format(polarity.text, filename))
                    fragments.append(SentimentDatum(sentiment, text))

    return fragments

def read_snippets(xml_directory):
    sentences = []
    for filename in glob.glob(xml_directory + '/xml/cet_*xml'):
        sentences.extend(get_phrases(filename))

    nlp = stanza.Pipeline('zh', processors='tokenize')
    snippets = []
    for sentence in sentences:
        doc = nlp(sentence.text)
        text = [token.text for sentence in doc.sentences for token in sentence.tokens]
        snippets.append(SentimentDatum(sentence.sentiment, text))
    random.shuffle(snippets)
    return snippets

def main(xml_directory, out_directory, short_name):
    snippets = read_snippets(xml_directory)

    print("Found {} phrases".format(len(snippets)))
    os.makedirs(out_directory, exist_ok=True)
    process_utils.write_splits(out_directory,
                               snippets,
                               (process_utils.Split("%s.train.json" % short_name, 0.8),
                                process_utils.Split("%s.dev.json" % short_name, 0.1),
                                process_utils.Split("%s.test.json" % short_name, 0.1)))


if __name__ == "__main__":
    random.seed(1234)
    xml_directory = sys.argv[1]
    out_directory = sys.argv[2]
    short_name = sys.argv[3]
    main(xml_directory, out_directory, short_name)

