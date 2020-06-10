import glob
import random
import sys

import xml.etree.ElementTree as ET

from collections import namedtuple

import stanza

import scripts.sentiment.process_utils as process_utils

"""
This processes a Chinese corpus, hosted here:

http://a1-www.is.tokushima-u.ac.jp/member/ren/Ren-CECps1.0/Ren-CECps1.0.html

The authors want a signed document saying you won't redistribute the corpus.

The corpus format is a bunch of .xml files, with sentences labeled with various emotions and an overall polarity.  Polarity is labeled as follows:

消极: negative
中性: neutral
积极: positive
"""

Fragment = namedtuple('Fragment', ['sentiment', 'text'])

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
                    fragments.append(Fragment(sentiment, text))

    return fragments

def main():
    xml_directory = sys.argv[1]
    out_directory = sys.argv[2]
    sentences = []
    for filename in glob.glob(xml_directory + '/xml/cet_*xml'):
        sentences.extend(get_phrases(filename))

    nlp = stanza.Pipeline('zh', processors='tokenize')
    snippets = []
    for sentence in sentences:
        doc = nlp(sentence.text)
        text = " ".join(" ".join(token.text for token in sentence.tokens) for sentence in doc.sentences)
        snippets.append(sentence.sentiment + " " + text)

    print("Found {} phrases".format(len(snippets)))
    random.seed(1000)
    random.shuffle(snippets)
    process_utils.write_splits(out_directory,
                               snippets,
                               (process_utils.Split("train.txt", 0.8),
                                process_utils.Split("dev.txt", 0.1),
                                process_utils.Split("test.txt", 0.1)))


if __name__ == "__main__":
    main()
