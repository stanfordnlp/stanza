from collections import namedtuple
import glob
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

Fragment = namedtuple('Fragment', ['begin', 'end', 'rating'])

"""
Extracts positive, neutral, and negative phrases from the ArguAna hotel review corpus

Run as follows:

python3 parse_arguana_xml.py split/training arguana_train.txt

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
            fragments.append(Fragment(begin=int(child.attrib['begin']),
                                      end=int(child.attrib['end']),
                                      rating="1"))
        elif child.tag == '{http:///de/aitools/ie/uima/type/arguana.ecore}Opinion':
            if child.attrib['polarity'] == 'negative':
                rating = "0"
            elif child.attrib['polarity'] == 'positive':
                rating = "2"
            else:
                raise ValueError("Unexpected polarity found in {}".format(filename))
            fragments.append(Fragment(begin=int(child.attrib['begin']),
                                      end=int(child.attrib['end']),
                                      rating=rating))


    phrases = [fragment.rating + " " + body[fragment.begin:fragment.end] for fragment in fragments]
    #phrases = [phrase.replace("\n", " ") for phrase in phrases]
    return phrases

def main():
    directory = sys.argv[1]
    out_filename = sys.argv[2]
    phrases = []
    for filename in glob.glob(directory + '/*/*xmi'):
        phrases.extend(get_phrases(filename))
    print("Found {} phrases".format(len(phrases)))
    tmp_filename = tempfile.NamedTemporaryFile(delete=False).name
    with open(tmp_filename, "w") as fout:
        for phrase in phrases:
            fout.write("%s\n" % (phrase))

    os.system("java edu.stanford.nlp.process.PTBTokenizer -preserveLines %s > %s" % (tmp_filename, out_filename))
    os.unlink(tmp_filename)    
    
if __name__ == "__main__":
    main()
