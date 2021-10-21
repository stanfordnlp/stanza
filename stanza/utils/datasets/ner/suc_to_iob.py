"""
Conversion tool to transform SUC3's xml format to IOB

Copyright 2017, Emil Stenstrom

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from bz2 import BZ2File
from xml.etree.ElementTree import iterparse
from enum import Enum, auto
import argparse

class ParseDetail(Enum):
    UNNAMED = auto()
    NAMED = auto()
    NAMED_DETAILED = auto()

def parse(fp, parse_detail=ParseDetail.NAMED, skiptypes=[], ner_tag="ne"):
    root = None
    ne_type = "O"
    ne_prefix = ""
    for event, elem in iterparse(fp, events=("start", "end")):
        if root is None:
            root = elem

        if event == "end" and elem.tag == "sentence":
            yield "\n"

        if event == "start" and elem.tag == ner_tag and elem.attrib["type"] not in skiptypes:
            if parse_detail == ParseDetail.UNNAMED:
                ne_type = "LABEL"
            elif parse_detail == ParseDetail.NAMED:
                ne_type = elem.attrib["type"]
            elif parse_detail == ParseDetail.NAMED_DETAILED:
                ne_type = elem.attrib["type"] + "-" + elem.attrib["subtype"]

            ne_prefix = "B-"

        if event == "end" and elem.tag == ner_tag:
            ne_type = "O"
            ne_prefix = ""

        if event == "end" and elem.tag == "w":
            yield elem.text + "\t" + ne_prefix + ne_type + "\n"

            if ne_type != "O":
                ne_prefix = "I-"

        root.clear()

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "infile",
        help="""
            Input for that contains the full SUC 3.0 XML.
            Can be the bz2-zipped version or the xml version.
        """
    )
    parser.add_argument(
        "outfile",
        help="Output file for IOB format."
    )
    parser.add_argument(
        "--ner_tag",
        help="XML tag to extract the NERs from.",
        choices=["ne", "name"],
        default="ne"
    )
    parser.add_argument(
        "--detail",
        help="Detail level that the file should be output in.",
        choices=[e.name for e in ParseDetail],
        default="NAMED"
    )
    parser.add_argument(
        "--skiptypes",
        help="Entity types that should be skipped in output.",
        nargs="+",
        default=[]
    )
    args = parser.parse_args(args=args)
    if args.ner_tag == 'name' and args.detail == 'NAMED_DETAILED':
        raise ValueError("<name> tags do not have subtypes in this dataset")

    MAGIC_BZ2_FILE_START = b"\x42\x5a\x68"
    fp = open(args.infile, "rb")
    is_bz2 = (fp.read(len(MAGIC_BZ2_FILE_START)) == MAGIC_BZ2_FILE_START)

    if is_bz2:
        fp = BZ2File(args.infile, "rb")
    else:
        fp = open(args.infile, "rb")

    with open(args.outfile, "w") as fout:
        for line in parse(fp, parse_detail=ParseDetail[args.detail], skiptypes=args.skiptypes, ner_tag=args.ner_tag):
            fout.write(line)

    fp.close()


if __name__ == '__main__':
    main()
