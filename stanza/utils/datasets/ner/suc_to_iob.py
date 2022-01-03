"""
Conversion tool to transform SUC3's xml format to IOB

Copyright 2017-2022, Emil Stenström

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

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
import argparse
from collections import Counter
import sys

def parse(fp, skiptypes=[]):
    root = None
    ne_prefix = ""
    ne_type = "O"
    name_prefix = ""
    name_type = "O"

    for event, elem in iterparse(fp, events=("start", "end")):
        if root is None:
            root = elem

        if event == "start":
            if elem.tag == "name":
                _type = name_type_to_label(elem.attrib["type"])
                if (
                    _type not in skiptypes and
                    not (_type == "ORG" and ne_type == "LOC")
                ):
                    name_type = _type
                    name_prefix = "B-"

            elif elem.tag == "ne":
                _type = ne_type_to_label(elem.attrib["type"])
                if "/" in _type:
                    _type = ne_type_to_label(_type[_type.index("/") + 1:])

                if _type not in skiptypes:
                    ne_type = _type
                    ne_prefix = "B-"

            elif elem.tag == "w":
                if name_type == "PER" and elem.attrib["pos"] == "NN":
                    name_type = "O"
                    name_prefix = ""

        elif event == "end":
            if elem.tag == "sentence":
                yield

            elif elem.tag == "name":
                name_type = "O"
                name_prefix = ""

            elif elem.tag == "ne":
                ne_type = "O"
                ne_prefix = ""

            elif elem.tag == "w":
                if name_type != "O" and name_type != "OTH":
                    yield elem.text, name_prefix, name_type
                elif ne_type != "O":
                    yield elem.text, ne_prefix, ne_type
                else:
                    yield elem.text, "", "O"

                if ne_type != "O":
                    ne_prefix = "I-"

                if name_type != "O":
                    name_prefix = "I-"

        root.clear()

def ne_type_to_label(ne_type):
    mapping = {
        "PRS": "PER",
    }
    return mapping.get(ne_type, ne_type)

def name_type_to_label(name_type):
    mapping = {
        "inst": "ORG",
        "product": "OBJ",
        "other": "OTH",
        "place": "LOC",
        "myth": "PER",
        "person": "PER",
        "event": "EVN",
        "work": "WRK",
        "animal": "PER",
    }
    return mapping.get(name_type)

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
        nargs="?",
        help="""
             Output file for IOB format.
             Optional - will print to stdout otherwise
        """
    )
    parser.add_argument(
        "--skiptypes",
        help="Entity types that should be skipped in output.",
        nargs="+",
        default=[]
    )
    parser.add_argument(
        "--stats_only",
        help="Show statistics of found labels at the end of output.",
        action='store_true',
        default=False
    )
    args = parser.parse_args(args)

    MAGIC_BZ2_FILE_START = b"\x42\x5a\x68"
    fp = open(args.infile, "rb")
    is_bz2 = (fp.read(len(MAGIC_BZ2_FILE_START)) == MAGIC_BZ2_FILE_START)

    if is_bz2:
        fp = BZ2File(args.infile, "rb")
    else:
        fp = open(args.infile, "rb")

    if args.outfile is not None:
        fout = open(args.outfile, "w", encoding="utf-8")
    else:
        fout = sys.stdout

    type_stats = Counter()
    for token in parse(fp, skiptypes=args.skiptypes):
        if not token:
            if not args.stats_only:
                fout.write("\n")
        else:
            word, prefix, label = token
            if args.stats_only:
                type_stats[label] += 1
            else:
                fout.write("%s\t%s%s\n" % (word, prefix, label))

    if args.stats_only:
        fout.write(str(type_stats) + "\n")

    fp.close()
    if args.outfile is not None:
        fout.close()


if __name__ == '__main__':
    main()
