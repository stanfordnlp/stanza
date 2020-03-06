#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple shell program to pipe in 
"""

import corenlp

import json
import re
import csv
import sys
from collections import namedtuple, OrderedDict

FLOAT_RE = re.compile(r"\d*\.\d+")
INT_RE = re.compile(r"\d+")

def dictstr(arg):
    """
    Parse a key=value string as a tuple (key, value) that can be provided as an argument to dict()
    """
    key, value = arg.split("=")

    if value.lower() == "true" or value.lower() == "false":
        value = bool(value)
    elif INT_RE.match(value):
        value = int(value)
    elif FLOAT_RE.match(value):
        value = float(value)
    return (key, value)


def do_annotate(args):
    args.props = dict(args.props) if args.props else {}
    if args.sentence_mode:
        args.props["ssplit.isOneSentence"] = True

    with corenlp.CoreNLPClient(annotators=args.annotators, properties=args.props, be_quiet=not args.verbose_server) as client:
        for line in args.input:
            if line.startswith("#"): continue

            ann = client.annotate(line.strip(), output_format=args.format)

            if args.format == "json":
                if args.sentence_mode:
                    ann = ann["sentences"][0]

                args.output.write(json.dumps(ann))
                args.output.write("\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Annotate data')
    parser.add_argument('-i', '--input', type=argparse.FileType('r'), default=sys.stdin, help="Input file to process; each line contains one document (default: stdin)")
    parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="File to write annotations to (default: stdout)")
    parser.add_argument('-f', '--format', choices=["json",], default="json", help="Output format")
    parser.add_argument('-a', '--annotators', nargs="+", type=str, default=["tokenize ssplit lemma pos"], help="A list of annotators")
    parser.add_argument('-s', '--sentence-mode', action="store_true",help="Assume each line of input is a sentence.")
    parser.add_argument('-v', '--verbose-server', action="store_true",help="Server is made verbose")
    parser.add_argument('-m', '--memory', type=str, default="4G", help="Memory to use for the server")
    parser.add_argument('-p', '--props', nargs="+", type=dictstr, help="Properties as a list of key=value pairs")
    parser.set_defaults(func=do_annotate)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)

if __name__ == "__main__":
    main()
