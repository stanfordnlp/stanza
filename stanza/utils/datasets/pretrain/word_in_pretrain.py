"""
Simple tool to query a word vector file to see if certain words are in that file
"""

import argparse
import os

from stanza.models.common.pretrain import Pretrain
from stanza.resources.common import DEFAULT_MODEL_DIR, download

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pretrain", default=None, type=str, help="Where to read the converted PT file")
    group.add_argument("--package", default=None, type=str, help="Use a pretrain package instead")
    parser.add_argument("--download_json", default=False, action='store_true', help="Download the json even if it already exists")
    parser.add_argument("words", type=str, nargs="+", help="Which words to search for")
    args = parser.parse_args()

    if args.pretrain:
        pt = Pretrain(args.pretrain)
    else:
        lang, package = args.package.split("_", 1)
        download(lang=lang, package=None, processors={"pretrain": package}, download_json=args.download_json)
        pt_filename = os.path.join(DEFAULT_MODEL_DIR, lang, "pretrain", "%s.pt" % package)
        pt = Pretrain(pt_filename)

    for word in args.words:
        print("{}: {}".format(word, word in pt.vocab))

if __name__ == "__main__":
    main()
