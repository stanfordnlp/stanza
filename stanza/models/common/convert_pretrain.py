"""
A utility script to load a word embedding file from a text file and save it as a .pt

Run it as follows:
  python stanza/models/common/convert_pretrain.py <.pt file> <text file> <# vectors>

Note that -1 for # of vectors will keep all the vectors.
You probably want to keep fewer than that for most publicly released
embeddings, though, as they can get quite large.

As a concrete example, you can convert a newly downloaded Faroese WV file as follows:
  python3 stanza/models/common/convert_pretrain.py ~/stanza/saved_models/pos/fo_farpahc.pretrain.pt ~/extern_data/wordvec/fasttext/faroese.txt -1
or save part of an Icelandic WV file:
  python3 stanza/models/common/convert_pretrain.py ~/stanza/saved_models/pos/is_icepahc.pretrain.pt ~/extern_data/wordvec/fasttext/icelandic.cc.is.300.vec 150000
Note that if the pretrain already exists, nothing will be changed.  It will not overwrite an existing .pt file.

"""

import argparse
import os
import sys

from stanza.models.common import pretrain

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_pt", default=None, help="Where to write the converted PT file")
    parser.add_argument("input_vec", default=None, help="Unconverted vectors file")
    parser.add_argument("max_vocab", type=int, default=-1, nargs="?", help="How many vectors to convert.  -1 means convert them all")
    args = parser.parse_args()

    if os.path.exists(args.output_pt):
        print("Not overwriting existing pretrain file in %s" % args.output_pt)

    if args.input_vec.endswith(".csv"):
        pt = pretrain.Pretrain(args.output_pt, max_vocab=args.max_vocab, csv_filename=args.input_vec)
    else:
        pt = pretrain.Pretrain(args.output_pt, args.input_vec, max_vocab=args.max_vocab)
    print("Pretrain is of size {}".format(len(pt.vocab)))

if __name__ == '__main__':
    main()
