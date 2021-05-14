"""
A utility script to load a word embedding file from a text file and save it as a .pt

Run it as follows:
  python stanza/models/common/convert_pretrain.py <.pt file> <text file> <# vectors>
Note that -1 for # of vectors will keep all the vectors
As a concrete example, you can convert a newly downloaded Faroese WV file as follows:
  python3 stanza/models/common/convert_pretrain.py ~/stanza/saved_models/pos/fo_farpahc.pretrain.pt ~/extern_data/wordvec/fasttext/faroese.txt -1
or save part of an Icelandic WV file:
  python3 stanza/models/common/convert_pretrain.py ~/stanza/saved_models/pos/is_icepahc.pretrain.pt ~/extern_data/wordvec/fasttext/icelandic.cc.is.300.vec 150000
Note that if the pretrain already exists, nothing will be changed.  It will not overwrite an existing .pt file.
"""

import os
import sys

from stanza.models.common import pretrain

def main():
    filename = sys.argv[1]
    vec_filename = sys.argv[2]
    if len(sys.argv) < 3:
        max_vocab = -1
    else:
        max_vocab = int(sys.argv[3])

    pt = pretrain.Pretrain(filename, vec_filename, max_vocab)
    print("Pretrain is of size {}".format(len(pt.vocab)))

if __name__ == '__main__':
    main()
