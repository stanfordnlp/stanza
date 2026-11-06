"""
A utility script to load a word embedding file and save it as a .pt

Run it as follows:
  python3 -m stanza.utils.pretrain.convert_pretrain <.pt file> <vector file> <# vectors>

The vector file can be a text file (.vec, .txt), a word2vec binary
file, or a fastText .bin model, optionally compressed with xz, gz, or
zip.  The format is detected from the contents of the file, except
for CSV files, which are recognized by a .csv extension.  The
readers are in stanza/utils/pretrain/vector_files.py  For a
fastText .bin model, each word's vector is computed from its subword
n-grams, the same as fastText's own .vec output.

Note that -1 for # of vectors will keep all the vectors.
You probably want to keep fewer than that for most publicly released
embeddings, though, as they can get quite large.  This matters even
more for fastText .bin models, where computing the n-gram vectors for
millions of words takes tens of GB of memory.

As a concrete example, you can convert a newly downloaded Faroese WV file as follows:
  python3 -m stanza.utils.pretrain.convert_pretrain ~/stanza/saved_models/pos/fo_farpahc.pretrain.pt ~/extern_data/wordvec/fasttext/faroese.txt -1
or save part of an Icelandic WV file:
  python3 -m stanza.utils.pretrain.convert_pretrain ~/stanza/saved_models/pos/is_icepahc.pretrain.pt ~/extern_data/wordvec/fasttext/icelandic.cc.is.300.vec 150000
or the most frequent words of a Polish fastText model:
  python3 -m stanza.utils.pretrain.convert_pretrain ~/stanza/saved_models/pos/pl_pdb.pretrain.pt ~/extern_data/wordvec/kgr10/kgr10.plain.skipgram.dim300.neg10.bin 250000
Note that if the pretrain already exists, nothing will be changed.  It will not overwrite an existing .pt file.

"""

import argparse
import os

from stanza.models.common.pretrain import Pretrain, PretrainedWordVocab
from stanza.models.common.vocab import VOCAB_PREFIX
from stanza.utils.pretrain import vector_files

def is_csv_file(filename):
    return any(filename.endswith(suffix) for suffix in (".csv", ".csv.gz", ".csv.xz"))

def convert_pretrain(input_vec, max_vocab=-1):
    """
    Read a word vector file and return it as a Pretrain

    max_vocab counts the VOCAB_PREFIX entries, so the resulting vocab
    has at most max_vocab entries.  -1 keeps every vector in the file.
    """
    if is_csv_file(input_vec):
        words, emb = vector_files.read_csv_vectors(input_vec)
    else:
        words, emb, failed = vector_files.read_vector_file(input_vec, max_vocab)

    if len(emb) - len(VOCAB_PREFIX) != len(words):
        raise RuntimeError("Loaded number of vectors does not match number of words.")

    if max_vocab > len(VOCAB_PREFIX) and max_vocab < len(words) + len(VOCAB_PREFIX):
        words = words[:max_vocab - len(VOCAB_PREFIX)]
        emb = emb[:max_vocab]

    return Pretrain(vocab=PretrainedWordVocab(words), emb=emb)

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("output_pt", default=None, help="Where to write the converted PT file")
    parser.add_argument("input_vec", default=None, help="Unconverted vectors file")
    parser.add_argument("max_vocab", type=int, default=-1, nargs="?", help="How many vectors to convert.  -1 means convert them all")
    args = parser.parse_args(args)

    if os.path.exists(args.output_pt):
        print("Not overwriting existing pretrain file in %s" % args.output_pt)
        return

    pt = convert_pretrain(args.input_vec, args.max_vocab)
    pt.save(args.output_pt)
    print("Pretrain is of size {}".format(len(pt.vocab)))

if __name__ == '__main__':
    main()
