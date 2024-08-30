"""
throughput.py
Benchmark Stanza's tokenizer for throughput
"""

import argparse
import cProfile
import stanza
from tqdm import tqdm
from nltk.corpus import brown
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pstats, io
from pstats import SortKey
from pathlib import Path
sortby = SortKey.CUMULATIVE

parser = argparse.ArgumentParser()
parser.add_argument("out_file", type=str)
parser.add_argument("--size", type=int, default=1e3)

if __name__ == "__main__":
    args = parser.parse_args()

    d = TreebankWordDetokenizer()

    try:
        sentences = brown.sents()
    except:
        import nltk
        nltk.download("brown")
        sentences = brown.sents()

    sentences = [d.detokenize(i) for i in sentences][:int(args.size)]
    sentences = sentences*int(args.size // len(sentences))

    nlp = stanza.Pipeline("en", processors="tokenize")

    # start and perform profiling
    pr = cProfile.Profile()

    pr.enable()
    for i in tqdm(sentences):
        nlp(i)
    pr.disable()

    pr.dump_stats(Path(args.out_file).with_suffix(".prof"))

    stream = io.open(str(Path(args.out_file).with_suffix(".txt")), mode='w')
    stats = pstats.Stats(pr, stream=stream).sort_stats(sortby)
    stats.print_stats()
    stream.close()





