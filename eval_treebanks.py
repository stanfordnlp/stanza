import sys
import os
from models.common.utils import ud_scores
from utils.conll18_ud_eval import UDError

golddir = sys.argv[1]
preddir = sys.argv[2]
dset = sys.argv[3]

with open('short_to_tb') as f:
    for line in f:
        line = line.strip()
        short, full = line.split(' ')

        try:
            res = ud_scores(os.path.join(golddir, "{}-ud-{}.conllu".format(short, dset)),
                os.path.join(preddir, "{}-ud-{}-pred.conllu".format(short, dset)))
        except (UDError, FileNotFoundError):
            print(full)
            continue

        output = [full]
        for metric in["Tokens", "Sentences", "Words", "UPOS", "XPOS", "UFeats", "AllTags", "Lemmas", "UAS", "LAS", "CLAS", "MLAS", "BLEX"]:
            output.append("{:.2f}".format(100 * res[metric].f1))
        print(" ".join(output))
