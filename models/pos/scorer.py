"""
Utils and wrappers for scoring lemmatizers.
"""
from models.common.utils import ud_scores

def score(system_conllu_file, gold_conllu_file):
    """ Wrapper for word segmenter scorer. """
    evaluation = ud_scores(gold_conllu_file, system_conllu_file)
    p = r = f = 1
    for k in ['UPOS', 'XPOS', 'UFeats']:
        el = evaluation[k]
        p *= el.precision
        r *= el.recall
    f = 0 if p == 0 or r == 0 else 2 * p * r / (p + r)
    return p, r, f

