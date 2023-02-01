"""
Utils and wrappers for scoring MWT
"""
from stanza.models.common.utils import ud_scores

def score(system_conllu_file, gold_conllu_file):
    """ Wrapper for word segmenter scorer. """
    evaluation = ud_scores(gold_conllu_file, system_conllu_file)
    el = evaluation["Words"]
    p, r, f = el.precision, el.recall, el.f1
    return p, r, f

