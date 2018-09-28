"""
Utils and wrappers for scoring lemmatizers.
"""
from models.common.utils import ud_scores

def score(system_conllu_file, gold_conllu_file):
    """ Wrapper for word segmenter scorer. """
    evaluation = ud_scores(gold_conllu_file, system_conllu_file)
    el = evaluation['LAS']
    p = el.precision
    r = el.recall
    f = el.f1
    print(evaluation['LAS'].f1 * 100, evaluation['MLAS'].f1 * 100, evaluation['BLEX'].f1 * 100)
    return p, r, f

