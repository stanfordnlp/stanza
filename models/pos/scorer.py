"""
Utils and wrappers for scoring lemmatizers.
"""
from models.common.utils import ud_scores

def score(system_conllu_file, gold_conllu_file):
    """ Wrapper for word segmenter scorer. """
    evaluation = ud_scores(gold_conllu_file, system_conllu_file)
    el = evaluation['AllTags']
    p = el.precision
    r = el.recall
    f = el.f1
    print(evaluation['UPOS'].f1 * 100, evaluation['XPOS'].f1 * 100, evaluation['UFeats'].f1 * 100, evaluation['AllTags'].f1 * 100)
    return p, r, f

