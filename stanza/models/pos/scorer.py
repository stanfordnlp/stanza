"""
Utils and wrappers for scoring taggers.
"""
import logging

from stanza.models.common.utils import ud_scores

logger = logging.getLogger('stanza')

def score(system_conllu_file, gold_conllu_file, verbose=True):
    """ Wrapper for tagger scorer. """
    evaluation = ud_scores(gold_conllu_file, system_conllu_file)
    el = evaluation['AllTags']
    p = el.precision
    r = el.recall
    f = el.f1
    if verbose:
        logger.info("UPOS\tXPOS\tUFeats\tAllTags")

        scores = [evaluation[k].aligned_accuracy * 100 for k in ['UPOS', 'XPOS', 'UFeats', 'AllTags']]
        logger.info("acc {:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(*scores))   


        scores = [evaluation[k].f1 * 100 for k in ['UPOS', 'XPOS', 'UFeats', 'AllTags']]
        logger.info("f1 {:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(*scores))

    return p, r, f, scores[0]

