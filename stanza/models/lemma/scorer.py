"""
Utils and wrappers for scoring lemmatizers.
"""

import logging

from stanza.models.common.utils import ud_scores

logger = logging.getLogger('stanza')

def score(system_conllu_file, gold_conllu_file):
    """ Wrapper for lemma scorer. """
    logger.debug("Evaluating system file %s vs gold file %s", system_conllu_file, gold_conllu_file)
    evaluation = ud_scores(gold_conllu_file, system_conllu_file)
    el = evaluation["Lemmas"]
    p, r, f = el.precision, el.recall, el.f1
    return p, r, f

