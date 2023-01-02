"""
Utils and wrappers for scoring parsers.
"""

from collections import Counter
import logging

from stanza.models.common.utils import ud_scores

logger = logging.getLogger('stanza')

def score_named_dependencies(pred_doc, gold_doc):
    if len(pred_doc.sentences) != len(gold_doc.sentences):
        logger.warning("Not evaluating individual dependency F1 on accound of document length mismatch")
        return
    for sent_idx, (x, y) in enumerate(zip(pred_doc.sentences, gold_doc.sentences)):
        if len(x.words) != len(y.words):
            logger.warning("Not evaluating individual dependency F1 on accound of sentence length mismatch")
            return

    tp = Counter()
    fp = Counter()
    fn = Counter()
    for pred_sentence, gold_sentence in zip(pred_doc.sentences, gold_doc.sentences):
        for pred_word, gold_word in zip(pred_sentence.words, gold_sentence.words):
            if pred_word.head == gold_word.head and pred_word.deprel == gold_word.deprel:
                tp[gold_word.deprel] = tp[gold_word.deprel] + 1
            else:
                fn[gold_word.deprel] = fn[gold_word.deprel] + 1
                fp[pred_word.deprel] = fp[pred_word.deprel] + 1

    labels = sorted(set(tp.keys()).union(fp.keys()).union(fn.keys()))
    max_len = max(len(x) for x in labels)
    log_lines = []
    log_line_fmt = "%" + str(max_len) + "s: p %.4f r %.4f f1 %.4f (%d actual)"
    for label in labels:
        if tp[label] == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = tp[label] / (tp[label] + fp[label])
            recall = tp[label] / (tp[label] + fn[label])
            f1 = 2 * (precision * recall) / (precision + recall)
        log_lines.append(log_line_fmt % (label, precision, recall, f1, tp[label] + fn[label]))
    logger.info("F1 scores for each dependency:\n  Note that unlabeled attachment errors hurt the labeled attachment scores\n%s" % "\n".join(log_lines))

def score(system_conllu_file, gold_conllu_file, verbose=True):
    """ Wrapper for UD parser scorer. """
    evaluation = ud_scores(gold_conllu_file, system_conllu_file)
    el = evaluation['LAS']
    p = el.precision
    r = el.recall
    f = el.f1
    if verbose:
        scores = [evaluation[k].f1 * 100 for k in ['LAS', 'MLAS', 'BLEX']]
        logger.info("LAS\tMLAS\tBLEX")
        logger.info("{:.2f}\t{:.2f}\t{:.2f}".format(*scores))
    return p, r, f

