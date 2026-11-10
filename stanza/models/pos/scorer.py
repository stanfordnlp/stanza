"""
Utils and wrappers for scoring taggers.
"""
from collections import Counter, defaultdict
import logging

from stanza.models.common.utils import ud_scores
from stanza.utils.confusion import format_confusion, format_label_f1

logger = logging.getLogger('stanza')

BLANK = "_"

def blank_value(value):
    """ Map a missing tag (None or "_") to "_" so it can be counted like any other label """
    if value is None or value == BLANK:
        return BLANK
    return value

def parse_feats(feats):
    """
    Turn a UFeats string such as "Mood=Imp|VerbForm=Fin" into a dict of name -> value

    None or "_" gives an empty dict.  A value such as "PronType=Int,Rel"
    is kept as the single value "Int,Rel".  A malformed piece with no
    "=" is kept as a feature with the value "_".
    """
    if feats is None or feats == BLANK:
        return {}
    result = {}
    for piece in feats.split("|"):
        if not piece:
            continue
        if "=" in piece:
            name, value = piece.split("=", 1)
        else:
            name, value = piece, BLANK
        result[name] = value
    return result

def check_doc_alignment(pred_doc, gold_doc):
    """
    Return True if the two docs have the same sentences and words, so the tags can be compared word by word

    The tagger is scored on gold tokenization, so in normal use this
    always holds.  Logs a warning and returns False otherwise.
    """
    if len(pred_doc.sentences) != len(gold_doc.sentences):
        logger.warning("Not evaluating individual tag F1 on account of document length mismatch")
        return False
    for pred_sentence, gold_sentence in zip(pred_doc.sentences, gold_doc.sentences):
        if len(pred_sentence.words) != len(gold_sentence.words):
            logger.warning("Not evaluating individual tag F1 on account of sentence length mismatch")
            return False
    return True

def count_tag_f1(pred_doc, gold_doc, field):
    """
    Count true positives, false positives, and false negatives for each value of one tag column

    field is the name of the Word attribute, such as "upos" or "xpos".
    A missing tag is counted as the label "_".

    Returns three Counters: tp, fp, fn
    """
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for pred_sentence, gold_sentence in zip(pred_doc.sentences, gold_doc.sentences):
        for pred_word, gold_word in zip(pred_sentence.words, gold_sentence.words):
            gold_tag = blank_value(getattr(gold_word, field))
            pred_tag = blank_value(getattr(pred_word, field))
            if gold_tag == pred_tag:
                tp[gold_tag] += 1
            else:
                fn[gold_tag] += 1
                fp[pred_tag] += 1
    return tp, fp, fn

def count_feature_scores(pred_doc, gold_doc):
    """
    Count per-feature scores, treating each feature independently of the others on the same word

    For example, a word with gold "Mood=Imp|VerbForm=Fin" and predicted
    "Mood=Ind|VerbForm=Fin" is a true positive for VerbForm=Fin, a false
    negative for Mood=Imp, and a false positive for Mood=Ind.

    Returns tp, fp, fn, confusion
      tp, fp, fn are Counters keyed by "Name=Value"
      confusion is name -> gold value -> predicted value -> count,
        over every word in the doc, with "_" for a word which does not
        have that feature.  So the "_","_" cell of each feature counts
        the words where neither gold nor prediction has the feature.
    """
    tp = Counter()
    fp = Counter()
    fn = Counter()

    word_pairs = []
    feature_names = set()
    for pred_sentence, gold_sentence in zip(pred_doc.sentences, gold_doc.sentences):
        for pred_word, gold_word in zip(pred_sentence.words, gold_sentence.words):
            gold_feats = parse_feats(gold_word.feats)
            pred_feats = parse_feats(pred_word.feats)
            word_pairs.append((gold_feats, pred_feats))
            feature_names.update(gold_feats.keys())
            feature_names.update(pred_feats.keys())

    confusion = {name: defaultdict(lambda: defaultdict(int)) for name in feature_names}
    for gold_feats, pred_feats in word_pairs:
        for name in feature_names:
            gold_value = gold_feats.get(name, BLANK)
            pred_value = pred_feats.get(name, BLANK)
            confusion[name][gold_value][pred_value] += 1
            if gold_value == pred_value:
                if gold_value != BLANK:
                    tp["%s=%s" % (name, gold_value)] += 1
            else:
                if gold_value != BLANK:
                    fn["%s=%s" % (name, gold_value)] += 1
                if pred_value != BLANK:
                    fp["%s=%s" % (name, pred_value)] += 1
    return tp, fp, fn, confusion

def has_gold_tags(gold_doc, field):
    """ Return True if any word in gold_doc has a non-blank value for field """
    return any(blank_value(getattr(word, field)) != BLANK
               for sentence in gold_doc.sentences
               for word in sentence.words)

def format_feature_confusion(confusion):
    """ Format one feature's confusion matrix with "_" (no value) as the first row and column """
    values = set(confusion.keys())
    for row in confusion.values():
        values.update(row.keys())
    values.discard(BLANK)
    labels = [BLANK] + sorted(values)
    return format_confusion(confusion, labels=labels, hide_zeroes=True)

def score_named_tags(pred_doc, gold_doc, feature_confusion=None):
    """
    Log the F1 of each UPOS, each XPOS, and each feature value, along with confusion matrices for the features

    A column is skipped if the gold doc has no values in it, such as
    a treebank with no XPOS.

    feature_confusion selects which features get a confusion matrix:
      None: every feature in the gold or predicted doc
      a list of feature names: only those (eg ["Mood"])
      an empty list: no confusion matrices
    """
    if not check_doc_alignment(pred_doc, gold_doc):
        return

    for field, name in (("upos", "UPOS"), ("xpos", "XPOS")):
        if not has_gold_tags(gold_doc, field):
            continue
        tp, fp, fn = count_tag_f1(pred_doc, gold_doc, field)
        log_lines = format_label_f1(tp, fp, fn)
        logger.info("F1 scores for each %s:\n%s", name, "\n".join(log_lines))

    if not has_gold_tags(gold_doc, "feats"):
        return

    tp, fp, fn, confusion = count_feature_scores(pred_doc, gold_doc)
    log_lines = format_label_f1(tp, fp, fn)
    if log_lines:
        logger.info("F1 scores for each feature value:\n  Each feature is scored on its own, regardless of the other features on the word\n%s", "\n".join(log_lines))

    if feature_confusion is None:
        names = sorted(confusion.keys())
    else:
        names = feature_confusion
    for name in names:
        if name not in confusion:
            logger.info("Feature %s does not occur in the gold or predicted data", name)
            continue
        logger.info("Confusion matrix for feature %s\n  Rows are gold, columns are predicted, and _ means the word does not have the feature\n%s",
                    name, format_feature_confusion(confusion[name]))

def score(system_conllu_file, gold_conllu_file, verbose=True, eval_type='AllTags'):
    """ Wrapper for tagger scorer. """
    evaluation = ud_scores(gold_conllu_file, system_conllu_file)
    el = evaluation[eval_type]
    p = el.precision
    r = el.recall
    f = el.f1
    if verbose:
        scores = [evaluation[k].f1 * 100 for k in ['UPOS', 'XPOS', 'UFeats', 'AllTags']]
        logger.info("UPOS\tXPOS\tUFeats\tAllTags")
        logger.info("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(*scores))
    return p, r, f
