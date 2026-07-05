"""
Utils and wrappers for scoring parsers.
"""

from collections import Counter, defaultdict, namedtuple
import logging

from stanza.models.common.utils import ud_scores
from stanza.models.depparse.head_constraints import SINGLETON_DEPREL_GROUPS, DEPREL_TO_GROUP

logger = logging.getLogger('stanza')

def score_named_dependencies(pred_doc, gold_doc, output_latex=False):
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
    #log_line_fmt = "%" + str(max_len) + "s: p %.4f r %.4f f1 %.4f (%d actual)"
    if output_latex:
        log_lines.append(r"\begin{tabular}{lrr}")
        log_lines.append(r"Reln & F1 & Total \\")
        log_line_fmt = "{label} & {f1:0.4f} & {actual} \\\\"
    else:
        log_line_fmt = "{label:>" + str(max_len) + "s}: p {precision:0.4f} r {recall:0.4f} f1 {f1:0.4f} ({actual} actual)"
    for label in labels:
        if tp[label] == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = tp[label] / (tp[label] + fp[label])
            recall = tp[label] / (tp[label] + fn[label])
            f1 = 2 * (precision * recall) / (precision + recall)
        actual = tp[label] + fn[label]
        template = {
            'label': label,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'actual': actual
        }
        log_lines.append(log_line_fmt.format(**template))
    if output_latex:
        log_lines.append(r"\end{tabular}")
    logger.info("F1 scores for each dependency:\n  Note that unlabeled attachment errors hurt the labeled attachment scores\n%s" % "\n".join(log_lines))

def _deprel_group_key(deprel):
    """ Maps a deprel to the SINGLETON_DEPREL_GROUPS key it belongs to, or None
    if it isn't covered by any singleton constraint. """
    return DEPREL_TO_GROUP.get(deprel)

ViolationExample = namedtuple("ViolationExample", ["label", "sent_idx", "head_text", "group_name", "dep_desc"])

def count_head_constraint_violations(pred_doc, collect_examples=True, label=None):
    """
    Pure counting pass over pred_doc: no logging, so this is safe (and cheap)
    to call once per chunk when a large corpus has to be parsed and processed
    in pieces rather than as a single Document.

    Counts, for each SINGLETON_DEPREL_GROUPS entry, how many heads in pred_doc
    have more than one dependent whose deprel maps to that group (see
    _deprel_group_key for the exact mapping, including the nsubj:outer/
    csubj:outer exception).

    For example, a verb with both an nsubj and a csubj child is one violation
    of the "subj" group, as is a verb with two nsubj children. A verb with one
    nsubj:outer and one ordinary nsubj is NOT a violation (different groups).

    Also tracks two extra things useful for deciding how to handle these:
      - valency_counts[group_name]: a Counter of how many dependents were
        actually involved in each violation (2, 3, ...), so eg "2 obj" and
        "3 obj" violations aren't conflated -- helps answer "is it always
        just doubled up, or does it go higher?"
      - violations_per_sentence: a Counter of how many separate (head, group)
        violations occurred together in the same sentence, across all
        sentences (sentences with 0 violations are included too) -- helps
        answer "when this happens, is it usually an isolated problem in the
        sentence, or do several crop up at once?"

    label is an optional identifier (eg a filename or chunk number) attached
    to any collected examples, so that when results from many chunks get
    merged together, debug output can still say which chunk an example came
    from. If collect_examples is False, examples are skipped entirely --
    worth doing for a very large corpus run where you only care about the
    aggregate counts, since the violation rate is typically low (~1%) but
    the raw text of every example can still add up over a huge corpus.

    Returns a dict with keys:
      'counts': Counter(group_name -> total violating heads)
      'valency_counts': dict(group_name -> Counter(valency -> number of heads))
      'violations_per_sentence': Counter(num_violations_in_sentence -> num_sentences)
      'total_sentences': int
      'examples': list of ViolationExample namedtuples (label, sent_idx, head_text, group_name, dep_desc)
    """
    violation_counts = Counter()
    valency_counts = defaultdict(Counter)
    violations_per_sentence = Counter()
    examples = []
    for sent_idx, sentence in enumerate(pred_doc.sentences):
        # words indexed by id so we can report the head's text as well
        words_by_id = {int(word.id): word for word in sentence.words}
        # head_id -> group_name -> list of dependent words in that group
        deps_per_head = defaultdict(lambda: defaultdict(list))
        for word in sentence.words:
            group_name = _deprel_group_key(word.deprel)
            if group_name is not None:
                deps_per_head[word.head][group_name].append(word)
        sentence_violation_count = 0
        for head_id, groups in deps_per_head.items():
            for group_name, deps in groups.items():
                if len(deps) > 1:
                    violation_counts[group_name] += 1
                    valency_counts[group_name][len(deps)] += 1
                    sentence_violation_count += 1
                    if collect_examples:
                        head_text = words_by_id[head_id].text if head_id in words_by_id else "ROOT"
                        dep_desc = ", ".join("{}/{}".format(w.text, w.deprel) for w in deps)
                        examples.append(ViolationExample(label, sent_idx, head_text, group_name, dep_desc))
        violations_per_sentence[sentence_violation_count] += 1

    return {
        'counts': violation_counts,
        'valency_counts': dict(valency_counts),
        'violations_per_sentence': violations_per_sentence,
        'total_sentences': len(pred_doc.sentences),
        'examples': examples,
    }

def merge_violation_results(results):
    """
    Combines multiple count_head_constraint_violations() outputs (eg one per
    chunk of a large corpus that had to be parsed in pieces) into a single
    aggregate dict with the same shape, suitable for passing to
    log_head_constraint_violations().
    """
    merged_counts = Counter()
    merged_valency = defaultdict(Counter)
    merged_per_sentence = Counter()
    merged_examples = []
    total_sentences = 0
    for result in results:
        merged_counts.update(result['counts'])
        for group_name, valency_counter in result['valency_counts'].items():
            merged_valency[group_name].update(valency_counter)
        merged_per_sentence.update(result['violations_per_sentence'])
        merged_examples.extend(result['examples'])
        total_sentences += result['total_sentences']
    return {
        'counts': merged_counts,
        'valency_counts': dict(merged_valency),
        'violations_per_sentence': merged_per_sentence,
        'total_sentences': total_sentences,
        'examples': merged_examples,
    }

def log_head_constraint_violations(result, log_examples=True):
    """
    Logging-only step: takes the dict returned by
    count_head_constraint_violations() (or merge_violation_results(), for
    a whole corpus processed in chunks) and logs a summary.
    """
    counts = result['counts']
    valency_counts = result['valency_counts']
    violations_per_sentence = result['violations_per_sentence']
    total_sentences = result['total_sentences']
    examples = result.get('examples', [])

    if log_examples:
        for example in examples:
            if example.label is None:
                logger.debug("Sentence %d: head '%s' has multiple '%s' dependents: %s",
                             example.sent_idx, example.head_text, example.group_name, example.dep_desc)
            else:
                logger.debug("%s sentence %d: head '%s' has multiple '%s' dependents: %s",
                             example.label, example.sent_idx, example.head_text, example.group_name, example.dep_desc)

    for group_name, count in counts.items():
        valency_desc = ", ".join("{}x{}: {}".format(valency, group_name, num)
                                  for valency, num in sorted(valency_counts[group_name].items()))
        logger.info("Head constraint violations for group '%s': %d (out of %d sentences) [%s]",
                    group_name, count, total_sentences, valency_desc)
    if not counts:
        logger.info("No head constraint violations found (groups checked: %s)",
                    ", ".join(SINGLETON_DEPREL_GROUPS.keys()))

    sentence_dist_desc = ", ".join("{} violations: {} sentences".format(n, num)
                                     for n, num in sorted(violations_per_sentence.items()))
    logger.info("Violations per sentence: %s", sentence_dist_desc)

def check_head_constraint_violations(pred_doc, log_examples=True):
    """
    Convenience wrapper: count + log in one call, for the common
    single-document case (eg the normal train/dev/test eval loop).
    For a corpus too large to hold as one Document, call
    count_head_constraint_violations() per chunk, merge_violation_results()
    to combine them, and log_head_constraint_violations() once at the end
    instead.
    """
    result = count_head_constraint_violations(pred_doc, collect_examples=log_examples)
    log_head_constraint_violations(result, log_examples=log_examples)
    return result

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

