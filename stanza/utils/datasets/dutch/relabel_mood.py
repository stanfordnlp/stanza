"""
Relabel Mood on finite verbs in the UD Dutch treebanks (Alpino, LassySmall).

The UD conversion of these treebanks (alud) under-assigns Mood=Imp and never
assigns Mood=Sub:

  - Mood=Imp comes from an Alpino parser attribute (stype="imparative") that
    is often missing, notably for coordinated imperatives and several of the
    grammar test suites
  - Mood=Sub is overwritten with Ind/Imp in alud, even though the gold CGN tag
    WW(pv,conj,...) marks the subjunctive

This script only changes the Mood feature.  Everything else in the file is
written back byte for byte.

Usage:

  python3 -m stanza.utils.datasets.dutch.relabel_mood in.conllu out.conllu [--changes changes.tsv]

or from other code, relabel_file(input_path, output_path) or relabel_lines(lines)

Rules:

  Sub: any token with XPOS WW|pv|conj|... gets Mood=Sub

  Imp: a finite present-tense verb gets Mood=Imp when
    - its XPOS is WW|pv|tgw|ev (the bare stem, as in "loop"), or
      WW|pv|tgw|met-t with a postverbal "u" subject ("Komt u binnen")
    - it heads (or is the aux/cop of) a root, conj, or parataxis clause
    - nothing precedes it in the clause except punctuation, a coordinating
      conjunction, a discourse marker, a vocative, or a subordinate clause
      ("Als je dat wilt, zeur dan niet"), or else whatever precedes it is
      set off by a comma ("In Outlook, ga naar ...")
    - the clause has no subject, or only a postverbal jij/u/jullie/gij/ge
      which does not follow a fronted constituent
    - the clause does not end in a question mark
    - a subjectless conj is not sharing the subject of the clause it is
      conjoined to ("Hij komt binnen en zet koffie")
    - the lemma is not one which has no imperative but is often subject
      dropped: zijn, the modals, hoeven, weten
  plus small lists of reviewed exceptions (EXCLUDE, FORCE)

Existing Mood=Imp tokens are kept, except for the reviewed false positives
in DEMOTE, which become Mood=Ind.

Checks:

  The rules depend on the conventions of the current UD release (XPOS format,
  deprel names, how auxiliaries attach).  After processing, the script
  compares the file against statistics measured on the releases this script
  was written for (converted with ALUD 2.14) and prints a WARNING when the file no longer looks like that, for example if
  the XPOS format changed or subjects stopped being labelled nsubj.  Any
  WARNING makes the script exit with status 1.  The output file is still
  written, but should not be trusted until the warning is understood.

  It also prints a NOTE when upstream appears to have fixed one of the
  problems this script works around, so that the corresponding rule or
  exception can be retired.  NOTEs do not change the exit status.  These
  checks are skipped when there are WARNINGs, since broken rules also add
  fewer tags.
"""

import argparse
import os
import re
import sys

SUBJECTS = {'nsubj', 'nsubj:pass', 'csubj', 'csubj:pass', 'expl', 'expl:pv'}
ADDRESS_PRONOUNS = {'jij', 'u', 'jullie', 'gij', 'ge'}
ALLOWED_BEFORE = {'punct', 'cc', 'discourse', 'vocative', 'advcl'}
# relations which do not count as a fronted constituent for V2 word order
UNFRONTED = {'punct', 'cc', 'discourse', 'vocative'}
CLAUSE_RELATIONS = {'root', 'conj', 'parataxis'}
NO_IMPERATIVE = {'zijn', 'kunnen', 'mogen', 'willen', 'zullen', 'moeten', 'hoeven', 'weten'}

# The exception lists are keyed by (sent_id, token id, form).  An entry is
# only applied if the form matches, so a retokenized sentence in a later
# release cannot redirect it to a different word.  Entries which do not
# match, or never fire, are reported at the end of the run.

# rule hits which were reviewed and are not imperatives
EXCLUDE = set()

# reviewed imperatives the rule does not reach
FORCE = set()

# existing Mood=Imp tokens which are not imperatives
DEMOTE = {
    ('cdb\\2994', 4, 'praat'),    # Op het ogenblik praat Berry Visser ... (overt subject)
}

# Thresholds for the convention checks.  The comments give the values
# measured on the Alpino and LassySmall releases this script was written for.
FINITE_XPOS = re.compile(r'^WW\|pv\|(tgw|verl|conj)\|(ev|mv|met-t)$')
MIN_FINITE = 500                 # files with fewer finite verbs are too small to check
MIN_XPOS_RATE = 0.99             # 1.000 in all files
MIN_MOOD_RATE = 0.99             # 1.000 in all files
MIN_SUBJECT_RATE = 0.75          # 0.88 - 0.95
MAX_ADDED_RATE = 0.015           # 0.000 - 0.003
MIN_UPSTREAM_IMP = 20            # files with fewer upstream Mood=Imp skip the recall check
MIN_RULE_RECALL = 0.50           # 0.82 - 0.85 in the train files
CORE_DEPRELS = ('root', 'nsubj', 'conj', 'parataxis', 'aux', 'cop', 'advcl', 'punct', 'cc', 'obj')

# Thresholds for noticing that upstream fixed things
MAX_FIXED_ADDED_SHARE = 0.05     # added Imp / all Imp: 0.33 Alpino train, 0.11 LassySmall train


class Report:
    """
    The result of relabeling one document

    changes is a list of (sent_id, token id, form, old mood, new mood, text)
    warnings and notes are lists of strings, as described in the module docstring
    """
    def __init__(self):
        self.changes = []
        self.warnings = []
        self.notes = []
        self.used = set()
        self.counts = {key: 0 for key in ('finite', 'finite_xpos', 'finite_mood', 'clause', 'clause_subject',
                                          'upstream_imp', 'upstream_imp_rule', 'added_imp',
                                          'subjunctive', 'subjunctive_upstream')}
        self.deprels = set()

    def warn(self, message):
        self.warnings.append(message)

    def note(self, message):
        self.notes.append(message)


def lookup(exceptions, sent_id, word, report):
    """True if (sent_id, word) is listed in exceptions with a matching form"""
    for key in exceptions:
        if key[0] == sent_id and key[1] == word.id:
            if key[2] != word.form:
                report.warn('exception %s does not match form %s' % (key, word.form))
                return None
            report.used.add(key)
            return key
    return None


def read_sentences(lines):
    sentence = []
    for line in lines:
        sentence.append(line)
        if line.strip() == '':
            yield sentence
            sentence = []
    if sentence:
        yield sentence


class Word:
    def __init__(self, line_index, columns):
        self.line_index = line_index
        self.id = int(columns[0])
        self.form = columns[1]
        self.lemma = columns[2]
        self.upos = columns[3]
        self.xpos = columns[4]
        self.feats = columns[5]
        self.head = int(columns[6])
        self.deprel = columns[7]

    def feature(self, key):
        for kv in self.feats.split('|'):
            if kv.startswith(key + '='):
                return kv[len(key) + 1:]
        return None


def set_mood(feats, mood):
    pieces = [kv for kv in feats.split('|') if kv != '_' and not kv.startswith('Mood=')]
    pieces.append('Mood=' + mood)
    pieces.sort(key=lambda kv: kv.split('=')[0].lower())
    return '|'.join(pieces)


def is_question(words, verb):
    """True if the first sentence-level punctuation after the verb is a ?"""
    for word in words:
        if word.id > verb.id and word.upos == 'PUNCT' and word.form in ('.', '!', '?', ';', ':'):
            return word.form == '?'
    return False


def clause_of(by_id, children, verb):
    """The head of the verb's clause, and the other members of the clause"""
    head = verb
    if verb.deprel in ('aux', 'aux:pass', 'cop'):
        head = by_id[verb.head]
    clause = children[head.id] + (children[verb.id] if head is not verb else [])
    clause = [w for w in clause if w is not verb]
    if head is not verb:
        clause.append(head)
    return head, clause


def is_subordinate_clause(children, word):
    """
    True for a fronted subordinate clause attached as parataxis

    Alpino analyzes "Als X, dan Y" as two discourse parts, which the UD
    conversion attaches as parataxis rather than advcl
    """
    return word.deprel.startswith('parataxis') and any(w.deprel == 'mark' for w in children[word.id])


def imperative_candidate(words, by_id, children, verb):
    xpos = verb.xpos.split('|')
    if verb.feature('VerbForm') != 'Fin' or xpos[:3] != ['WW', 'pv', 'tgw']:
        return False
    # compound lemmas such as om_zijn, dat_zijn
    if verb.lemma.split('_')[-1] in NO_IMPERATIVE:
        return False

    head, clause = clause_of(by_id, children, verb)
    if head.deprel.split(':')[0] not in CLAUSE_RELATIONS:
        return False
    before = [w for w in clause if w.id < verb.id]
    # anything may precede the verb if it is set off by a comma: "In Outlook, ga naar ..."
    comma_before = verb.id - 1 in by_id and by_id[verb.id - 1].form == ','
    if not comma_before and any(w.deprel.split(':')[0] not in ALLOWED_BEFORE and
                                not is_subordinate_clause(children, w) for w in before):
        return False
    fronted = any(w.deprel.split(':')[0] not in UNFRONTED for w in before)

    subjects = [w for w in clause if w.deprel in SUBJECTS]
    if any(w.form.lower() not in ADDRESS_PRONOUNS or w.id < verb.id for w in subjects):
        return False
    # after a fronted constituent, a postverbal subject is ordinary V2 word order:
    # "Als ik jou er een stuur, neem jij het van mij over"
    if subjects and fronted:
        return False
    if xpos[3] == 'met-t' and not any(w.form.lower() == 'u' for w in subjects):
        return False
    if xpos[3] not in ('ev', 'met-t'):
        return False
    if is_question(words, verb):
        return False

    if head.deprel.startswith('conj') and not subjects:
        first = by_id[head.head]
        if any(w.deprel in SUBJECTS for w in children[first.id]):
            return False
    return True


def count_finite(report, by_id, children, word):
    counts = report.counts
    counts['finite'] += 1
    counts['finite_xpos'] += bool(FINITE_XPOS.match(word.xpos))
    counts['finite_mood'] += word.feature('Mood') is not None
    head, clause = clause_of(by_id, children, word)
    if head.deprel.split(':')[0] in CLAUSE_RELATIONS:
        counts['clause'] += 1
        counts['clause_subject'] += any(w.deprel in SUBJECTS for w in clause)


def process_sentence(lines, report):
    sent_id = None
    words = []
    for index, line in enumerate(lines):
        if line.startswith('# sent_id = '):
            sent_id = line[len('# sent_id = '):].strip()
        if not line or line.startswith('#') or line.strip() == '':
            continue
        columns = line.rstrip('\n').split('\t')
        if '-' in columns[0] or '.' in columns[0]:
            continue
        words.append(Word(index, columns))

    by_id = {w.id: w for w in words}
    children = {w.id: [] for w in words}
    children[0] = []
    for w in words:
        children[w.head].append(w)
        report.deprels.add(w.deprel.split(':')[0])

    counts = report.counts
    text = next((l[len('# text = '):].strip() for l in lines if l.startswith('# text = ')), '')
    for word in words:
        mood = word.feature('Mood')
        finite = word.feature('VerbForm') == 'Fin'
        if finite:
            count_finite(report, by_id, children, word)
        candidate = finite and imperative_candidate(words, by_id, children, word)

        new_mood = None
        if word.xpos.startswith('WW|pv|conj'):
            counts['subjunctive'] += 1
            counts['subjunctive_upstream'] += mood == 'Sub'
            new_mood = 'Sub'
        elif lookup(DEMOTE, sent_id, word, report):
            if mood != 'Imp':
                report.note('DEMOTE entry %s is no longer Mood=Imp upstream (Mood=%s) and can be removed' %
                            ((sent_id, word.id, word.form), mood or 'unset'))
            new_mood = 'Ind'
        elif mood == 'Imp':
            counts['upstream_imp'] += 1
            counts['upstream_imp_rule'] += candidate
            if lookup(FORCE, sent_id, word, report):
                report.note('FORCE entry %s is already Mood=Imp upstream and can be removed' %
                            ((sent_id, word.id, word.form),))
            if lookup(EXCLUDE, sent_id, word, report):
                report.note('EXCLUDE entry %s is Mood=Imp upstream; upstream disagrees with the review' %
                            ((sent_id, word.id, word.form),))
            continue
        elif lookup(FORCE, sent_id, word, report):
            new_mood = 'Imp'
        elif candidate and not lookup(EXCLUDE, sent_id, word, report):
            new_mood = 'Imp'

        if new_mood is None or new_mood == mood:
            continue
        if new_mood == 'Imp':
            counts['added_imp'] += 1
        columns = lines[word.line_index].rstrip('\n').split('\t')
        columns[5] = set_mood(columns[5], new_mood)
        lines[word.line_index] = '\t'.join(columns) + '\n'
        report.changes.append((sent_id, word.id, word.form, mood, new_mood, text))
    return lines


def check_conventions(report):
    """Warn if the file no longer looks like the data the rules were written for"""
    counts = report.counts
    if counts['finite'] < MIN_FINITE:
        report.note('only %d finite verbs, too few for the convention checks' % counts['finite'])
        return

    def rate(numerator, denominator):
        return counts[numerator] / counts[denominator] if counts[denominator] else 0.0

    missing = [d for d in CORE_DEPRELS if d not in report.deprels]
    if missing:
        report.warn('expected deprels missing entirely: %s.  The deprel conventions have changed; '
                    'the imperative rule may add wrong tags or none at all' % ' '.join(missing))
    if rate('finite_xpos', 'finite') < MIN_XPOS_RATE:
        report.warn('only %.1f%% of finite verbs have XPOS of the form WW|pv|tense|agreement (expected ~100%%).  '
                    'The XPOS format has changed; both rules key on it' % (100 * rate('finite_xpos', 'finite')))
    if rate('finite_mood', 'finite') < MIN_MOOD_RATE:
        report.warn('only %.1f%% of finite verbs have a Mood feature (expected ~100%%).  '
                    'The feature conventions have changed' % (100 * rate('finite_mood', 'finite')))
    if counts['clause'] and rate('clause_subject', 'clause') < MIN_SUBJECT_RATE:
        report.warn('only %.1f%% of finite main clauses have a subject (expected 88-95%%).  Subjects may be '
                    'labelled differently, in which case declaratives get tagged as imperatives' %
                    (100 * rate('clause_subject', 'clause')))
    if rate('added_imp', 'finite') > MAX_ADDED_RATE:
        report.warn('added Mood=Imp to %.1f%% of finite verbs (expected at most ~0.3%%).  '
                    'The imperative rule is probably misfiring' % (100 * rate('added_imp', 'finite')))
    if counts['upstream_imp'] >= MIN_UPSTREAM_IMP and rate('upstream_imp_rule', 'upstream_imp') < MIN_RULE_RECALL:
        report.warn('the imperative rule recognizes only %.1f%% of the upstream Mood=Imp tokens '
                    '(expected ~80%%).  The tree conventions have changed and the rule is probably '
                    'missing imperatives' % (100 * rate('upstream_imp_rule', 'upstream_imp')))


def check_upstream_fixes(report):
    """Note when upstream has fixed what this script works around"""
    counts = report.counts
    if counts['subjunctive'] and counts['subjunctive_upstream'] == counts['subjunctive']:
        report.note('all %d WW|pv|conj tokens are already Mood=Sub upstream; the Sub rule is no longer needed' %
                    counts['subjunctive'])
    elif counts['subjunctive_upstream']:
        report.note('%d of %d WW|pv|conj tokens are already Mood=Sub upstream; upstream has started fixing Sub' %
                    (counts['subjunctive_upstream'], counts['subjunctive']))
    total_imp = counts['upstream_imp'] + counts['added_imp']
    if counts['upstream_imp'] >= MIN_UPSTREAM_IMP and counts['added_imp'] / total_imp < MAX_FIXED_ADDED_SHARE:
        report.note('the rule added only %d of %d imperatives; upstream imperative tagging may have been fixed, '
                    'and the Imp rule may no longer be worth running' % (counts['added_imp'], total_imp))


def relabel_lines(lines, check=True):
    """
    Relabel Mood in the lines of a CoNLL-U document.

    Returns the relabeled lines and a Report holding the changes, warnings
    and notes.  The input list is not modified.

    check=False skips the convention and upstream checks, which are
    calibrated on the UD treebanks.  This is for data which is known to look
    different, such as a corpus which is mostly imperatives.  Warnings about
    the exception lists are still reported.
    """
    report = Report()
    output = []
    sent_ids = set()
    for sentence in read_sentences(list(lines)):
        sent_ids.update(l[len('# sent_id = '):].strip() for l in sentence if l.startswith('# sent_id = '))
        output.extend(process_sentence(sentence, report))

    # each file only contains some of the listed sentences, so only report
    # entries whose sentence is in this file but which did not fire
    for key in sorted((EXCLUDE | FORCE | DEMOTE) - report.used):
        if key[0] in sent_ids:
            report.warn('exception %s did not apply' % (key,))
    if check:
        check_conventions(report)
        # if the conventions have drifted, a low count of additions means the
        # rules are broken, not that upstream fixed anything
        if not report.warnings:
            check_upstream_fixes(report)
    return output, report


def relabel_file(input_path, output_path, changes_path=None):
    """
    Relabel Mood in input_path and write the result to output_path.

    If changes_path is given, a TSV line for each change is appended to it.
    Returns the Report.  Nothing is printed and the caller decides what to do
    about warnings; see print_report for the command line behavior.
    """
    with open(input_path, encoding='utf-8', newline='') as fin:
        lines = fin.readlines()
    output, report = relabel_lines(lines)
    with open(output_path, 'w', encoding='utf-8', newline='') as fout:
        fout.writelines(output)
    if changes_path:
        filename = os.path.basename(input_path)
        with open(changes_path, 'a', encoding='utf-8', newline='') as fout:
            for change in report.changes:
                fout.write('\t'.join([filename] + [str(x) for x in change]) + '\n')
    return report


def print_report(report, input_path, output_path, stream=None):
    """Print a summary of the report to stream, sys.stderr by default"""
    if stream is None:
        stream = sys.stderr
    counts = {}
    for change in report.changes:
        counts[change[4]] = counts.get(change[4], 0) + 1
    summary = ', '.join('%d -> %s' % (v, k) for k, v in sorted(counts.items())) or 'no changes'
    print('%s: %s' % (input_path, summary), file=stream)
    for note in report.notes:
        print('  NOTE: %s' % note, file=stream)
    for warning in report.warnings:
        print('  WARNING: %s' % warning, file=stream)
    if report.warnings:
        print('  %d WARNING(S) for %s: check them before using %s' % (len(report.warnings), input_path, output_path),
              file=stream)


def main(args=None):
    """
    Command line entry point.  args defaults to sys.argv[1:]

    Exits with status 1 if there were any warnings.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='input CoNLL-U file')
    parser.add_argument('output', help='output CoNLL-U file')
    parser.add_argument('--changes', default=None, help='append a TSV of every change to this file')
    args = parser.parse_args(args)

    report = relabel_file(args.input, args.output, args.changes)
    print_report(report, args.input, args.output)
    if report.warnings:
        sys.exit(1)


if __name__ == '__main__':
    main()
