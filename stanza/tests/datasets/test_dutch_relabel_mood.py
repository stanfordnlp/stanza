"""
Tests for stanza/utils/datasets/dutch/relabel_mood.py

The test sentences are small handmade Dutch sentences, one per rule or edge
case.  Each is built from (form, lemma, upos, xpos, feats, head, deprel)
rows by the sentence() helper, so the trees only have the columns the rules
look at.

The convention checks only run on files with at least MIN_FINITE finite
verbs, so the tests which exercise them lower MIN_FINITE with monkeypatch.
"""

import os
import tempfile

import pytest

from stanza.tests import TEST_WORKING_DIR
from stanza.utils.datasets.dutch import relabel_mood

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FIN_SING = 'Mood=Ind|Number=Sing|Tense=Pres|VerbForm=Fin'
IMP_SING = 'Mood=Imp|Number=Sing|Tense=Pres|VerbForm=Fin'

def sentence(sent_id, text, rows):
    """Build a CoNLL-U sentence from (form, lemma, upos, xpos, feats, head, deprel) rows"""
    lines = ['# sent_id = %s\n' % sent_id, '# text = %s\n' % text]
    for idx, (form, lemma, upos, xpos, feats, head, deprel) in enumerate(rows, start=1):
        lines.append('\t'.join([str(idx), form, lemma, upos, xpos, feats, str(head), deprel, '_', '_']) + '\n')
    lines.append('\n')
    return lines

def moods(lines):
    """{(sent_id, token id): Mood} for every token in lines with a Mood feature"""
    result = {}
    sent_id = None
    for line in lines:
        if line.startswith('# sent_id = '):
            sent_id = line[len('# sent_id = '):].strip()
        elif line[:1].isdigit():
            columns = line.split('\t')
            for kv in columns[5].split('|'):
                if kv.startswith('Mood='):
                    result[(sent_id, int(columns[0]))] = kv[len('Mood='):]
    return result

def relabel(*sentences):
    """Relabel the given sentences, returning their moods and the Report"""
    lines = [line for s in sentences for line in s]
    output, report = relabel_mood.relabel_lines(lines)
    return moods(output), report

# Schep de soep en leg het brood.
#   two coordinated imperatives, neither tagged Imp
COORDINATED_IMPERATIVE = sentence('imp1', 'Schep de soep en leg het brood.', [
    ('Schep',  'scheppen', 'VERB',  'WW|pv|tgw|ev',               FIN_SING,       0, 'root'),
    ('de',     'de',       'DET',   'LID|bep|stan|rest',          'Definite=Def', 3, 'det'),
    ('soep',   'soep',     'NOUN',  'N|soort|ev|basis|zijd|stan', 'Number=Sing',  1, 'obj'),
    ('en',     'en',       'CCONJ', 'VG|neven',                   '_',            5, 'cc'),
    ('leg',    'leggen',   'VERB',  'WW|pv|tgw|ev',               FIN_SING,       1, 'conj'),
    ('het',    'het',      'DET',   'LID|bep|stan|evon',          'Definite=Def', 7, 'det'),
    ('brood',  'brood',    'NOUN',  'N|soort|ev|basis|onz|stan',  'Number=Sing',  5, 'obj'),
    ('.',      '.',        'PUNCT', 'LET',                        '_',            1, 'punct'),
])

# Ik schep de soep.
#   1sg declarative with the same verb form as the imperative
DECLARATIVE = sentence('decl1', 'Ik schep de soep.', [
    ('Ik',     'ik',       'PRON',  'VNW|pers|pron|nomin|vol|1|ev', 'Case=Nom|Person=1|PronType=Prs', 2, 'nsubj'),
    ('schep',  'scheppen', 'VERB',  'WW|pv|tgw|ev',                 FIN_SING,       0, 'root'),
    ('de',     'de',       'DET',   'LID|bep|stan|rest',            'Definite=Def', 4, 'det'),
    ('soep',   'soep',     'NOUN',  'N|soort|ev|basis|zijd|stan',   'Number=Sing',  2, 'obj'),
    ('.',      '.',        'PUNCT', 'LET',                          '_',            2, 'punct'),
])

# Hij komt binnen en zet koffie.
#   the subjectless conjunct "zet" shares the subject of "komt"
SHARED_SUBJECT = sentence('shared1', 'Hij komt binnen en zet koffie.', [
    ('Hij',    'hij',      'PRON',  'VNW|pers|pron|nomin|vol|3|ev|masc', 'Case=Nom|Person=3|PronType=Prs', 2, 'nsubj'),
    ('komt',   'komen',    'VERB',  'WW|pv|tgw|met-t',                   FIN_SING,      0, 'root'),
    ('binnen', 'binnen',   'ADV',   'BW',                                '_',           2, 'compound:prt'),
    ('en',     'en',       'CCONJ', 'VG|neven',                          '_',           5, 'cc'),
    ('zet',    'zetten',   'VERB',  'WW|pv|tgw|ev',                      FIN_SING,      2, 'conj'),
    ('koffie', 'koffie',   'NOUN',  'N|soort|ev|basis|zijd|stan',        'Number=Sing', 5, 'obj'),
    ('.',      '.',        'PUNCT', 'LET',                               '_',           2, 'punct'),
])

# Komt u morgen?
POLITE_QUESTION = sentence('q1', 'Komt u morgen?', [
    ('Komt',   'komen',    'VERB',  'WW|pv|tgw|met-t',                    FIN_SING, 0, 'root'),
    ('u',      'u',        'PRON',  'VNW|pers|pron|nomin|vol|2b|getal',   'Case=Nom|Person=2|PronType=Prs', 1, 'nsubj'),
    ('morgen', 'morgen',   'ADV',   'BW',                                 '_',      1, 'advmod'),
    ('?',      '?',        'PUNCT', 'LET',                                '_',      1, 'punct'),
])

# Gaat u zitten!
POLITE_IMPERATIVE = sentence('pol1', 'Gaat u zitten!', [
    ('Gaat',   'gaan',     'VERB',  'WW|pv|tgw|met-t',                    FIN_SING,       0, 'root'),
    ('u',      'u',        'PRON',  'VNW|pers|pron|nomin|vol|2b|getal',   'Case=Nom|Person=2|PronType=Prs', 1, 'nsubj'),
    ('zitten', 'zitten',   'VERB',  'WW|inf|vrij|zonder',                 'VerbForm=Inf', 1, 'xcomp'),
    ('!',      '!',        'PUNCT', 'LET',                                '_',            1, 'punct'),
])

def fronted_clause_sentence(sent_id, with_subject):
    """
    Als ik bel, neem op.       imperative after a fronted clause
    Als ik bel, neem jij op.   ordinary V2 word order, not an imperative
    """
    rows = [
        ('Als',  'als',      'SCONJ', 'VG|onder',                     '_',      3, 'mark'),
        ('ik',   'ik',       'PRON',  'VNW|pers|pron|nomin|vol|1|ev', 'Case=Nom|Person=1|PronType=Prs', 3, 'nsubj'),
        ('bel',  'bellen',   'VERB',  'WW|pv|tgw|ev',                 FIN_SING, 5, 'advcl'),
        (',',    ',',        'PUNCT', 'LET',                          '_',      3, 'punct'),
        ('neem', 'op_nemen', 'VERB',  'WW|pv|tgw|ev',                 FIN_SING, 0, 'root'),
    ]
    if with_subject:
        rows.append(('jij', 'jij', 'PRON', 'VNW|pers|pron|nomin|vol|2v|ev', 'Case=Nom|Person=2|PronType=Prs', 5, 'nsubj'))
    rows.append(('op', 'op', 'ADP',   'VZ|fin', '_', 5, 'compound:prt'))
    rows.append(('.',  '.',  'PUNCT', 'LET',    '_', 5, 'punct'))
    text = 'Als ik bel, neem jij op.' if with_subject else 'Als ik bel, neem op.'
    return sentence(sent_id, text, rows)

def comma_fronted_sentence(sent_id, with_subject):
    """
    In Outlook, ga naar Bestand.       imperative after a fronted PP and a comma
    In Outlook, ga jij naar Bestand.   ordinary V2 word order, not an imperative
    """
    rows = [
        ('In',      'in',      'ADP',   'VZ|init',                    '_',           2, 'case'),
        ('Outlook', 'Outlook', 'PROPN', 'N|eigen|ev|basis|zijd|stan', 'Number=Sing', 4, 'obl'),
        (',',       ',',       'PUNCT', 'LET',                        '_',           2, 'punct'),
        ('ga',      'gaan',    'VERB',  'WW|pv|tgw|ev',               FIN_SING,      0, 'root'),
    ]
    if with_subject:
        rows.append(('jij', 'jij', 'PRON', 'VNW|pers|pron|nomin|vol|2v|ev', 'Case=Nom|Person=2|PronType=Prs', 4, 'nsubj'))
    offset = len(rows)
    rows.append(('naar',    'naar',    'ADP',   'VZ|init',                   '_',           offset + 2, 'case'))
    rows.append(('Bestand', 'Bestand', 'PROPN', 'N|eigen|ev|basis|onz|stan', 'Number=Sing', 4,          'obl'))
    rows.append(('.',       '.',       'PUNCT', 'LET',                       '_',           4,          'punct'))
    text = 'In Outlook, ga jij naar Bestand.' if with_subject else 'In Outlook, ga naar Bestand.'
    return sentence(sent_id, text, rows)

def parataxis_conditional_sentence(sent_id, mark):
    """
    Als het misgaat herstart dan de computer.
    Zo het misgaat herstart dan de computer.

    The conditional clause is attached as parataxis, as alud does for "Als X, dan Y".
    There is no comma, so the imperative rule has to recognize the clause as
    subordinate from its mark.  With mark=False the fronted parataxis has an
    advmod instead of a mark, and the verb is not an imperative.
    """
    first = ('Als', 'als', 'SCONJ', 'VG|onder', '_', 3, 'mark') if mark else ('Zo', 'zo', 'ADV', 'BW', '_', 3, 'advmod')
    rows = [
        first,
        ('het',      'het',        'PRON',  'VNW|pers|pron|stan|red|3|ev|onz', 'Person=3|PronType=Prs', 3, 'nsubj'),
        ('misgaat',  'mis_gaan',   'VERB',  'WW|pv|tgw|met-t',                 FIN_SING,       4, 'parataxis'),
        ('herstart', 'herstarten', 'VERB',  'WW|pv|tgw|ev',                    FIN_SING,       0, 'root'),
        ('dan',      'dan',        'ADV',   'BW',                              '_',            4, 'advmod'),
        ('de',       'de',         'DET',   'LID|bep|stan|rest',               'Definite=Def', 7, 'det'),
        ('computer', 'computer',   'NOUN',  'N|soort|ev|basis|zijd|stan',      'Number=Sing',  4, 'obj'),
        ('.',        '.',          'PUNCT', 'LET',                             '_',            4, 'punct'),
    ]
    text = '%s het misgaat herstart dan de computer.' % first[0]
    return sentence(sent_id, text, rows)

# Leve de koning!
#   gold subjunctive tag, converted as Mood=Ind
SUBJUNCTIVE = sentence('sub1', 'Leve de koning!', [
    ('Leve',   'leven',    'VERB',  'WW|pv|conj|ev',              FIN_SING,       0, 'root'),
    ('de',     'de',       'DET',   'LID|bep|stan|rest',          'Definite=Def', 3, 'det'),
    ('koning', 'koning',   'NOUN',  'N|soort|ev|basis|zijd|stan', 'Number=Sing',  1, 'nsubj'),
    ('!',      '!',        'PUNCT', 'LET',                        '_',            1, 'punct'),
])

# Op het ogenblik praat Jan.
#   tagged Mood=Imp despite the overt subject
FALSE_IMPERATIVE = sentence('demote1', 'Op het ogenblik praat Jan.', [
    ('Op',       'op',       'ADP',   'VZ|init',                    '_',            3, 'case'),
    ('het',      'het',      'DET',   'LID|bep|stan|evon',          'Definite=Def', 3, 'det'),
    ('ogenblik', 'ogenblik', 'NOUN',  'N|soort|ev|basis|onz|stan',  'Number=Sing',  4, 'obl'),
    ('praat',    'praten',   'VERB',  'WW|pv|tgw|ev',               IMP_SING,       0, 'root'),
    ('Jan',      'Jan',      'PROPN', 'N|eigen|ev|basis|zijd|stan', 'Number=Sing',  4, 'nsubj'),
    ('.',        '.',        'PUNCT', 'LET',                        '_',            4, 'punct'),
])


# ---------------------------------------------------------------------------
# The imperative rule
# ---------------------------------------------------------------------------

class TestImperativeRule:

    def test_coordinated_imperatives(self):
        """Both conjuncts of a coordinated imperative become Mood=Imp"""
        result, report = relabel(COORDINATED_IMPERATIVE)
        assert result == {('imp1', 1): 'Imp', ('imp1', 5): 'Imp'}
        assert len(report.changes) == 2

    def test_declarative_unchanged(self):
        """A 1sg declarative has the same verb form, but a preverbal subject"""
        result, report = relabel(DECLARATIVE)
        assert result == {('decl1', 2): 'Ind'}
        assert report.changes == []

    def test_shared_subject_conj_unchanged(self):
        """A subjectless conjunct which shares its subject is not an imperative"""
        result, _ = relabel(SHARED_SUBJECT)
        assert result[('shared1', 5)] == 'Ind'

    def test_question_unchanged(self):
        """Komt u morgen? is a question, not a polite imperative"""
        result, _ = relabel(POLITE_QUESTION)
        assert result[('q1', 1)] == 'Ind'

    def test_polite_imperative(self):
        """Gaat u zitten! is an imperative despite the met-t form and the subject"""
        result, _ = relabel(POLITE_IMPERATIVE)
        assert result[('pol1', 1)] == 'Imp'

    def test_fronted_clause(self):
        """An imperative can follow a fronted clause, but a postverbal jij after one is V2 word order"""
        result, _ = relabel(fronted_clause_sentence('front1', False), fronted_clause_sentence('front2', True))
        assert result[('front1', 5)] == 'Imp'
        assert result[('front2', 5)] == 'Ind'
        # the verb of the subordinate clause is never a candidate
        assert result[('front1', 3)] == 'Ind'

    def test_comma_fronted(self):
        """Anything set off by a comma may precede an imperative, but not a V2 declarative"""
        result, _ = relabel(comma_fronted_sentence('comma1', False), comma_fronted_sentence('comma2', True))
        assert result[('comma1', 4)] == 'Imp'
        assert result[('comma2', 4)] == 'Ind'

    def test_fronted_without_comma(self):
        """Without the comma, a fronted constituent means V2 word order"""
        no_comma = sentence('comma3', 'In Outlook ga naar Bestand.', [
            ('In',      'in',      'ADP',   'VZ|init',                    '_',           2, 'case'),
            ('Outlook', 'Outlook', 'PROPN', 'N|eigen|ev|basis|zijd|stan', 'Number=Sing', 3, 'obl'),
            ('ga',      'gaan',    'VERB',  'WW|pv|tgw|ev',               FIN_SING,      0, 'root'),
            ('naar',    'naar',    'ADP',   'VZ|init',                    '_',           5, 'case'),
            ('Bestand', 'Bestand', 'PROPN', 'N|eigen|ev|basis|onz|stan',  'Number=Sing', 3, 'obl'),
            ('.',       '.',       'PUNCT', 'LET',                        '_',           3, 'punct'),
        ])
        result, _ = relabel(no_comma)
        assert result[('comma3', 3)] == 'Ind'

    def test_fronted_subordinate_parataxis(self):
        """Alpino attaches "Als X, dan Y" as parataxis; an imperative may follow it, even without a comma"""
        result, _ = relabel(parataxis_conditional_sentence('para1', True), parataxis_conditional_sentence('para2', False))
        assert result[('para1', 3)] == 'Ind'
        assert result[('para1', 4)] == 'Imp'
        # a fronted parataxis with no mark is not a subordinate clause
        assert result[('para2', 4)] == 'Ind'

    def test_existing_imperative_kept(self):
        """Existing Mood=Imp tokens are left alone, even ones the rule would not find"""
        result, report = relabel(FALSE_IMPERATIVE)
        assert result[('demote1', 4)] == 'Imp'
        assert report.changes == []


# ---------------------------------------------------------------------------
# The subjunctive rule
# ---------------------------------------------------------------------------

class TestSubjunctiveRule:

    def test_subjunctive(self):
        """A gold WW|pv|conj verb becomes Mood=Sub"""
        result, report = relabel(SUBJUNCTIVE)
        assert result == {('sub1', 1): 'Sub'}
        assert report.changes[0][3:5] == ('Ind', 'Sub')

    def test_subjunctive_overrides_imperative(self):
        """The gold subjunctive tag wins over an Imp from the converter, as in God beter het"""
        imperative = [line.replace('Mood=Ind', 'Mood=Imp') for line in SUBJUNCTIVE]
        result, _ = relabel(imperative)
        assert result == {('sub1', 1): 'Sub'}


# ---------------------------------------------------------------------------
# EXCLUDE, FORCE and DEMOTE
# ---------------------------------------------------------------------------

class TestExceptions:

    def test_demote(self, monkeypatch):
        """A DEMOTE entry turns an existing Imp into Ind"""
        monkeypatch.setattr(relabel_mood, 'DEMOTE', {('demote1', 4, 'praat')})
        result, report = relabel(FALSE_IMPERATIVE)
        assert result[('demote1', 4)] == 'Ind'
        assert not report.warnings

    def test_exclude(self, monkeypatch):
        """An EXCLUDE entry stops the rule from tagging a token"""
        monkeypatch.setattr(relabel_mood, 'EXCLUDE', {('imp1', 5, 'leg')})
        result, _ = relabel(COORDINATED_IMPERATIVE)
        assert result == {('imp1', 1): 'Imp', ('imp1', 5): 'Ind'}

    def test_force(self, monkeypatch):
        """A FORCE entry tags a token the rule does not reach"""
        monkeypatch.setattr(relabel_mood, 'FORCE', {('decl1', 2, 'schep')})
        result, _ = relabel(DECLARATIVE)
        assert result == {('decl1', 2): 'Imp'}

    def test_form_mismatch(self, monkeypatch):
        """An entry whose form does not match is not applied, and both problems are reported"""
        monkeypatch.setattr(relabel_mood, 'DEMOTE', {('demote1', 4, 'praten')})
        result, report = relabel(FALSE_IMPERATIVE)
        assert result[('demote1', 4)] == 'Imp'
        assert any('does not match form praat' in w for w in report.warnings)
        assert any('did not apply' in w for w in report.warnings)

    def test_entry_for_other_file_ignored(self, monkeypatch):
        """Entries for sentences which are not in this file are not reported"""
        monkeypatch.setattr(relabel_mood, 'DEMOTE', {('elsewhere', 4, 'praat')})
        _, report = relabel(FALSE_IMPERATIVE)
        assert not report.warnings


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------

class TestOutputFormat:

    def test_only_mood_changes(self):
        """Only the Mood feature changes; every other column and feature is untouched"""
        lines = [line for s in (COORDINATED_IMPERATIVE, SUBJUNCTIVE, DECLARATIVE) for line in s]
        output, _ = relabel_mood.relabel_lines(lines)
        assert len(output) == len(lines)
        for before, after in zip(lines, output):
            if before == after:
                continue
            before, after = before.split('\t'), after.split('\t')
            assert before[:5] == after[:5] and before[6:] == after[6:]
            assert [kv for kv in before[5].split('|') if not kv.startswith('Mood=')] == \
                   [kv for kv in after[5].split('|') if not kv.startswith('Mood=')]

    def test_input_not_modified(self):
        """relabel_lines does not modify the list it is given"""
        lines = list(COORDINATED_IMPERATIVE)
        relabel_mood.relabel_lines(lines)
        assert lines == COORDINATED_IMPERATIVE

    def test_set_mood(self):
        """set_mood keeps the features in UD order"""
        assert relabel_mood.set_mood('Number=Sing|Tense=Pres|VerbForm=Fin', 'Imp') == 'Mood=Imp|Number=Sing|Tense=Pres|VerbForm=Fin'
        assert relabel_mood.set_mood('Mood=Ind|Number=Sing', 'Sub') == 'Mood=Sub|Number=Sing'
        assert relabel_mood.set_mood('ExtPos=SCONJ|Mood=Ind|VerbForm=Fin', 'Sub') == 'ExtPos=SCONJ|Mood=Sub|VerbForm=Fin'
        assert relabel_mood.set_mood('_', 'Imp') == 'Mood=Imp'


# ---------------------------------------------------------------------------
# Convention and upstream checks
# ---------------------------------------------------------------------------

class TestChecks:

    def test_small_file_skips_convention_checks(self):
        """Files with few finite verbs get a note instead of the convention checks"""
        _, report = relabel(COORDINATED_IMPERATIVE)
        assert not report.warnings
        assert any('too few for the convention checks' in n for n in report.notes)

    def test_drift_warning(self, monkeypatch):
        """Subjects relabelled from nsubj to subj are reported loudly"""
        monkeypatch.setattr(relabel_mood, 'MIN_FINITE', 1)
        drifted = [line.replace('\tnsubj\t', '\tsubj\t') for line in DECLARATIVE]
        _, report = relabel(drifted)
        assert any('nsubj' in w for w in report.warnings)
        assert any('have a subject' in w for w in report.warnings)

    def test_upstream_fixed_note(self, monkeypatch):
        """If the subjunctives are already Mood=Sub, the Sub rule is reported as no longer needed"""
        monkeypatch.setattr(relabel_mood, 'MIN_FINITE', 1)
        monkeypatch.setattr(relabel_mood, 'CORE_DEPRELS', ('root',))
        monkeypatch.setattr(relabel_mood, 'MIN_SUBJECT_RATE', 0.0)
        already_sub = [line.replace('Mood=Ind', 'Mood=Sub') for line in SUBJUNCTIVE]
        result, report = relabel(already_sub)
        assert result == {('sub1', 1): 'Sub'}
        assert report.changes == []
        assert any('Sub rule is no longer needed' in n for n in report.notes)

    def test_check_false(self, monkeypatch):
        """check=False skips the convention checks, for data known to look different"""
        monkeypatch.setattr(relabel_mood, 'MIN_FINITE', 1)
        drifted = [line.replace('\tnsubj\t', '\tsubj\t') for line in DECLARATIVE]
        _, report = relabel_mood.relabel_lines(drifted, check=False)
        assert not report.warnings
        assert not report.notes

    def test_check_false_keeps_exception_warnings(self, monkeypatch):
        """check=False still reports exception list entries which do not match"""
        monkeypatch.setattr(relabel_mood, 'DEMOTE', {('demote1', 4, 'praten')})
        _, report = relabel_mood.relabel_lines(FALSE_IMPERATIVE, check=False)
        assert any('does not match form praat' in w for w in report.warnings)

    def test_no_upstream_note_with_warnings(self, monkeypatch):
        """With convention warnings, a lack of additions is not reported as an upstream fix"""
        monkeypatch.setattr(relabel_mood, 'MIN_FINITE', 1)
        already_sub = [line.replace('Mood=Ind', 'Mood=Sub') for line in SUBJUNCTIVE]
        _, report = relabel(already_sub)
        assert report.warnings
        assert not any('no longer needed' in n for n in report.notes)


# ---------------------------------------------------------------------------
# File API and command line
# ---------------------------------------------------------------------------

class TestFiles:

    def test_relabel_file(self):
        """relabel_file writes the relabeled file and appends to the changes file"""
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
            input_path = os.path.join(test_dir, 'in.conllu')
            output_path = os.path.join(test_dir, 'out.conllu')
            changes_path = os.path.join(test_dir, 'changes.tsv')
            with open(input_path, 'w', encoding='utf-8') as fout:
                fout.write(''.join(COORDINATED_IMPERATIVE + SUBJUNCTIVE))

            report = relabel_mood.relabel_file(input_path, output_path, changes_path)
            assert len(report.changes) == 3

            with open(output_path, encoding='utf-8') as fin:
                assert moods(fin.readlines()) == {('imp1', 1): 'Imp', ('imp1', 5): 'Imp', ('sub1', 1): 'Sub'}
            with open(changes_path, encoding='utf-8') as fin:
                changes = fin.read().splitlines()
            assert len(changes) == 3
            assert all(line.startswith('in.conllu\t') for line in changes)

    def test_main(self, capsys):
        """main accepts its arguments as a list and prints a summary"""
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
            input_path = os.path.join(test_dir, 'in.conllu')
            output_path = os.path.join(test_dir, 'out.conllu')
            with open(input_path, 'w', encoding='utf-8') as fout:
                fout.write(''.join(COORDINATED_IMPERATIVE))
            relabel_mood.main([input_path, output_path])
            assert '2 -> Imp' in capsys.readouterr().err
            assert os.path.exists(output_path)

    def test_main_exits_on_warning(self, monkeypatch):
        """main exits with status 1 if there are any warnings"""
        monkeypatch.setattr(relabel_mood, 'DEMOTE', {('demote1', 4, 'praten')})
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
            input_path = os.path.join(test_dir, 'in.conllu')
            with open(input_path, 'w', encoding='utf-8') as fout:
                fout.write(''.join(FALSE_IMPERATIVE))
            with pytest.raises(SystemExit) as exc:
                relabel_mood.main([input_path, os.path.join(test_dir, 'out.conllu')])
            assert exc.value.code == 1
