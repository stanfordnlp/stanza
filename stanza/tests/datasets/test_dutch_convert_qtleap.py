"""
Tests for stanza/utils/datasets/dutch/convert_qtleap.py

The test sentences are small handmade trees in the form alud produces for
qtleap1a, with sent_ids from the file names.  TREE_FIXES lists sentences of
the real qtleap1a, so the tests which convert a document replace it with
monkeypatch.
"""

import os
import tempfile

import pytest

from stanza.tests import TEST_WORKING_DIR
from stanza.utils.datasets.dutch import convert_qtleap
from stanza.utils.datasets.dutch.merge_colon_sentences import read_sentences, sentence_comment

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FIN_SING = 'Mood=Ind|Number=Sing|Tense=Pres|VerbForm=Fin'

def sentence(sent_id, text, rows):
    """
    Build a sentence, as a list of lines without newlines, from
    (form, lemma, upos, xpos, feats, head, deprel) rows
    """
    lines = ['# source = qtleap1a/%s.xml' % sent_id, '# sent_id = %s' % sent_id, '# text = %s' % text]
    for idx, (form, lemma, upos, xpos, feats, head, deprel) in enumerate(rows, start=1):
        lines.append('\t'.join([str(idx), form, lemma, upos, xpos, feats, str(head), deprel, '%d:%s' % (head, deprel), '_']))
    return lines

def ga_naar(sent_id, xpos='WW|pv|tgw|ev'):
    """Ga naar Instellingen."""
    return sentence(sent_id, 'Ga naar Instellingen.', [
        ('Ga',           'gaan',         'VERB',  xpos,                    FIN_SING,      0, 'root'),
        ('naar',         'naar',         'ADP',   'VZ|init',               '_',           3, 'case'),
        ('Instellingen', 'Instellingen', 'PROPN', 'N|eigen|mv|basis',      'Number=Plur', 1, 'obl'),
        ('.',            '.',            'PUNCT', 'LET',                   '_',           1, 'punct'),
    ])

# Reset de router.
#   the object is misparsed as the subject, as in the real qtleap1a
RESET_ROUTER = sentence('Batch1_answers_nl-5-1', 'Reset de router.', [
    ('Reset',  'resetten', 'VERB',  'WW|pv|tgw|ev',               FIN_SING,       0, 'root'),
    ('de',     'de',       'DET',   'LID|bep|stan|rest',          'Definite=Def', 3, 'det'),
    ('router', 'router',   'NOUN',  'N|soort|ev|basis|zijd|stan', 'Number=Sing',  1, 'nsubj'),
    ('.',      '.',        'PUNCT', 'LET',                        '_',            1, 'punct'),
])

def rows_of(lines):
    """{(sent_id, token id): columns} for every token line"""
    result = {}
    sent_id = None
    for line in lines:
        if line.startswith('# sent_id = '):
            sent_id = line[len('# sent_id = '):].strip()
        elif line[:1].isdigit():
            columns = line.rstrip('\n').split('\t')
            result[(sent_id, int(columns[0]))] = columns
    return result

def convert(monkeypatch, *sentences, fixes=None):
    monkeypatch.setattr(convert_qtleap, 'TREE_FIXES', fixes or {})
    return convert_qtleap.convert_sentences([list(s) for s in sentences])


# ---------------------------------------------------------------------------
# Document structure
# ---------------------------------------------------------------------------

class TestDocument:

    def test_document_order_and_newdoc(self, monkeypatch):
        """Sentences are sorted by answer and sentence number, with a newdoc for each answer"""
        lines, _ = convert(monkeypatch, ga_naar('Batch1_answers_nl-10-1'), ga_naar('Batch1_answers_nl-2-2'), ga_naar('Batch1_answers_nl-2-1'))
        ids = [line.strip() for line in lines if line.startswith('# sent_id') or line.startswith('# newdoc')]
        assert ids == ['# newdoc id = Batch1_answers_nl-2',
                       '# sent_id = Batch1_answers_nl-2-1',
                       '# sent_id = Batch1_answers_nl-2-2',
                       '# newdoc id = Batch1_answers_nl-10',
                       '# sent_id = Batch1_answers_nl-10-1']

    def test_unexpected_sent_id(self, monkeypatch):
        """sent_ids which are not qtleap1a file names are an error"""
        with pytest.raises(ValueError):
            convert(monkeypatch, ga_naar('cdb\\1'))

    def test_unix_newlines(self, monkeypatch):
        """Every output line ends with a single Unix newline"""
        lines, _ = convert(monkeypatch, ga_naar('Batch1_answers_nl-1-1'))
        assert all(line.endswith('\n') and not line.endswith('\r\n') for line in lines)


# ---------------------------------------------------------------------------
# Mood
# ---------------------------------------------------------------------------

class TestMood:

    def test_imperative(self, monkeypatch):
        """Imperatives are tagged by relabel_mood"""
        lines, report = convert(monkeypatch, ga_naar('Batch1_answers_nl-1-1'))
        assert 'Mood=Imp' in rows_of(lines)[('Batch1_answers_nl-1-1', 1)][5]
        assert not report.warnings

    def test_ga_retagged(self, monkeypatch):
        """Imperative ga tagged as a subjunctive is retagged, and so becomes Mood=Imp rather than Sub"""
        lines, _ = convert(monkeypatch, ga_naar('Batch1_answers_nl-1-1', xpos='WW|pv|conj|ev'))
        columns = rows_of(lines)[('Batch1_answers_nl-1-1', 1)]
        assert columns[4] == 'WW|pv|tgw|ev'
        assert 'Mood=Imp' in columns[5]

    def test_no_convention_warnings(self, monkeypatch):
        """A document of imperatives does not set off the treebank convention checks"""
        sentences = [ga_naar('Batch1_answers_nl-%d-1' % i) for i in range(1, 601)]
        _, report = convert(monkeypatch, *sentences)
        assert not report.warnings
        assert len(report.changes) == 600


# ---------------------------------------------------------------------------
# Tree fixes
# ---------------------------------------------------------------------------

class TestTreeFixes:

    def test_object_fix(self, monkeypatch):
        """A misparsed subject becomes the object, after which the verb is an imperative"""
        fixes = {'Batch1_answers_nl-5-1': [(3, 'router', convert_qtleap.object_of(1))]}
        lines, _ = convert(monkeypatch, RESET_ROUTER, fixes=fixes)
        rows = rows_of(lines)
        assert rows[('Batch1_answers_nl-5-1', 3)][6:9] == ['1', 'obj', '1:obj']
        assert 'Mood=Imp' in rows[('Batch1_answers_nl-5-1', 1)][5]

    def test_unfixed_tree_is_not_imperative(self, monkeypatch):
        """Without the fix, the misparsed subject blocks the imperative rule"""
        lines, _ = convert(monkeypatch, RESET_ROUTER)
        assert 'Mood=Ind' in rows_of(lines)[('Batch1_answers_nl-5-1', 1)][5]

    def test_possessive_je(self):
        """possessive_je makes je the possessive of the given noun"""
        edits = convert_qtleap.possessive_je(3)
        assert edits['HEAD'] == '3'
        assert edits['DEPREL'] == 'nmod:poss'
        assert edits['DEPS'] == '3:nmod:poss'
        assert 'Poss=Yes' in edits['FEATS']
        assert edits['XPOS'].startswith('VNW|bez|det')

    def test_form_mismatch(self, monkeypatch):
        """A fix whose form does not match the token is an error"""
        fixes = {'Batch1_answers_nl-5-1': [(3, 'modem', convert_qtleap.object_of(1))]}
        with pytest.raises(ValueError):
            convert(monkeypatch, RESET_ROUTER, fixes=fixes)

    def test_missing_token(self, monkeypatch):
        """A fix for a token which is not in the sentence is an error"""
        fixes = {'Batch1_answers_nl-5-1': [(9, 'router', convert_qtleap.object_of(1))]}
        with pytest.raises(ValueError):
            convert(monkeypatch, RESET_ROUTER, fixes=fixes)

    def test_missing_sentence(self, monkeypatch):
        """A fix for a sentence which is not in the input is an error"""
        fixes = {'Batch1_answers_nl-99-1': [(3, 'router', convert_qtleap.object_of(1))]}
        with pytest.raises(ValueError):
            convert(monkeypatch, RESET_ROUTER, fixes=fixes)

    def test_fixes_cover_real_data_layout(self):
        """Every entry of the real TREE_FIXES has a qtleap1a sent_id and edits only known columns"""
        for sent_id, fixes in convert_qtleap.TREE_FIXES.items():
            assert convert_qtleap.SENT_ID_RE.match(sent_id)
            for token_id, form, edits in fixes:
                assert isinstance(token_id, int) and form
                assert set(edits) <= set(convert_qtleap.COLUMNS)


# ---------------------------------------------------------------------------
# File API and command line
# ---------------------------------------------------------------------------

def write_sentences(path, sentences):
    with open(path, 'w', encoding='utf-8') as fout:
        for s in sentences:
            fout.write('\n'.join(s))
            fout.write('\n\n')

class TestFiles:

    def test_convert_file(self, monkeypatch):
        """convert_file writes the converted document"""
        monkeypatch.setattr(convert_qtleap, 'TREE_FIXES', {})
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
            input_path = os.path.join(test_dir, 'in.conllu')
            output_path = os.path.join(test_dir, 'out.conllu')
            write_sentences(input_path, [ga_naar('Batch1_answers_nl-1-2'), ga_naar('Batch1_answers_nl-1-1')])
            report = convert_qtleap.convert_file(input_path, output_path)
            assert len(report.changes) == 2
            with open(output_path, encoding='utf-8') as fin:
                sentences = read_sentences(fin)
            assert [sentence_comment(s, 'sent_id') for s in sentences] == ['Batch1_answers_nl-1-1', 'Batch1_answers_nl-1-2']
            assert sentence_comment(sentences[0], 'newdoc id') == 'Batch1_answers_nl-1'

    def test_main(self, monkeypatch, capsys):
        """main accepts its arguments as a list and prints a summary"""
        monkeypatch.setattr(convert_qtleap, 'TREE_FIXES', {})
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
            input_path = os.path.join(test_dir, 'in.conllu')
            output_path = os.path.join(test_dir, 'out.conllu')
            write_sentences(input_path, [ga_naar('Batch1_answers_nl-1-1')])
            convert_qtleap.main([input_path, output_path])
            assert '1 -> Imp' in capsys.readouterr().err
            assert os.path.exists(output_path)
