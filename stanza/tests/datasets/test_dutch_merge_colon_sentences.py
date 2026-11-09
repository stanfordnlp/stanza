"""
Tests for stanza/utils/datasets/dutch/merge_colon_sentences.py

The test sentences are small handmade trees built by the sentence() helper
from (form, upos, head, deprel) rows, plus optional enhanced deps and misc.
The cdb sent_ids are what trigger merging, as in UD_Dutch-Alpino.

  cdb\\10   Hij zegt:
  cdb\\11   Ik kom.

should become one sentence, cdb\\10, in which "kom" is a parataxis of
"zegt" and the colon is punct of "kom".
"""

import os
import tempfile

import pytest

from stanza.tests import TEST_WORKING_DIR
from stanza.utils.datasets.dutch import merge_colon_sentences

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def sentence(sent_id, text, rows):
    """
    Build a sentence, as a list of lines, from rows

    Each row is (id, form, upos, head, deprel, deps, misc).  deps defaults to
    head:deprel and misc to _ when the row is shorter.  id may be None for
    the next regular token, or a string for MWT ranges and empty nodes.
    """
    lines = ['# sent_id = %s' % sent_id, '# text = %s' % text]
    next_id = 1
    for row in rows:
        token_id, form, upos, head, deprel = row[:5]
        deps = row[5] if len(row) > 5 else '%s:%s' % (head, deprel)
        misc = row[6] if len(row) > 6 else '_'
        if token_id is None:
            token_id = str(next_id)
            next_id += 1
        lines.append('\t'.join([token_id, form, form.lower(), upos, '_', '_', str(head), deprel, deps, misc]))
    return lines

HIJ_ZEGT = sentence('cdb\\10', 'Hij zegt:', [
    (None, 'Hij',  'PRON',  2, 'nsubj'),
    (None, 'zegt', 'VERB',  0, 'root', '0:root', 'SpaceAfter=No'),
    (None, ':',    'PUNCT', 2, 'punct'),
])

IK_KOM = sentence('cdb\\11', 'Ik kom.', [
    (None, 'Ik',  'PRON',  2, 'nsubj'),
    (None, 'kom', 'VERB',  0, 'root', '0:root', 'SpaceAfter=No'),
    (None, '.',   'PUNCT', 2, 'punct'),
])

def columns_of(sentence_lines):
    """{token id: columns} for every token line, including empty nodes and MWTs"""
    return {line.split('\t')[0]: line.split('\t') for line in sentence_lines if not line.startswith('#')}

def merge(*sentences):
    return merge_colon_sentences.merge_sentences([list(s) for s in sentences])


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

class TestMerge:

    def test_simple_merge(self):
        """Hij zegt: + Ik kom. becomes one sentence, attached as the treebank attaches colons"""
        output, report = merge(HIJ_ZEGT, IK_KOM)
        assert len(output) == 1
        assert report.merged == [['cdb\\10', 'cdb\\11']]

        merged = output[0]
        assert merge_colon_sentences.sentence_comment(merged, 'sent_id') == 'cdb\\10'
        assert merge_colon_sentences.sentence_comment(merged, 'text') == 'Hij zegt: Ik kom.'
        assert merge_colon_sentences.sentence_comment(merged, 'merged_sent_id') == 'cdb\\11'

        columns = columns_of(merged)
        assert list(columns) == ['1', '2', '3', '4', '5', '6']
        # the colon is punct of the continuation's root
        assert columns['3'][6:9] == ['5', 'punct', '5:punct']
        # the continuation's root is a parataxis of the first root
        assert columns['5'][6:9] == ['2', 'parataxis', '2:parataxis']
        # the rest of the continuation is shifted
        assert columns['4'][6:9] == ['5', 'nsubj', '5:nsubj']
        assert columns['6'][6:9] == ['5', 'punct', '5:punct']
        # the first sentence is otherwise unchanged
        assert columns['2'][6:10] == ['0', 'root', '0:root', 'SpaceAfter=No']

    def test_space_after_no_on_colon(self):
        """A colon with SpaceAfter=No is joined to the continuation without a space"""
        no_space = [line.replace('\tpunct\t2:punct\t_', '\tpunct\t2:punct\tSpaceAfter=No') for line in HIJ_ZEGT]
        output, _ = merge(no_space, IK_KOM)
        assert merge_colon_sentences.sentence_comment(output[0], 'text') == 'Hij zegt:Ik kom.'

    def test_chain(self):
        """In A: B: C, C is attached to the clause ending with the second colon"""
        b = sentence('cdb\\11', 'Ik weet:', [
            (None, 'Ik',   'PRON',  2, 'nsubj'),
            (None, 'weet', 'VERB',  0, 'root', '0:root', 'SpaceAfter=No'),
            (None, ':',    'PUNCT', 2, 'punct'),
        ])
        c = sentence('cdb\\12', 'Ja.', [
            (None, 'Ja', 'INTJ',  0, 'root', '0:root', 'SpaceAfter=No'),
            (None, '.',  'PUNCT', 1, 'punct'),
        ])
        output, report = merge(HIJ_ZEGT, b, c)
        assert len(output) == 1
        assert report.merged == [['cdb\\10', 'cdb\\11', 'cdb\\12']]
        merged = output[0]
        assert merge_colon_sentences.sentence_comment(merged, 'text') == 'Hij zegt: Ik weet: Ja.'
        assert [line for line in merged if line.startswith('# merged_sent_id')] == \
            ['# merged_sent_id = cdb\\11', '# merged_sent_id = cdb\\12']
        columns = columns_of(merged)
        assert columns['5'][6:8] == ['2', 'parataxis']   # weet -> zegt
        assert columns['7'][6:8] == ['5', 'parataxis']   # Ja -> weet
        assert columns['6'][6:8] == ['7', 'punct']       # second colon -> Ja

    def test_continuation_earlier_in_file(self):
        """The continuation is found and removed even if it comes first in the file"""
        output, report = merge(IK_KOM, HIJ_ZEGT)
        assert len(output) == 1
        assert report.merged == [['cdb\\10', 'cdb\\11']]

    def test_other_sentences_keep_their_order(self):
        """Unrelated sentences are left in place around the merged sentence"""
        other = sentence('cdb\\50', 'Nee.', [
            (None, 'Nee', 'INTJ',  0, 'root', '0:root', 'SpaceAfter=No'),
            (None, '.',   'PUNCT', 1, 'punct'),
        ])
        output, _ = merge(other, HIJ_ZEGT, IK_KOM)
        assert [merge_colon_sentences.sentence_comment(s, 'sent_id') for s in output] == ['cdb\\50', 'cdb\\10']
        assert output[0] == other


# ---------------------------------------------------------------------------
# Sentences which are not merged
# ---------------------------------------------------------------------------

class TestNotMerged:

    def test_no_continuation(self):
        """A colon-final sentence whose continuation is missing is reported and left alone"""
        output, report = merge(HIJ_ZEGT)
        assert output == [HIJ_ZEGT]
        assert report.merged == []
        assert report.unmerged == ['cdb\\10']

    def test_other_section(self):
        """Only the sections in MERGE_SECTIONS are merged"""
        first = [line.replace('cdb\\10', 'g_suite\\10') for line in HIJ_ZEGT]
        second = [line.replace('cdb\\11', 'g_suite\\11') for line in IK_KOM]
        output, report = merge(first, second)
        assert output == [first, second]
        assert report.merged == [] and report.unmerged == []

    def test_mid_sentence_colon(self):
        """A colon inside a sentence is not a reason to merge"""
        mid = sentence('cdb\\10', 'Hij zegt: ja.', [
            (None, 'Hij',  'PRON',  2, 'nsubj'),
            (None, 'zegt', 'VERB',  0, 'root', '0:root', 'SpaceAfter=No'),
            (None, ':',    'PUNCT', 4, 'punct'),
            (None, 'ja',   'INTJ',  2, 'parataxis', '2:parataxis', 'SpaceAfter=No'),
            (None, '.',    'PUNCT', 2, 'punct'),
        ])
        output, report = merge(mid, IK_KOM)
        assert output == [mid, IK_KOM]
        assert report.merged == []


# ---------------------------------------------------------------------------
# Renumbering
# ---------------------------------------------------------------------------

class TestRenumbering:

    def test_shift_id(self):
        assert merge_colon_sentences.shift_id('3', 4) == '7'
        assert merge_colon_sentences.shift_id('3-4', 4) == '7-8'
        assert merge_colon_sentences.shift_id('3.1', 4) == '7.1'
        with pytest.raises(ValueError):
            merge_colon_sentences.shift_id('a', 4)

    def test_shift_deps(self):
        assert merge_colon_sentences.shift_deps('_', 3, '2:parataxis') == '_'
        assert merge_colon_sentences.shift_deps('0:root', 3, '2:parataxis') == '2:parataxis'
        assert merge_colon_sentences.shift_deps('1:nsubj|4.1:nsubj:xsubj', 3, '2:parataxis') == '4:nsubj|7.1:nsubj:xsubj'
        # relation names with colons are kept intact
        assert merge_colon_sentences.shift_deps('1:conj:en', 3, '2:parataxis') == '4:conj:en'

    def test_shift_misc(self):
        assert merge_colon_sentences.shift_misc('CopiedFrom=3', 4) == 'CopiedFrom=7'
        assert merge_colon_sentences.shift_misc('SpaceAfter=No|CopiedFrom=3', 4) == 'SpaceAfter=No|CopiedFrom=7'
        assert merge_colon_sentences.shift_misc('SpaceAfter=No', 4) == 'SpaceAfter=No'

    def test_empty_nodes_and_mwt_in_continuation(self):
        """Empty nodes, their CopiedFrom, multi-head deps and MWT ranges are all shifted"""
        continuation = sentence('cdb\\11', 'Ik kom en jij.', [
            (None,  'Ik',  'PRON',  2, 'nsubj', '2:nsubj|4.1:nsubj'),
            (None,  'kom', 'VERB',  0, 'root'),
            (None,  'en',  'CCONJ', 4, 'cc', '4.1:cc'),
            (None,  'jij', 'PRON',  2, 'conj', '4.1:nsubj'),
            ('4.1', 'kom', 'VERB',  '_', '_', '2:conj:en', 'CopiedFrom=2'),
            ('5-6', 'jij.', '_',    '_', '_', '_'),
            (None,  'x',   'X',     4, 'dep'),
            (None,  '.',   'PUNCT', 2, 'punct'),
        ])
        output, _ = merge(HIJ_ZEGT, continuation)
        columns = columns_of(output[0])
        assert columns['4'][8] == '5:nsubj|7.1:nsubj'
        assert columns['7.1'][6:10] == ['_', '_', '5:conj:en', 'CopiedFrom=5']
        assert '8-9' in columns
        assert columns['8'][6] == '7'


# ---------------------------------------------------------------------------
# Tree checks
# ---------------------------------------------------------------------------

class TestCheckTree:

    def test_valid(self):
        merge_colon_sentences.check_tree(HIJ_ZEGT)

    def test_two_roots(self):
        bad = [line.replace('\t2\tnsubj\t2:nsubj', '\t0\troot\t0:root') for line in HIJ_ZEGT]
        with pytest.raises(ValueError):
            merge_colon_sentences.check_tree(bad)

    def test_cycle(self):
        bad = sentence('cdb\\1', 'a b c', [
            (None, 'a', 'X', 0, 'root'),
            (None, 'b', 'X', 3, 'dep'),
            (None, 'c', 'X', 2, 'dep'),
        ])
        with pytest.raises(ValueError):
            merge_colon_sentences.check_tree(bad)

    def test_head_out_of_range(self):
        bad = [line.replace('\t2\tpunct\t2:punct', '\t9\tpunct\t9:punct') for line in HIJ_ZEGT]
        with pytest.raises(ValueError):
            merge_colon_sentences.check_tree(bad)


# ---------------------------------------------------------------------------
# File API and command line
# ---------------------------------------------------------------------------

def write_sentences(path, sentences):
    with open(path, 'w', encoding='utf-8') as fout:
        for s in sentences:
            fout.write('\n'.join(s))
            fout.write('\n\n')

class TestFiles:

    def test_merge_file(self):
        """merge_file writes the merged sentences and returns the report"""
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
            input_path = os.path.join(test_dir, 'in.conllu')
            output_path = os.path.join(test_dir, 'out.conllu')
            write_sentences(input_path, [HIJ_ZEGT, IK_KOM])
            report = merge_colon_sentences.merge_file(input_path, output_path)
            assert report.merged == [['cdb\\10', 'cdb\\11']]
            with open(output_path, encoding='utf-8') as fin:
                sentences = merge_colon_sentences.read_sentences(fin)
            assert len(sentences) == 1
            assert merge_colon_sentences.sentence_comment(sentences[0], 'text') == 'Hij zegt: Ik kom.'

    def test_unmerged_file_unchanged(self):
        """A file with nothing to merge is written back byte for byte"""
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
            input_path = os.path.join(test_dir, 'in.conllu')
            output_path = os.path.join(test_dir, 'out.conllu')
            not_cdb = [line.replace('cdb\\10', 'g_suite\\10') for line in HIJ_ZEGT]
            write_sentences(input_path, [IK_KOM, not_cdb])
            merge_colon_sentences.merge_file(input_path, output_path)
            with open(input_path, encoding='utf-8') as fin:
                expected = fin.read()
            with open(output_path, encoding='utf-8') as fin:
                assert fin.read() == expected

    def test_main(self, capsys):
        """main accepts its arguments as a list and prints a summary"""
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
            input_path = os.path.join(test_dir, 'in.conllu')
            output_path = os.path.join(test_dir, 'out.conllu')
            write_sentences(input_path, [HIJ_ZEGT, IK_KOM])
            merge_colon_sentences.main([input_path, output_path])
            assert 'merged 1 colon-final sentences into 1 sentences' in capsys.readouterr().err
            assert os.path.exists(output_path)
