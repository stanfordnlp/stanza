"""
Merge colon-final sentences in UD_Dutch-Alpino with the sentence that continues them.

The cdb section of UD_Dutch-Alpino (the Eindhoven corpus material) was
segmented with a sentence break after every colon, so the treebank has
"sentences" such as

  cdb\\1887   Hij zegt:
  cdb\\1888   " Inderdaad dit kan ik niet ontkennen.

About 3% of Alpino train ends in a colon this way, which teaches the
sentence splitter to split after colons.  In ordinary Dutch text, as in
English, the colon introduces speech or a list within the sentence.

This script merges each colon-final cdb\\N with cdb\\N+1, its continuation
in the original text.  A chain of colon-final sentences is merged into one.
The merged sentences are intended as tokenizer training data, but they are
still well formed trees, built the way the treebank attaches material after
a mid-sentence colon in the large majority of cases:

  - the root of the continuation becomes a parataxis of the clause the
    colon ends, which is the first root unless several sentences are chained
  - the colon becomes punct of the continuation's root

The merged sentence keeps the first sent_id, and each sentence merged into
it is recorded in a "# merged_sent_id" comment.  Sentences which are not
merged are written back unchanged.

Usage:

  python3 -m stanza.utils.datasets.dutch.merge_colon_sentences in.conllu out.conllu

or from other code, merge_file(input_path, output_path) or merge_sentences(sentences)
"""

import argparse
import re
import sys

# sent_ids of the sections segmented after colons, with the sentence number
MERGE_SECTIONS = re.compile(r'^(cdb)\\(\d+)$')

ID_RE = re.compile(r'^(\d+)(?:([-.])(\d+))?$')
COPIED_FROM_RE = re.compile(r'(^|\|)CopiedFrom=(\d+)(?=\||$)')


class MergeReport:
    """
    The result of merging one document

    merged is a list of the sent_ids in each merged sentence, in order
    unmerged is a list of colon-final sent_ids in MERGE_SECTIONS with no continuation
    """
    def __init__(self):
        self.merged = []
        self.unmerged = []


def read_sentences(lines):
    """Split the lines of a CoNLL-U document into sentences, as lists of lines without newlines"""
    sentences = []
    current = []
    for line in lines:
        line = line.rstrip('\r\n')
        if line.strip() == '':
            if current:
                sentences.append(current)
                current = []
            continue
        current.append(line)
    if current:
        sentences.append(current)
    return sentences


def sentence_comment(sentence, key):
    prefix = '# %s = ' % key
    for line in sentence:
        if line.startswith(prefix):
            return line[len(prefix):]
    return None


def token_rows(sentence):
    """The columns of the regular tokens, skipping comments, MWT ranges, and empty nodes"""
    rows = []
    for line in sentence:
        if line.startswith('#'):
            continue
        columns = line.split('\t')
        if columns[0].isdigit():
            rows.append(columns)
    return rows


def ends_in_colon(sentence):
    rows = token_rows(sentence)
    return bool(rows) and rows[-1][1] == ':'


def shift_id(token_id, delta):
    """Shift a token id, MWT range (3-4), or empty node id (5.1) by delta"""
    match = ID_RE.match(token_id)
    if not match:
        raise ValueError('Unknown token id %s' % token_id)
    start = str(int(match.group(1)) + delta)
    if match.group(2) == '-':
        return '%s-%d' % (start, int(match.group(3)) + delta)
    if match.group(2) == '.':
        return '%s.%s' % (start, match.group(3))
    return start


def shift_deps(deps, delta, new_root):
    """
    Shift the heads of an enhanced deps column by delta

    A 0:root relation is replaced with new_root, for example 3:parataxis
    """
    if deps == '_':
        return deps
    pieces = []
    for piece in deps.split('|'):
        head, relation = piece.split(':', 1)
        if head == '0':
            pieces.append(new_root)
        else:
            pieces.append('%s:%s' % (shift_id(head, delta), relation))
    return '|'.join(pieces)


def shift_misc(misc, delta):
    """Shift the token referenced by CopiedFrom, which UD Dutch uses on empty nodes"""
    return COPIED_FROM_RE.sub(lambda m: '%sCopiedFrom=%d' % (m.group(1), int(m.group(2)) + delta), misc)


def root_id(sentence):
    roots = [int(columns[0]) for columns in token_rows(sentence) if columns[6] == '0']
    if len(roots) != 1:
        raise ValueError('Expected exactly one root in %s, found %d' % (sentence_comment(sentence, 'sent_id'), len(roots)))
    return roots[0]


def merge_pair(first, second):
    """
    Merge the continuation second into the colon-final sentence first

    Returns the merged sentence as a list of lines
    """
    first_rows = token_rows(first)
    offset = len(first_rows)
    colon_id = offset
    # the continuation attaches to the clause the colon ends, which is the
    # root of first unless first is itself the result of an earlier merge
    attach_to = int(first_rows[-1][6])
    second_root = root_id(second) + offset

    first_text = sentence_comment(first, 'text')
    second_text = sentence_comment(second, 'text')
    if first_text is None or second_text is None:
        raise ValueError('Cannot merge sentences without # text: %s %s' %
                         (sentence_comment(first, 'sent_id'), sentence_comment(second, 'sent_id')))
    colon_misc = first_rows[-1][9]
    separator = '' if 'SpaceAfter=No' in colon_misc.split('|') else ' '
    text = first_text + separator + second_text.lstrip()

    merged = []
    for line in first:
        if line.startswith('# text = '):
            merged.append('# text = ' + text)
            continue
        if line.startswith('#'):
            merged.append(line)
            continue
        columns = line.split('\t')
        if columns[0] == str(colon_id):
            columns[6] = str(second_root)
            if columns[8] != '_':
                columns[8] = '%d:%s' % (second_root, columns[7])
        merged.append('\t'.join(columns))

    comment_index = max(i for i, line in enumerate(merged) if line.startswith('#')) + 1
    merged.insert(comment_index, '# merged_sent_id = %s' % sentence_comment(second, 'sent_id'))

    for line in second:
        if line.startswith('#'):
            continue
        columns = line.split('\t')
        columns[0] = shift_id(columns[0], offset)
        if columns[6] == '0':
            columns[6] = str(attach_to)
            columns[7] = 'parataxis'
        elif columns[6] != '_':
            columns[6] = str(int(columns[6]) + offset)
        columns[8] = shift_deps(columns[8], offset, '%d:parataxis' % attach_to)
        columns[9] = shift_misc(columns[9], offset)
        merged.append('\t'.join(columns))
    check_tree(merged)
    return merged


def check_tree(sentence):
    """Raise ValueError if the sentence does not have contiguous ids and a single rooted tree"""
    sent_id = sentence_comment(sentence, 'sent_id')
    rows = token_rows(sentence)
    ids = [int(columns[0]) for columns in rows]
    if ids != list(range(1, len(rows) + 1)):
        raise ValueError('Token ids of %s are not contiguous' % sent_id)
    heads = {int(columns[0]): int(columns[6]) for columns in rows}
    if any(head < 0 or head > len(rows) for head in heads.values()):
        raise ValueError('A head in %s is out of range' % sent_id)
    root_id(sentence)
    for token in heads:
        seen = set()
        while token != 0:
            if token in seen:
                raise ValueError('Cycle in %s' % sent_id)
            seen.add(token)
            token = heads[token]


def continuation_id(sent_id):
    match = MERGE_SECTIONS.match(sent_id or '')
    if not match:
        return None
    return '%s\\%d' % (match.group(1), int(match.group(2)) + 1)


def merge_sentences(sentences):
    """
    Merge colon-final sentences in sentences, a list of lists of lines

    Returns the new list of sentences and a MergeReport.  A merged sentence
    takes the place of its first sentence, and the continuations are removed
    from wherever they were in the file.
    """
    report = MergeReport()
    by_id = {sentence_comment(s, 'sent_id'): s for s in sentences}

    # first find every sentence which continues a colon-final sentence,
    # so it can be skipped no matter where it is in the file
    continuations = set()
    for sentence in sentences:
        if not ends_in_colon(sentence):
            continue
        sent_id = sentence_comment(sentence, 'sent_id')
        next_id = continuation_id(sent_id)
        if next_id is None:
            continue
        if next_id in by_id:
            continuations.add(next_id)
        else:
            report.unmerged.append(sent_id)

    output = []
    for sentence in sentences:
        sent_id = sentence_comment(sentence, 'sent_id')
        if sent_id in continuations:
            continue
        chain = [sent_id]
        while ends_in_colon(sentence) and continuation_id(chain[-1]) in continuations:
            next_id = continuation_id(chain[-1])
            sentence = merge_pair(sentence, by_id[next_id])
            chain.append(next_id)
        if len(chain) > 1:
            report.merged.append(chain)
        output.append(sentence)
    return output, report


def merge_file(input_path, output_path):
    """
    Merge the colon-final sentences of input_path, writing the result to output_path

    Returns the MergeReport
    """
    with open(input_path, encoding='utf-8') as fin:
        sentences = read_sentences(fin)
    output, report = merge_sentences(sentences)
    with open(output_path, 'w', encoding='utf-8', newline='\n') as fout:
        for sentence in output:
            for line in sentence:
                fout.write(line)
                fout.write('\n')
            fout.write('\n')
    return report


def print_report(report, input_path, stream=None):
    """Print a summary of the report to stream, sys.stderr by default"""
    if stream is None:
        stream = sys.stderr
    removed = sum(len(chain) - 1 for chain in report.merged)
    print('%s: merged %d colon-final sentences into %d sentences' % (input_path, removed, len(report.merged)), file=stream)
    if report.unmerged:
        print('  %d colon-final sentences have no continuation in this file: %s' %
              (len(report.unmerged), ' '.join(report.unmerged)), file=stream)


def main(args=None):
    """Command line entry point.  args defaults to sys.argv[1:]"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='input CoNLL-U file')
    parser.add_argument('output', help='output CoNLL-U file')
    args = parser.parse_args(args)

    report = merge_file(args.input, args.output)
    print_report(report, args.input)


if __name__ == '__main__':
    main()
