"""
Convert the Dutch QTLeap helpdesk answers (qtleap1a in the Alpino treebank) to UD

qtleap1a is 449 sentences of answers from an IT helpdesk, nearly half of
whose finite verbs are imperatives ("Ga naar Instellingen en kies ...").
The UD Dutch treebanks have very few imperatives, so this is useful both
as extra training data and as a test of how well a model tags them.

The trees are in the Alpino repository on GitHub, in Treebank/qtleap1a.
First convert them with alud, https://github.com/rug-compling/alud :

  cd alud/v2 && GOPROXY=direct go build ./cmd/alud
  ls Alpino/Treebank/qtleap1a/*.xml | alud -i '.*/(.*)\\.xml' > qtleap1a.alud.conllu

then

  python3 -m stanza.utils.datasets.dutch.convert_qtleap qtleap1a.alud.conllu nl_qtleap.conllu

The public trees do not have the parser attributes that the UD release of
Alpino is converted from, so compared with the official conversion:

  - Mood=Imp is never assigned by alud.  This script assigns it with
    relabel_mood, the same as for Alpino and LassySmall.  With the tree
    fixes below, its output on this text was checked by hand against every
    finite verb heading a main clause.  The only debatable cases are a
    conjunct sharing the subject je of an inverted indicative, "In het menu
    klik je op ... en probeer dan ...", which is left as Ind.
  - ExtPos is missing on fixed expressions.  In this text that affects
    about 20 tokens.

This script also:

  - fixes the trees in TREE_FIXES, which have a parse error in exactly the
    construction this data is for: the object of an imperative, or a
    possessive je, analyzed as the subject ("Reset de router",
    "Probeer je gegevens opnieuw in te voeren")
  - retags imperative "ga" from WW(pv,conj,ev) to WW(pv,tgw,ev).  The UD
    Dutch treebanks never tag "ga" as a subjunctive
  - puts the sentences in document order and adds a newdoc for each
    answer, so that the tokenizer sees each answer as one paragraph

Usage from other code: convert_file(input_path, output_path) or convert_sentences(sentences)
"""

import argparse
import re
import sys

from stanza.utils.datasets.dutch import relabel_mood
from stanza.utils.datasets.dutch.merge_colon_sentences import read_sentences, sentence_comment

SENT_ID_RE = re.compile(r'^(Batch\d+_answers_nl-(\d+))-(\d+)$')

XPOS, FEATS, HEAD, DEPREL, DEPS = 4, 5, 6, 7, 8
COLUMNS = {'XPOS': XPOS, 'FEATS': FEATS, 'HEAD': HEAD, 'DEPREL': DEPREL, 'DEPS': DEPS}

POSSESSIVE_JE = {
    'XPOS': 'VNW|bez|det|stan|red|2v|ev|prenom|zonder|agr',
    'FEATS': 'Person=2|Poss=Yes|PronType=Prs',
    'DEPREL': 'nmod:poss',
}

def possessive_je(noun):
    """Edits which turn a misparsed subject je into the possessive of token noun"""
    return dict(POSSESSIVE_JE, HEAD=str(noun), DEPS='%d:nmod:poss' % noun)

def object_of(verb):
    """Edits which turn a misparsed subject into the object of token verb"""
    return {'DEPREL': 'obj', 'DEPS': '%d:obj' % verb}

# sent_id -> list of (token id, form, edits)
TREE_FIXES = {
    'Batch1_answers_nl-127-1': [(2, 'je', possessive_je(3))],          # Probeer je gegevens opnieuw in te voeren.
    'Batch1_answers_nl-128-1': [(2, 'je', possessive_je(3))],          # Probeer je inloggegevens opnieuw in te voeren ...
    'Batch1_answers_nl-189-1': [(2, 'je', possessive_je(3))],          # Probeer je bureaublad instellingen te veranderen.
    'Batch1_answers_nl-25-1':  [(2, 'je', possessive_je(3))],          # Verbind je computer rechtstreeks met ...
    'Batch1_answers_nl-279-2': [(2, 'je', possessive_je(3))],          # Probeer je computer op te starten ...
    'Batch1_answers_nl-71-2':  [(8, 'je', possessive_je(9))],          # Als dit niet lukt, controleer dan je inloggegevens ...
    'Batch1_answers_nl-83-1':  [(3, 'router', object_of(1))],          # Reset de router, door ...
    'Batch1_answers_nl-280-1': [(3, 'PS3', object_of(1))],             # Start de PS3 op, ga naar ...
    'Batch1_answers_nl-247-2': [(10, 'computer', object_of(8))],       # Als dit niet het geval is, herstart de computer.
    # Om PDF bestanden te kunnen wijzigen, heb je PDF software nodig ...
    # je is attached as the subject of wijzigen rather than heb
    'Batch1_answers_nl-256-1': [(9, 'je', {'HEAD': '8', 'DEPS': '6:nsubj:xsubj|8:nsubj'})],
    # ... in het bereik van (de Ip-adressen van) de router liggen.
    # the closing bracket on Ip-adressen is non-projective over "van"
    'Batch1_answers_nl-215-1': [(19, ')', {'HEAD': '21', 'DEPS': '21:punct'})],
    # Probeer onnodige programma's te verwijderenen voer een schijfopruiming uit.
    # "voer" is tagged as the past tense of varen
    'Batch1_answers_nl-184-1': [(6, 'voer', {'XPOS': 'WW|pv|tgw|ev',
                                             'FEATS': 'Mood=Ind|Number=Sing|Tense=Pres|VerbForm=Fin'})],
}


def sort_key(sentence):
    sent_id = sentence_comment(sentence, 'sent_id')
    match = SENT_ID_RE.match(sent_id or '')
    if not match:
        raise ValueError('Unexpected sent_id %s: expected the qtleap1a file names, such as Batch1_answers_nl-1-1' % sent_id)
    return int(match.group(2)), int(match.group(3))


def fix_tree(sentence):
    """Apply the TREE_FIXES for this sentence, checking that each edited token has the expected form"""
    sent_id = sentence_comment(sentence, 'sent_id')
    fixes = {token_id: (form, edits) for token_id, form, edits in TREE_FIXES.get(sent_id, [])}
    used = set()
    fixed = []
    for line in sentence:
        columns = line.split('\t')
        if line.startswith('#') or not columns[0].isdigit():
            fixed.append(line)
            continue
        token_id = int(columns[0])
        if token_id in fixes:
            form, edits = fixes[token_id]
            if columns[1] != form:
                raise ValueError('TREE_FIXES expected %s at %s %d, found %s' % (form, sent_id, token_id, columns[1]))
            for column, value in edits.items():
                columns[COLUMNS[column]] = value
            used.add(token_id)
        elif columns[1].lower() == 'ga' and columns[XPOS] == 'WW|pv|conj|ev':
            columns[XPOS] = 'WW|pv|tgw|ev'
        fixed.append('\t'.join(columns))
    if used != set(fixes):
        raise ValueError('TREE_FIXES for %s did not apply to tokens %s' % (sent_id, sorted(set(fixes) - used)))
    return fixed


def convert_sentences(sentences):
    """
    Convert the alud output for qtleap1a, as a list of sentences of lines without newlines

    Returns the CoNLL-U lines of the converted document, with newlines, and
    the relabel_mood Report
    """
    found = {sentence_comment(s, 'sent_id') for s in sentences}
    missing = sorted(set(TREE_FIXES) - found)
    if missing:
        raise ValueError('Sentences in TREE_FIXES not found in the input: %s' % ' '.join(missing))

    sentences = sorted((fix_tree(s) for s in sentences), key=sort_key)
    lines = []
    previous_answer = None
    for sentence in sentences:
        answer = SENT_ID_RE.match(sentence_comment(sentence, 'sent_id')).group(1)
        comments = [line for line in sentence if line.startswith('#')]
        tokens = [line for line in sentence if not line.startswith('#')]
        if answer != previous_answer:
            lines.append('# newdoc id = %s\n' % answer)
            previous_answer = answer
        lines.extend(line + '\n' for line in comments)
        lines.extend(line + '\n' for line in tokens)
        lines.append('\n')
    return relabel_mood.relabel_lines(lines, check=False)


def convert_file(input_path, output_path):
    """Convert the alud output in input_path, writing the result to output_path.  Returns the relabel_mood Report"""
    with open(input_path, encoding='utf-8') as fin:
        sentences = read_sentences(fin)
    lines, report = convert_sentences(sentences)
    with open(output_path, 'w', encoding='utf-8', newline='\n') as fout:
        fout.writelines(lines)
    return report


def main(args=None):
    """Command line entry point.  args defaults to sys.argv[1:]"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='alud output for Alpino/Treebank/qtleap1a')
    parser.add_argument('output', help='output CoNLL-U file')
    args = parser.parse_args(args)

    report = convert_file(args.input, args.output)
    relabel_mood.print_report(report, args.input, args.output)
    if report.warnings:
        sys.exit(1)


if __name__ == '__main__':
    main()
