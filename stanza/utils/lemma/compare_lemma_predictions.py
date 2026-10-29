"""
Compare the predictions of two lemmatizers on the same file, split by where each answer comes from.

The lemmatizer answers a word from its dictionary if it can, and
otherwise uses the seq2seq model.  Two models trained with the same
dictionary, such as with and without --secondary_rate, can only differ
on the words the dictionary does not cover.  This script splits the
words of the gold file into

  train    - the dictionary built from the training file covers it
  lexicon  - only the external dictionary (--secondary_file) covers it
  seq2seq  - neither dictionary covers it

and reports the accuracy of each model on each group, along with
every word where the two models disagree.

The prediction files are the --output_file of the lemmatizer, or the
.pred.conllu files written by run_lemma --save_output, for the same
gold file.

Usage:
  python3 -m stanza.utils.lemma.compare_lemma_predictions \\
      --gold data/lemma/en_combined.dev.gold.conllu \\
      --pred_a dict_only.dev.pred.conllu --pred_b rate_0.1.dev.pred.conllu \\
      --train data/lemma/en_combined.train.in.conllu \\
      --lexicon data/lemma/en_combined.dict.conllu \\
      --output compare.md
"""

import argparse
from collections import Counter, defaultdict
import sys

from stanza.models.lemma.trainer import Trainer

CATEGORIES = ("train", "lexicon", "seq2seq")

def read_words(filename):
    """
    Read (form, upos, lemma) for every word in a CoNLL-U file, skipping MWT ranges and empty nodes
    """
    words = []
    with open(filename, encoding='utf-8') as fin:
        for line in fin:
            if not line.strip() or line.startswith('#'):
                continue
            pieces = line.rstrip('\r\n').split('\t')
            if len(pieces) < 4 or not pieces[0].isdigit():
                continue
            words.append((pieces[1], pieces[3], pieces[2]))
    return words

def build_dictionary(triples, caseless=False):
    """
    Build the dictionary part of a lemmatizer, with the same lookup as the trained model
    """
    trainer = Trainer(args={'dict_only': True, 'caseless': caseless}, device='cpu')
    if caseless:
        triples = [(form.lower(), upos, lemma) for form, upos, lemma in triples]
    trainer.train_dict(triples)
    return trainer

def categorize(gold_words, train_words, lexicon_words, caseless=False):
    """
    Return the category of each gold word: train, lexicon, or seq2seq
    """
    pairs = [(form, upos) for form, upos, _ in gold_words]
    in_train = build_dictionary(train_words, caseless).skip_seq2seq(pairs)
    if lexicon_words:
        in_either = build_dictionary(train_words + lexicon_words, caseless).skip_seq2seq(pairs)
    else:
        in_either = in_train
    categories = []
    for train_hit, either_hit in zip(in_train, in_either):
        if train_hit:
            categories.append("train")
        elif either_hit:
            categories.append("lexicon")
        else:
            categories.append("seq2seq")
    return categories

def compare(gold_words, pred_a, pred_b, categories):
    """
    Count correct answers for each model in each category, and collect the words where the models disagree

    Words with a blank gold lemma are not scored, the same as the UD scorer.
    Returns (totals, differences) where totals[category] is a Counter of
    'words', 'a', 'b' and differences is a Counter of
    (category, form, upos, gold, lemma_a, lemma_b)
    """
    if not (len(gold_words) == len(pred_a) == len(pred_b)):
        raise ValueError("The gold file has %d words, but the predictions have %d and %d" % (len(gold_words), len(pred_a), len(pred_b)))
    totals = defaultdict(Counter)
    differences = Counter()
    for idx, ((form, upos, gold), (form_a, _, lemma_a), (form_b, _, lemma_b), category) in enumerate(zip(gold_words, pred_a, pred_b, categories)):
        if form != form_a or form != form_b:
            raise ValueError("Word %d is %s in the gold file, but %s and %s in the predictions" % (idx, form, form_a, form_b))
        if gold == '_':
            continue
        for key in (category, "all"):
            totals[key]['words'] += 1
            totals[key]['a'] += lemma_a == gold
            totals[key]['b'] += lemma_b == gold
        if lemma_a != lemma_b:
            differences[(category, form, upos, gold, lemma_a, lemma_b)] += 1
    return totals, differences

def markdown_row(*fields):
    return "| " + " | ".join(str(x) for x in fields) + " |"

def format_report(totals, differences, name_a="A", name_b="B"):
    lines = []
    lines.append(markdown_row("words", "count", name_a, name_b, "%s - %s" % (name_b, name_a)))
    lines.append(markdown_row("---", "---", "---", "---", "---"))
    for category in CATEGORIES + ("all",):
        counts = totals.get(category)
        if not counts or not counts['words']:
            continue
        acc_a = 100.0 * counts['a'] / counts['words']
        acc_b = 100.0 * counts['b'] / counts['words']
        lines.append(markdown_row(category, counts['words'], "%.2f" % acc_a, "%.2f" % acc_b, "%+d words" % (counts['b'] - counts['a'])))
    lines.append("")

    for category in CATEGORIES:
        rows = [(key, count) for key, count in differences.items() if key[0] == category]
        if not rows:
            continue
        rows.sort(key=lambda x: (-x[1], x[0][1].lower(), x[0][2]))
        better = sum(count for key, count in rows if key[5] == key[3])
        worse = sum(count for key, count in rows if key[4] == key[3])
        lines.append("## %s: %s better on %d, worse on %d" % (category, name_b, better, worse))
        lines.append("")
        lines.append(markdown_row("count", "form", "upos", "gold", name_a, name_b))
        lines.append(markdown_row("---", "---", "---", "---", "---", "---"))
        for (_, form, upos, gold, lemma_a, lemma_b), count in rows:
            lines.append(markdown_row(count, form, upos, gold, lemma_a, lemma_b))
        lines.append("")
    return "\n".join(lines)

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Compare the predictions of two lemmatizers on the same file")
    parser.add_argument('--gold', required=True, help='Gold CoNLL-U file')
    parser.add_argument('--pred_a', required=True, help='Predictions of the first model')
    parser.add_argument('--pred_b', required=True, help='Predictions of the second model')
    parser.add_argument('--name_a', default='A', help='Name for the first model in the report')
    parser.add_argument('--name_b', default='B', help='Name for the second model in the report')
    parser.add_argument('--train', required=True, nargs='+', help='Training file(s) the dictionary was built from')
    parser.add_argument('--lexicon', default=None, nargs='*', help='External dictionary file(s), such as the --secondary_file')
    parser.add_argument('--caseless', action='store_true', help='The models are caseless, so the dictionary lookups are lowercased')
    parser.add_argument('--output', default=None, help='Where to write the report.  Defaults to stdout')
    return parser.parse_args(args=args)

def main(args=None):
    args = parse_args(args)
    gold_words = read_words(args.gold)
    train_words = [word for filename in args.train for word in read_words(filename)]
    lexicon_words = [word for filename in (args.lexicon or []) for word in read_words(filename)]
    categories = categorize(gold_words, train_words, lexicon_words, args.caseless)
    totals, differences = compare(gold_words, read_words(args.pred_a), read_words(args.pred_b), categories)
    report = format_report(totals, differences, args.name_a, args.name_b)
    if args.output:
        with open(args.output, 'w', encoding='utf-8', newline='\n') as fout:
            fout.write(report)
            fout.write("\n")
    else:
        print(report)

if __name__ == '__main__':
    main()
