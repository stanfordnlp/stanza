"""
Convert the SPECIALIST Lexicon inflection table to CoNLL-U rows for the English lemmatizer.

The SPECIALIST Lexicon is a large syntactic lexicon of general and
biomedical English, released annually by the US National Library of
Medicine:

  https://lhncbc.nlm.nih.gov/LSG/Projects/lexicon/current/web/index.html

The file used here is inflVars.data from the release's MISC directory,
for example

  https://data.lhncbc.nlm.nih.gov/public/lsg/lexicon/2025/release/LEX_DOC/MISC/inflVars.data

Each line has six |-separated fields:

  inflected form | category | inflection | EUI | uninflected form | citation form

  housewives|128|8|E0032099|housewife|housewife

category and inflection are bit masks.  The categories used here are
adj (1), adv (2), noun (128), and verb (1024).  The closed classes
(aux, modal, pron, det, prep, conj, compl) are skipped, as the
treebanks cover those words and the conventions differ from UD.
SPECIALIST has no category for proper nouns, so a noun whose lemma
has a capital letter, such as Alzheimer or DNA, is tagged PROPN.

The lemma is the uninflected form, not the citation form.  The
uninflected form keeps the spelling variant of the inflected form
(colours -> colour), whereas the citation form is the canonical
spelling shared by every variant of an entry, which would turn
house-wives into housewife.

Rows are skipped if the form or lemma has whitespace (multiword
terms), if the form is a negated contraction such as didn't, which
UD splits into two words, or if the form or lemma has symbols other
than - ' and . such as (1)O(2) or %LN.  A (form, UPOS) pair with more than one
lemma, such as axes -> axe / axis, is dropped entirely so that the
seq2seq model makes the decision instead of the dictionary.

The output has one single-word sentence per (form, UPOS, lemma),
the same format as the Sloleks dictionary for Slovenian.
prepare_lemma_treebank writes it to data/lemma/en_combined.dict.conllu
from extern_data/english/specialist/inflVars.data, and run_lemma
passes that file to the lemmatizer as its --secondary_file.

The SPECIALIST Lexicon is distributed under the terms and conditions of
the SPECIALIST NLP Tools:

  https://lhncbc.nlm.nih.gov/LSG/Docs/termsAndConditions.html

Those terms require that any redistribution of the data, which
includes a lemmatizer model whose dictionary was built from it,
attributes the source as the SPECIALIST NLP Tools with the release
number and date, includes the list of conditions with the
distribution, and describes any modifications.  This script is the
description of the modifications.

Usage:
  python3 -m stanza.utils.datasets.lemma.convert_en_specialist inflVars.data specialist_lemmas.conllu

Optionally, compare the result against UD training and dev data:
  python3 -m stanza.utils.datasets.lemma.convert_en_specialist inflVars.data specialist_lemmas.conllu \\
      --train_files en_ewt-ud-train.conllu en_gum-ud-train.conllu \\
      --dev_files en_ewt-ud-dev.conllu en_gum-ud-dev.conllu \\
      --report specialist_report.md
"""

import argparse
from collections import Counter, defaultdict
import logging
import os
import sys

logger = logging.getLogger('stanza')

# SPECIALIST category bit -> UPOS
CATEGORY_MAP = {
    1:    'ADJ',
    2:    'ADV',
    128:  'NOUN',
    1024: 'VERB',
}

# apostrophes which appear in negated contractions such as didn't
NEGATION_SUFFIXES = ("n't", "n’t")

# punctuation which can appear inside a dictionary word, such as
# mother-in-law, o'clock, or Ph.D.  Anything else outside of letters
# and digits makes the row look like notation rather than a word
WORD_PUNCTUATION = set("-'’.")


def is_word(text):
    """
    Check that text is made of letters, digits, and the punctuation found inside ordinary words

    Letters and digits in any script are allowed, so IL-2 and β-blocker
    pass, while rows such as (1)O(2) or %LN do not.  At least one letter
    is required.
    """
    return (all(x.isalnum() or x in WORD_PUNCTUATION for x in text) and
            any(x.isalpha() for x in text))

def read_inflection_table(filename, skipped=None):
    """
    Read inflVars.data, yielding (form, upos, lemma) for each usable row.

    Rows outside the open class categories, multiword rows, negated
    contractions, and rows with symbols in the form or lemma are skipped.
    Duplicates are not removed here.

    If skipped is a Counter, it is updated with the number of rows
    skipped for each of those reasons.
    """
    if skipped is None:
        skipped = Counter()
    with open(filename, encoding='utf-8') as fin:
        for line_idx, line in enumerate(fin):
            line = line.rstrip('\r\n')
            if not line:
                continue
            pieces = line.split('|')
            if len(pieces) < 6:
                raise ValueError("Line %d of %s has %d fields, expected 6: %s" % (line_idx + 1, filename, len(pieces), line))
            form, category, _inflection, _eui, lemma, _citation = pieces[:6]
            upos = CATEGORY_MAP.get(int(category))
            if upos is None:
                skipped['category'] += 1
                continue
            if not form or not lemma:
                skipped['blank'] += 1
                continue
            if any(x.isspace() for x in form) or any(x.isspace() for x in lemma):
                skipped['multiword'] += 1
                continue
            if form.lower().endswith(NEGATION_SUFFIXES):
                skipped['negation'] += 1
                continue
            if not is_word(form) or not is_word(lemma):
                skipped['symbols'] += 1
                continue
            # SPECIALIST has no separate category for proper nouns, so
            # names (Alzheimer, Kaposi, Monday) are nouns.  In UD they
            # are PROPN, and keeping them as NOUN teaches the seq2seq
            # model to keep the capital letters of common nouns
            if upos == 'NOUN' and lemma != lemma.lower():
                upos = 'PROPN'
            yield form, upos, lemma


def build_lemma_dict(triples):
    """
    Group (form, upos, lemma) triples into {(form, upos): lemma}.

    Returns the unambiguous dictionary and a dict of the ambiguous
    (form, upos) pairs mapped to the sorted list of their lemmas.
    """
    lemmas = defaultdict(set)
    for form, upos, lemma in triples:
        lemmas[(form, upos)].add(lemma)

    unambiguous = {}
    ambiguous = {}
    for key, values in lemmas.items():
        if len(values) == 1:
            unambiguous[key] = next(iter(values))
        else:
            ambiguous[key] = sorted(values)
    return unambiguous, ambiguous


def sent_id(form, upos):
    """
    A sent_id for a dictionary entry, unique since each (form, upos) has one lemma

    UPOS goes before the form, since UPOS never has a - in it but the form might
    """
    return f"en_specialist-{upos}-{form}"

def write_conllu(lemma_dict, output_filename):
    """
    Write {(form, upos): lemma} as single-word CoNLL-U sentences, sorted by form then UPOS.
    """
    with open(output_filename, 'w', encoding='utf-8', newline='\n') as fout:
        for (form, upos) in sorted(lemma_dict.keys()):
            lemma = lemma_dict[(form, upos)]
            fout.write(f"# sent_id = {sent_id(form, upos)}\n")
            fout.write(f"# text = {form}\n")
            fout.write(f"1\t{form}\t{lemma}\t{upos}\t_\t_\t0\troot\t_\t_\n")
            fout.write("\n")


def convert_file(input_filename, output_filename):
    """
    Convert inflVars.data to a CoNLL-U dictionary file.

    Returns the unambiguous {(form, upos): lemma} dict and the ambiguous pairs.
    """
    skipped = Counter()
    lemma_dict, ambiguous = build_lemma_dict(read_inflection_table(input_filename, skipped))
    write_conllu(lemma_dict, output_filename)
    logger.info("Wrote %d (form, upos, lemma) entries to %s, skipping %d ambiguous (form, upos) pairs",
                len(lemma_dict), output_filename, len(ambiguous))
    logger.info("Rows skipped when reading %s: %s", input_filename,
                ", ".join("%s %d" % (reason, count) for reason, count in sorted(skipped.items())))
    upos_counts = Counter(upos for _, upos in lemma_dict)
    logger.info("Entries by UPOS: %s", ", ".join("%s %d" % (upos, count) for upos, count in sorted(upos_counts.items())))
    return lemma_dict, ambiguous


def build_dictionary(paths, output_filename):
    """
    Convert inflVars.data from STANZA_EXTERN_DIR/english/specialist to a CoNLL-U dictionary at output_filename
    """
    infl_vars = os.path.join(paths["STANZA_EXTERN_DIR"], "english", "specialist", "inflVars.data")
    if not os.path.exists(infl_vars):
        raise FileNotFoundError("Could not find the SPECIALIST Lexicon inflection table at %s ... please download inflVars.data from the MISC directory of the current release at https://lhncbc.nlm.nih.gov/LSG/Projects/lexicon/current/web/release/index.html or use --no_external_dictionary to prepare the dataset without it" % infl_vars)
    convert_file(infl_vars, output_filename)


def read_conllu_triples(filenames):
    """
    Read (form, upos, lemma) for every word in the given CoNLL-U files.

    MWT ranges and empty nodes are skipped.
    """
    triples = []
    for filename in filenames:
        with open(filename, encoding='utf-8') as fin:
            for line in fin:
                if not line.strip() or line.startswith('#'):
                    continue
                pieces = line.rstrip('\r\n').split('\t')
                if len(pieces) < 4 or not pieces[0].isdigit():
                    continue
                triples.append((pieces[1], pieces[3], pieces[2]))
    return triples


def build_dict_trainer(triples):
    """
    Build a dict-only lemmatizer Trainer from (form, upos, lemma) triples.

    This uses the lemmatizer's own train_dict, so the lookups in the
    report match what a model trained with --secondary_file would do.
    """
    from stanza.models.lemma.trainer import Trainer
    trainer = Trainer(args={'dict_only': True}, device='cpu')
    trainer.train_dict(triples)
    return trainer


def train_disagreements(lemma_dict, train_triples):
    """
    Find (form, upos) pairs where the SPECIALIST lemma differs from the most common training lemma.

    Returns a list of (count, form, upos, train_lemma, specialist_lemma), most frequent first.
    """
    counts = Counter(train_triples)
    best = {}
    for (form, upos, lemma), count in counts.most_common():
        key = (form, upos)
        if key not in best:
            best[key] = (lemma, count)

    disagreements = []
    for key, (train_lemma, count) in best.items():
        specialist_lemma = lemma_dict.get(key)
        if specialist_lemma is not None and specialist_lemma != train_lemma:
            disagreements.append((count, key[0], key[1], train_lemma, specialist_lemma))
    disagreements.sort(key=lambda x: (-x[0], x[1], x[2]))
    return disagreements


def dev_comparison(lemma_dict, train_triples, dev_triples):
    """
    Compare dictionary lookups on the dev data with and without the SPECIALIST entries.

    Returns a summary dict and a list of the dev words whose lookup changed:
      (form, upos, gold, before, after)
    where before is None if the training dictionary had no entry for the word.
    """
    specialist_triples = [(form, upos, lemma) for (form, upos), lemma in lemma_dict.items()]
    before_trainer = build_dict_trainer(train_triples)
    after_trainer = build_dict_trainer(train_triples + specialist_triples)

    pairs = [(form, upos) for form, upos, _ in dev_triples]
    before_hit = before_trainer.skip_seq2seq(pairs)
    after_hit = after_trainer.skip_seq2seq(pairs)
    before_preds = before_trainer.predict_dict(pairs)
    after_preds = after_trainer.predict_dict(pairs)

    summary = Counter()
    changed = []
    for (form, upos, gold), b_hit, a_hit, b_pred, a_pred in zip(dev_triples, before_hit, after_hit, before_preds, after_preds):
        summary['words'] += 1
        if not b_hit and a_hit:
            summary['newly_covered'] += 1
            if a_pred == gold:
                summary['newly_covered_correct'] += 1
            changed.append((form, upos, gold, None, a_pred))
        elif b_hit and a_hit and b_pred != a_pred:
            summary['changed'] += 1
            if b_pred == gold:
                summary['changed_was_correct'] += 1
            if a_pred == gold:
                summary['changed_now_correct'] += 1
            changed.append((form, upos, gold, b_pred, a_pred))
    return summary, changed


def markdown_row(*fields):
    return "| " + " | ".join(str(x) for x in fields) + " |"


def write_report(report_filename, lemma_dict, ambiguous, train_triples=None, dev_triples=None):
    """
    Write a markdown report of how the SPECIALIST dictionary compares to the UD data.
    """
    lines = []
    lines.append("# SPECIALIST lemma dictionary report")
    lines.append("")
    upos_counts = Counter(upos for _, upos in lemma_dict.keys())
    lines.append("Entries: %d (%s)" % (len(lemma_dict), ", ".join("%s %d" % (upos, upos_counts[upos]) for upos in sorted(upos_counts))))
    lines.append("")
    lines.append("Ambiguous (form, upos) pairs dropped: %d" % len(ambiguous))
    lines.append("")

    if train_triples:
        disagreements = train_disagreements(lemma_dict, train_triples)
        lines.append("## Disagreements with the training data")
        lines.append("")
        lines.append("Training lemmas take precedence over these in the dictionary,")
        lines.append("but they show where SPECIALIST conventions differ from UD.")
        lines.append("")
        lines.append(markdown_row("count", "form", "upos", "train lemma", "SPECIALIST lemma"))
        lines.append(markdown_row("---", "---", "---", "---", "---"))
        for count, form, upos, train_lemma, specialist_lemma in disagreements:
            lines.append(markdown_row(count, form, upos, train_lemma, specialist_lemma))
        lines.append("")

    if train_triples and dev_triples:
        summary, changed = dev_comparison(lemma_dict, train_triples, dev_triples)
        lines.append("## Effect on dev dictionary lookups")
        lines.append("")
        lines.append("Dev words: %d" % summary['words'])
        lines.append("")
        lines.append("Newly covered by the dictionary: %d, of which %d match the gold lemma" %
                     (summary['newly_covered'], summary['newly_covered_correct']))
        lines.append("")
        lines.append("Dictionary answer changed: %d, previously correct %d, now correct %d" %
                     (summary['changed'], summary['changed_was_correct'], summary['changed_now_correct']))
        lines.append("")
        lines.append("Newly covered words go through the seq2seq model without the")
        lines.append("dictionary, so the effect on the full lemmatizer depends on how")
        lines.append("many of them the seq2seq model already gets right.")
        lines.append("")
        wrong = [x for x in changed if x[4] != x[2]]
        lines.append("### Dev words where the new dictionary answer is wrong")
        lines.append("")
        lines.append(markdown_row("form", "upos", "gold", "before", "after"))
        lines.append(markdown_row("---", "---", "---", "---", "---"))
        for form, upos, gold, before, after in sorted(wrong, key=lambda x: (x[0], x[1])):
            lines.append(markdown_row(form, upos, gold, before if before is not None else "-", after))
        lines.append("")

    with open(report_filename, 'w', encoding='utf-8', newline='\n') as fout:
        fout.write("\n".join(lines))
        fout.write("\n")


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Convert the SPECIALIST Lexicon inflVars.data to a CoNLL-U lemma dictionary")
    parser.add_argument('input_file', help='inflVars.data from the SPECIALIST Lexicon release')
    parser.add_argument('output_file', help='CoNLL-U file to write')
    parser.add_argument('--train_files', nargs='*', default=None, help='UD training files to compare against')
    parser.add_argument('--dev_files', nargs='*', default=None, help='UD dev files for estimating the effect on dictionary lookups.  Requires --train_files')
    parser.add_argument('--report', default=None, help='Where to write a markdown report comparing the dictionary to the UD files')
    args = parser.parse_args(args=args)
    if args.dev_files and not args.train_files:
        parser.error("--dev_files requires --train_files")
    if (args.train_files or args.dev_files) and not args.report:
        parser.error("--train_files and --dev_files are only used with --report")
    return args


def main(args=None):
    args = parse_args(args)
    lemma_dict, ambiguous = convert_file(args.input_file, args.output_file)
    print("Wrote %d entries to %s, skipping %d ambiguous (form, upos) pairs" % (len(lemma_dict), args.output_file, len(ambiguous)), file=sys.stderr)
    if args.report:
        train_triples = read_conllu_triples(args.train_files) if args.train_files else None
        dev_triples = read_conllu_triples(args.dev_files) if args.dev_files else None
        write_report(args.report, lemma_dict, ambiguous, train_triples, dev_triples)
        print("Wrote report to %s" % args.report, file=sys.stderr)


if __name__ == '__main__':
    main()
