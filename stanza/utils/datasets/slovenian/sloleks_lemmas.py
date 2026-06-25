#!/usr/bin/env python3
"""
Convert Sloleks 3.x XML lexicon files to CoNLL-U rows.
Usage: python3 sloleks_to_conllu.py file1.xml [file2.xml ...] > output.conllu
       python3 sloleks_to_conllu.py *.xml > sloleks.conllu

Sloleks 3.1 is available at: https://www.clarin.si/repository/xmlui/handle/11356/2080

This data goes into the Slovenian combined lemmatizer, improving the coverage of the dictionary
"""

import glob
import os
import sys
import re
from collections import defaultdict

# JOS category -> UPOS mapping (skipping conjunction, abbreviation, residual)
CATEGORY_MAP = {
    'noun':         'NOUN',
    'verb':         'VERB',
    'adjective':    'ADJ',
    'adverb':       'ADV',
    'pronoun':      'PRON',
    'numeral':      'NUM',
    'preposition':  'ADP',
    'particle':     'PART',
    'interjection': 'INTJ',
}

SKIP_CATEGORIES = {'conjunction', 'abbreviation', 'residual'}


def parse_file(path):
    """Stream-parse one Sloleks XML file, yielding (form, upos, lemma) triples."""
    with open(path, encoding='utf-8') as f:
        content = f.read()

    # Split into entries
    entries = re.split(r'<entry>', content)[1:]  # skip preamble before first entry

    for entry_text in entries:
        # Extract lemma
        lemma_m = re.search(r'<headword>\s*<lemma>([^<]+)</lemma>', entry_text)
        if not lemma_m:
            continue
        lemma = lemma_m.group(1).strip()

        # Extract category
        cat_m = re.search(r'<category>([^<]+)</category>', entry_text)
        if not cat_m:
            continue
        category = cat_m.group(1).strip()

        if category in SKIP_CATEGORIES:
            continue
        upos = CATEGORY_MAP.get(category)
        if upos is None:
            continue

        # Extract all orthographic forms (not accentuation, not pronunciation)
        # Each <orthography> block contains one <form>
        # We want unique forms only (some paradigm slots share surface forms)
        seen_forms = set()
        for form_m in re.finditer(
            r'<orthography[^>]*>.*?<form>([^<]+)</form>',
            entry_text,
            re.DOTALL
        ):
            form = form_m.group(1).strip()
            if form and form not in seen_forms:
                seen_forms.add(form)
                yield (form, upos, lemma)


def sloleks_to_dict(filenames):
    """
    Convert a list of Sloleks XML filenames to CoNLL-U lines.

    Returns a list of strings, each being one line of CoNLL-U output
    (including blank lines that separate sentences). Deduplicates across
    files on (form.lower(), upos, lemma).
    """
    lines = []
    seen = set()

    for path in filenames:
        for form, upos, lemma in parse_file(path):
            key = (form.lower(), upos, lemma)
            if key in seen:
                continue
            seen.add(key)

            yield(form, upos, lemma)

    return lines

def write_sloleks_dict_file(filenames, output_filename):
    with open(output_filename, 'w', encoding='utf-8') as fout:
        for form, upos, lemma in sloleks_to_dict(filenames):
            fout.write(f"# text = {form}\n")
            fout.write(f"1\t{form}\t{lemma}\t{upos}\t_\t_\t0\troot\t_\t_\n")
            fout.write("\n")

def convert_directory(input_directory, output_filename):
    input_filenames = glob.glob(os.path.join(input_directory, "sloleks_3.1_???.xml"))
    write_sloleks_dict_file(input_filenames, output_filename)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 sloleks_to_conllu.py file1.xml [file2.xml ...]",
              file=sys.stderr)
        sys.exit(1)

    n_sentences = 0
    for form, upos, lemma in sloleks_to_dict(sys.argv[1:]):
        # Minimal CoNLL-U sentence (one token per sentence)
        # Columns: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
        n_sentences += 1
        print(f"# text = {form}")
        print(f"1\t{form}\t{lemma}\t{upos}\t_\t_\t0\troot\t_\t_")
        print()

    print(f"# wrote {n_sentences} unique (form, upos, lemma) rows", file=sys.stderr)


if __name__ == '__main__':
    main()
