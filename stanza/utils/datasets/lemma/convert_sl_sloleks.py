#!/usr/bin/env python3
"""
Convert Sloleks 3.x XML lexicon files to CoNLL-U rows.
Usage: python3 -m stanza.utils.datasets.lemma.convert_sl_sloleks file1.xml [file2.xml ...] > output.conllu
       python3 -m stanza.utils.datasets.lemma.convert_sl_sloleks *.xml > sloleks.conllu

Sloleks 3.1 is available at: https://www.clarin.si/repository/xmlui/handle/11356/2080

This data goes into the Slovenian combined lemmatizer, improving the coverage of the dictionary.
prepare_lemma_treebank writes it to data/lemma/sl_combined.dict.conllu
from extern_data/slovenian/Sloleks.3.1, and run_lemma passes that file
to the lemmatizer as its --secondary_file.
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

def sent_ids():
    """
    Build sent_ids for (form, upos, lemma) entries, numbering any repeats

    The id has the form and lemma in it, so the same entry gets the same
    id each time the dictionary is built.  A form can have more than one
    lemma, so the lemma is part of the id.  Whitespace becomes _, and the
    rare id which would otherwise repeat (for example, from hyphens in
    the form and lemma) gets a number on the end.
    """
    seen = set()
    def build(form, upos, lemma):
        base = "-".join(["sl_sloleks", upos, "_".join(form.split()), "_".join(lemma.split())])
        new_id = base
        idx = 2
        while new_id in seen:
            new_id = "%s-%d" % (base, idx)
            idx += 1
        seen.add(new_id)
        return new_id
    return build

def write_sloleks_dict_file(filenames, output_filename):
    build_id = sent_ids()
    with open(output_filename, 'w', encoding='utf-8', newline='\n') as fout:
        for form, upos, lemma in sloleks_to_dict(filenames):
            fout.write(f"# sent_id = {build_id(form, upos, lemma)}\n")
            fout.write(f"# text = {form}\n")
            fout.write(f"1\t{form}\t{lemma}\t{upos}\t_\t_\t0\troot\t_\t_\n")
            fout.write("\n")

def convert_directory(input_directory, output_filename):
    # sorted so that the output, and the numbering of any repeated ids, is the same on each run
    input_filenames = sorted(glob.glob(os.path.join(input_directory, "sloleks_3.1_???.xml")))
    write_sloleks_dict_file(input_filenames, output_filename)

def build_dictionary(paths, output_filename):
    """
    Convert the Sloleks 3.1 download in STANZA_EXTERN_DIR to a CoNLL-U dictionary at output_filename
    """
    sloleks_dir = os.path.join(paths["STANZA_EXTERN_DIR"], "slovenian", "Sloleks.3.1")
    if not os.path.exists(sloleks_dir):
        raise FileNotFoundError("Could not find the Sloleks 3.1 dataset at %s ... please download it from https://www.clarin.si/repository/xmlui/handle/11356/2080 or use --no_external_dictionary to prepare the dataset without it" % sloleks_dir)
    convert_directory(sloleks_dir, output_filename)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m stanza.utils.datasets.lemma.convert_sl_sloleks file1.xml [file2.xml ...]",
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
