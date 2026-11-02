"""
Build exact dictionary entries for the capitalized and uppercase forms of the lowercase NOUNs in a training file

A NOUN at the start of a sentence or in a headline is capitalized,
but the lemmatizer's dictionary is case-sensitive.  If the dictionary
has the capitalized form at all, it is usually as a PROPN, such as
Traders or Model, and the dictionary answers the NOUN with the PROPN
lemma.  Otherwise the seq2seq model has to decide whether to keep the
capital letter.

This writes Capitalized and UPPERCASE NOUN entries with the lemma of
the lowercase NOUN, such as Beer and BEER -> beer from beer.  The
lemmatizer adds them with --exact_dict_file, which only changes the
lookup for that exact form as a NOUN, so Mercury as a PROPN is still
Mercury.

Single letters are skipped, since a letter used as a noun keeps its
case in UD English.  Forms which already appear as a
NOUN in the training file are skipped as well.

Usage:
  python3 -m stanza.utils.datasets.lemma.noun_case_variants data/lemma/en_combined.train.in.conllu data/lemma/en_combined.exact_dict.conllu
"""

import argparse
from collections import Counter, defaultdict

from stanza.models.common.doc import TEXT, UPOS, LEMMA
from stanza.utils.conll import CoNLL

def noun_case_variants(triples):
    """
    Return sorted (form, 'NOUN', lemma) for the Capitalized and UPPERCASE versions of each lowercase NOUN in triples
    """
    lemmas = defaultdict(Counter)
    known = set()
    for form, upos, lemma in triples:
        if upos != 'NOUN' or not lemma or lemma == '_':
            continue
        known.add(form)
        if form.islower() and len(form) > 1:
            lemmas[form][lemma] += 1

    entries = {}
    for form, counts in lemmas.items():
        lemma = counts.most_common(1)[0][0]
        for variant in (form.capitalize(), form.upper()):
            if variant != form and variant not in known:
                entries[variant] = lemma
    return [(form, 'NOUN', lemma) for form, lemma in sorted(entries.items())]

def write_conllu(entries, output_filename):
    """
    Write (form, upos, lemma) entries as single-word CoNLL-U sentences
    """
    with open(output_filename, 'w', encoding='utf-8', newline='\n') as fout:
        for form, upos, lemma in entries:
            fout.write(f"# sent_id = noun_case-{upos}-{form}\n")
            fout.write(f"# text = {form}\n")
            fout.write(f"1\t{form}\t{lemma}\t{upos}\t_\t_\t0\troot\t_\t_\n")
            fout.write("\n")

def build_exact_dictionary(train_file, output_filename):
    """
    Write the NOUN case variants of train_file to output_filename
    """
    doc = CoNLL.conll2doc(input_file=train_file)
    entries = noun_case_variants(doc.get([TEXT, UPOS, LEMMA]))
    write_conllu(entries, output_filename)
    print("Wrote %d NOUN case variants from %s to %s" % (len(entries), train_file, output_filename))
    return entries

def main(args=None):
    parser = argparse.ArgumentParser(description="Build exact dictionary entries for capitalized and uppercase NOUNs")
    parser.add_argument('train_file', help='CoNLL-U training file')
    parser.add_argument('output_file', help='Where to write the dictionary')
    args = parser.parse_args(args=args)
    build_exact_dictionary(args.train_file, args.output_file)

if __name__ == '__main__':
    main()
