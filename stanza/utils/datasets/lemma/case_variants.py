"""
Build exact dictionary entries for the capitalized and uppercase forms of the lowercase words in a training file

A word at the start of a sentence or in a headline is capitalized, but
the lemmatizer's dictionary is case-sensitive.  If the dictionary has
the capitalized form at all, it is usually as a PROPN, such as Traders
or Model, and the dictionary answers the word with the PROPN lemma.
Otherwise the seq2seq model has to decide whether to keep the capital
letter.

This writes Capitalized and UPPERCASE entries with the lemma of the
lowercase word, such as Beer and BEER -> beer from the NOUN beer, or
Comía and COMÍA -> comer from the VERB comía.  The lemmatizer adds them
with --exact_dict_file, which only changes the lookup for that exact
form and UPOS, so Mercury as a PROPN is still Mercury.

Only the given UPOS tags are used: NOUN for English, and the open
classes for Spanish.  Single letters are skipped, since a letter used
as a noun keeps its case in UD English.  A form which already appears
with that UPOS in the training file is skipped as well.

Usage:
  python3 -m stanza.utils.datasets.lemma.case_variants data/lemma/en_combined.train.in.conllu data/lemma/en_combined.exact_dict.conllu
  python3 -m stanza.utils.datasets.lemma.case_variants data/lemma/es_combined.train.in.conllu data/lemma/es_combined.exact_dict.conllu --upos NOUN VERB AUX ADJ ADV
"""

import argparse
from collections import Counter, defaultdict

from stanza.models.common.doc import TEXT, UPOS, LEMMA
from stanza.utils.conll import CoNLL

ENGLISH_UPOS = ('NOUN',)
SPANISH_UPOS = ('NOUN', 'VERB', 'AUX', 'ADJ', 'ADV')

def case_variants(triples, upos_tags=ENGLISH_UPOS):
    """
    Return sorted (form, upos, lemma) for the Capitalized and UPPERCASE versions of each lowercase word in triples with one of upos_tags
    """
    lemmas = defaultdict(Counter)
    known = set()
    for form, upos, lemma in triples:
        if upos not in upos_tags or not lemma or lemma == '_':
            continue
        known.add((form, upos))
        if form.islower() and len(form) > 1:
            lemmas[(form, upos)][lemma] += 1

    entries = {}
    for (form, upos), counts in lemmas.items():
        lemma = counts.most_common(1)[0][0]
        for variant in (form.capitalize(), form.upper()):
            if variant != form and (variant, upos) not in known:
                entries[(variant, upos)] = lemma
    return [(form, upos, lemma) for (form, upos), lemma in sorted(entries.items())]

def write_conllu(entries, output_filename):
    """
    Write (form, upos, lemma) entries as single-word CoNLL-U sentences
    """
    with open(output_filename, 'w', encoding='utf-8', newline='\n') as fout:
        for form, upos, lemma in entries:
            fout.write(f"# sent_id = case_variants-{upos}-{form}\n")
            fout.write(f"# text = {form}\n")
            fout.write(f"1\t{form}\t{lemma}\t{upos}\t_\t_\t0\troot\t_\t_\n")
            fout.write("\n")

def build_exact_dictionary(train_file, output_filename, upos_tags=ENGLISH_UPOS):
    """
    Write the case variants of the words in train_file with one of upos_tags to output_filename
    """
    doc = CoNLL.conll2doc(input_file=train_file)
    entries = case_variants(doc.get([TEXT, UPOS, LEMMA]), upos_tags)
    write_conllu(entries, output_filename)
    print("Wrote %d %s case variants from %s to %s" % (len(entries), "/".join(upos_tags), train_file, output_filename))
    return entries

def build_english_dictionary(train_file, output_filename):
    return build_exact_dictionary(train_file, output_filename, ENGLISH_UPOS)

def build_spanish_dictionary(train_file, output_filename):
    return build_exact_dictionary(train_file, output_filename, SPANISH_UPOS)

def main(args=None):
    parser = argparse.ArgumentParser(description="Build exact dictionary entries for capitalized and uppercase words")
    parser.add_argument('train_file', help='CoNLL-U training file')
    parser.add_argument('output_file', help='Where to write the dictionary')
    parser.add_argument('--upos', nargs='+', default=list(ENGLISH_UPOS), help='Which UPOS tags to build variants for.  Default %(default)s')
    args = parser.parse_args(args=args)
    build_exact_dictionary(args.train_file, args.output_file, tuple(args.upos))

if __name__ == '__main__':
    main()
