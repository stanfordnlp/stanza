"""
Convert a German Wiktionary dump to CoNLL-U rows for the German lemmatizer.

The dump is the pages-articles file from

  https://dumps.wikimedia.org/dewiktionary/latest/dewiktionary-latest-pages-articles-multistream.xml.bz2

It is parsed with wiktionary-de-parser, which changes its output
format between versions, so this script checks for the exact version
it was written against:

  pip install wiktionary-de-parser==0.16.0

or, from a Stanza checkout, pip install -e .[wiktionary]

That version requires Python 3.13 or later.  Only this script needs
it, so the library is imported when the script runs rather than when
the module is imported, and the rest of Stanza is unaffected.

German nouns, verbs, adjectives, and adverbs are kept, with names
(towns, given names, surnames, and so on) tagged PROPN:

- An entry's inflection table gives the forms of its lemma.  The lemma
  itself is included, since verb tables do not list the infinitive.
  Only the fields for cases, adjective degrees, and verb forms are
  used.  The others, such as the gender or the auxiliary verb, are
  counted in the log.
- An inflected verb entry such as "liebst" (a {{Grundformverweis}}
  page) maps that form to its lemma.  Inflected noun and adjective
  entries are skipped, since their forms are in the tables of the
  lemmas, and the comparative and superlative pages point to the
  comparative rather than the base adjective.
- A noun with adjectival declension, such as "Abgeordneter", has only
  its stem in the table.  The stem ("Abgeordnete") is the lemma, as in
  the German GSD treebank, and the forms are the stem with the endings
  -n, -m, and -r (-s for neuter nouns), plus the page name.
- An entry for a variant spelling, such as "Geografie" for
  "Geographie" or the old spelling "Geschoß" for "Geschoss", is its own
  lemma, keeping the spelling of the text.
- Entries and forms with whitespace, and forms with leftover wiki
  markup, are skipped.  Entries with more than one POS are skipped.
- Every form with ß also gets a version with ss, for Swiss spelling.
- A (form, POS) with more than one lemma is dropped entirely, so
  that the seq2seq model makes the decision instead of the dictionary.
  The exception is a form shared by a variant spelling and the lemma
  it points to, such as "Geschosse" in the tables of both "Geschoß" and
  "Geschoss", which goes to the lemma the variant points to.
  --ambiguous_file writes out the forms which were dropped.

The output has one single-word sentence per (form, UPOS, lemma).
It is used as part of the de_combined lemma training data, from

  https://github.com/stanfordnlp/handparsed-treebank/tree/master/german-lemmas-wiki

German Wiktionary is licensed under CC BY-SA 4.0.

Usage:
  python3 -m stanza.utils.datasets.lemma.convert_de_wiktionary dewiktionary-latest-pages-articles-multistream.xml.bz2 de_wiki_lemmas.conllu
"""

import argparse
from collections import Counter, defaultdict
import logging
import os
import sys

logger = logging.getLogger('stanza')

PARSER_PACKAGE = "wiktionary-de-parser"
PARSER_VERSION = "0.16.0"

DUMP_FILENAME = "dewiktionary-latest-pages-articles-multistream.xml.bz2"

# German Wiktionary POS -> UPOS
POS_MAP = {
    "Adjektiv":   "ADJ",
    "Adverb":     "ADV",
    "Substantiv": "NOUN",
    "Verb":       "VERB",
}

# The parser files names under Substantiv, with one of these subtypes.
# They are PROPN, so the town Aalen does not compete with Aalen as a
# form of the noun Aal
NAME_SUBTYPES = {"Toponym", "Vorname", "Nachname", "Familienname", "Patronym",
                 "Eigenname", "Straßenname", "Bauwerksname", "Göttername"}

# The first word of the inflection table fields which hold forms of
# the word.  The parser translates the German field names to English
# one word at a time, from its own list of grammar terms:
#   Nominativ Singular -> nominative_singular
#   Präsens_er, sie, es -> present_3sg
# A word it does not recognize, including one with something attached
# such as the * in Hilfsverb* or Genitiv Singular*, is kept in German,
# so the German names are here as well.  Any other field, such as
# gender (Genus), auxiliary (Hilfsverb), or no_plural (kein Plural),
# is not used, other than the stem (Stamm) of a noun with adjectival
# declension.
FORM_FIELDS = {
    # cases of nouns and adjectives
    "nominative", "genitive", "dative", "accusative",
    "nominativ", "genitiv", "dativ", "akkusativ",
    # adjective degrees
    "positive", "comparative", "superlative",
    "positiv", "komparativ", "superlativ",
    # verb forms
    "present", "preterite", "subjunctive", "imperative", "participle", "infinitive",
    "präsens", "präteritum", "konjunktiv", "imperativ", "partizip", "infinitiv",
}

# The stem field of a noun with adjectival declension, used to build its forms
STEM_FIELDS = {"stem", "stamm"}

# Characters which mean a form still has wiki markup in it
MARKUP_CHARACTERS = set("[]{}<>|=")

# Values from fields such as the gender or "no plural".  If one turns
# up as a form, a field was sorted into the wrong group
LEAKED_FIELD_VALUES = {"f", "m", "n", "ja", "1"}


def check_parser_version(allow_other_version=False):
    """
    Check that the installed wiktionary-de-parser is the version this script was written for

    A different version raises an error, unless allow_other_version is
    set, in which case it logs a warning instead.
    """
    from importlib.metadata import version, PackageNotFoundError
    try:
        installed = version(PARSER_PACKAGE)
    except PackageNotFoundError:
        raise ImportError("This script needs %s %s, which requires Python 3.13+.  Please install it with\n  pip install %s==%s" %
                          (PARSER_PACKAGE, PARSER_VERSION, PARSER_PACKAGE, PARSER_VERSION)) from None
    if installed != PARSER_VERSION:
        message = ("This script was written for %s %s, but %s is installed.  Its output format changes between versions, so the results may be wrong.  Please install it with\n  pip install %s==%s" %
                   (PARSER_PACKAGE, PARSER_VERSION, installed, PARSER_PACKAGE, PARSER_VERSION))
        if not allow_other_version:
            raise ImportError(message + "\nor use --allow_other_parser_version to run anyway")
        logger.warning(message)


def parse_dump(dump_filename, workers=None):
    """
    Yield the parsed entries of every page in the dump

    workers is the number of processes to parse with, defaulting to all but one of the CPUs
    """
    from wiktionary_de_parser import WiktionaryDump
    dump = WiktionaryDump(dump_file_path=dump_filename)
    yield from dump.iter_parsed(workers=workers)


def is_reference(entry, reference_type):
    return entry.reference is not None and entry.reference.type == reference_type


def field_type(key):
    """
    Return the first word of an inflection table field, without any * or digits attached

    hilfsverb* -> hilfsverb, genitive_singular* -> genitive
    """
    return "".join(x for x in key.split("_")[0] if x.isalpha())


def table_forms(entry, unused_fields=None):
    """
    Return the forms in an entry's inflection table

    If unused_fields is a Counter, it is updated with the field types which are not forms
    """
    if not entry.inflection:
        return set()
    forms = set()
    for key, value in entry.inflection.items():
        key_type = field_type(key)
        if key_type in FORM_FIELDS:
            forms.add(value)
        elif key_type in STEM_FIELDS:
            # used by adjectival_forms
            continue
        elif unused_fields is not None:
            unused_fields[key_type] += 1
    return forms


def adjectival_forms(entry):
    """
    Return (lemma, forms) for a noun with adjectival declension, or None for any other entry

    The table of such a noun only has its stem, which is the weak form
    (der Abgeordnete), so the forms come from adding the endings to it
    """
    if not entry.inflection:
        return None
    stems = [value for key, value in entry.inflection.items() if field_type(key) in STEM_FIELDS]
    if not stems:
        return None
    stem = stems[0]
    gender = entry.inflection.get("gender")
    strong_ending = "s" if gender == "n" else "r"
    forms = {stem, stem + "n", stem + "m", stem + strong_ending, entry.page_name}
    return stem, forms


def entry_forms(entry, skipped, unused_fields=None):
    """
    Return (lemma, upos, forms, canonical) for one parsed entry, or None if the entry is not used

    canonical is the lemma a variant spelling points to, or None for an
    entry which is not a variant.

    skipped is a Counter of the reasons entries are not used, and
    unused_fields is a Counter of the inflection table fields which are not forms
    """
    if entry.language_code != 'de':
        skipped['language'] += 1
        return None
    if len(entry.pos) == 0:
        skipped['no pos'] += 1
        return None
    if len(entry.pos) > 1:
        skipped['multiple pos'] += 1
        return None
    upos = POS_MAP.get(entry.pos[0].pos)
    if upos is None:
        skipped['pos'] += 1
        return None
    if upos == 'NOUN' and NAME_SUBTYPES.intersection(entry.pos[0].subtypes):
        upos = 'PROPN'

    if is_reference(entry, "inflected"):
        if upos != 'VERB':
            skipped['inflected non-verb'] += 1
            return None
        lemma = entry.lemma
        forms = {entry.page_name}
        canonical = None
    else:
        # a variant spelling is its own lemma, so the lemma keeps the spelling of the text
        if is_reference(entry, "variant"):
            lemma = entry.page_name
            canonical = entry.lemma
        else:
            lemma = entry.lemma
            canonical = None
        forms = table_forms(entry, unused_fields)
        adjectival = adjectival_forms(entry) if upos == 'NOUN' else None
        if adjectival is not None:
            lemma, stem_forms = adjectival
            forms.update(stem_forms)
        forms.add(lemma)

    if any(x.isspace() for x in lemma):
        skipped['multiword'] += 1
        return None

    leaked = forms & LEAKED_FIELD_VALUES
    if leaked:
        raise ValueError("Found %s as a form of %s.  The parser may have changed the names of the inflection table fields.  Entry:\n%s" % (sorted(leaked), lemma, entry))
    if lemma != 'sein' and 'sein' in forms:
        raise ValueError("Found the auxiliary verb sein as a form of %s.  The parser may have changed the names of the inflection table fields.  Entry:\n%s" % (lemma, entry))

    forms = {form for form in forms
             if not any(x.isspace() for x in form) and not any(x in MARKUP_CHARACTERS for x in form)}
    if not forms:
        skipped['no usable forms'] += 1
        return None
    return lemma, upos, forms, canonical


def extract_lemmas(entries, skipped=None, unused_fields=None, ambiguous=None):
    """
    Build a sorted list of (lemma, form, upos) from parsed Wiktionary entries

    If skipped is a Counter, it is updated with the number of entries
    which were not used for each reason, plus the number of (form, upos)
    pairs dropped for having more than one lemma.  If unused_fields is
    a Counter, it is updated with the inflection table fields which are
    not forms.  If ambiguous is a list, the dropped forms are added to
    it as (form, upos, lemmas).
    """
    if skipped is None:
        skipped = Counter()

    form_to_lemma = defaultdict(set)
    lemma_to_form = defaultdict(set)
    # (variant spelling, upos) -> the lemma it points to
    canonical_lemma = {}
    for entry in entries:
        result = entry_forms(entry, skipped, unused_fields)
        if result is None:
            continue
        lemma, upos, forms, canonical = result
        if canonical is not None and canonical != lemma:
            canonical_lemma[(lemma, upos)] = canonical
        for form in forms:
            for variant in {form, form.replace("ß", "ss")}:
                form_to_lemma[(variant, upos)].add(lemma)
                lemma_to_form[(lemma, upos)].add(variant)

    # If the same form and POS maps to two different lemmas, that form
    # is removed from all of them, so the lemmatizer is not confused.
    # If the lemmas are a variant spelling and the lemma it points to,
    # such as Geschoß and Geschoss for Geschosse, the form goes to the
    # lemma the variant points to instead
    for (form, upos), lemmas in sorted(form_to_lemma.items()):
        if len(lemmas) <= 1:
            continue
        targets = {canonical_lemma.get((lemma, upos), lemma) for lemma in lemmas}
        if len(targets) == 1 and next(iter(targets)) in lemmas:
            skipped['variant form'] += 1
            keep = next(iter(targets))
            for lemma in lemmas:
                if lemma != keep:
                    lemma_to_form[(lemma, upos)].discard(form)
            continue
        skipped['ambiguous form'] += 1
        if ambiguous is not None:
            ambiguous.append((form, upos, sorted(lemmas)))
        for lemma in lemmas:
            lemma_to_form[(lemma, upos)].discard(form)

    processed = []
    for (lemma, upos) in sorted(lemma_to_form):
        for form in sorted(lemma_to_form[(lemma, upos)]):
            processed.append((lemma, form, upos))
    return processed


def write_conllu(processed, output_filename):
    """
    Write (lemma, form, upos) as single-word CoNLL-U sentences
    """
    with open(output_filename, 'w', encoding='utf-8', newline='\n') as fout:
        for lemma, form, upos in processed:
            fout.write("# sent_id = %s_%s_%s\n" % (form, lemma, upos))
            fout.write("# text = %s\n" % form)
            fout.write("1\t%s\t%s\t%s\t_\t_\t0\troot\t_\t_\n" % (form, lemma, upos))
            fout.write("\n")


def write_ambiguous(ambiguous, ambiguous_filename):
    """
    Write the dropped (form, upos, lemmas) as tab separated lines
    """
    with open(ambiguous_filename, 'w', encoding='utf-8', newline='\n') as fout:
        for form, upos, lemmas in ambiguous:
            fout.write("%s\t%s\t%s\n" % (form, upos, " ".join(lemmas)))


def convert_dump(dump_filename, output_filename, workers=None, allow_other_version=False, ambiguous_filename=None):
    """
    Convert a German Wiktionary dump to a CoNLL-U lemma file

    If ambiguous_filename is given, the forms dropped for having more
    than one lemma are written there.

    Returns the list of (lemma, form, upos) written
    """
    check_parser_version(allow_other_version)
    skipped = Counter()
    unused_fields = Counter()
    ambiguous = []
    processed = extract_lemmas(parse_dump(dump_filename, workers), skipped, unused_fields, ambiguous)
    write_conllu(processed, output_filename)
    if ambiguous_filename:
        write_ambiguous(ambiguous, ambiguous_filename)
    logger.info("Wrote %d (form, upos, lemma) entries to %s", len(processed), output_filename)
    variant_forms = skipped.pop('variant form', 0)
    logger.info("Skipped: %s", ", ".join("%s %d" % (reason, count) for reason, count in sorted(skipped.items())))
    logger.info("Forms shared by a variant spelling and the lemma it points to, given to that lemma: %d", variant_forms)
    logger.info("Inflection table fields not used as forms: %s", ", ".join("%s %d" % (field, count) for field, count in unused_fields.most_common()))
    return processed


def build_dictionary(paths, output_filename):
    """
    Convert the German Wiktionary dump in STANZA_EXTERN_DIR/german to a CoNLL-U dictionary at output_filename
    """
    dump_filename = os.path.join(paths["STANZA_EXTERN_DIR"], "german", DUMP_FILENAME)
    if not os.path.exists(dump_filename):
        raise FileNotFoundError("Could not find the German Wiktionary dump at %s ... please download it from https://dumps.wikimedia.org/dewiktionary/latest/" % dump_filename)
    convert_dump(dump_filename, output_filename)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Convert a German Wiktionary dump to a CoNLL-U lemma file")
    parser.add_argument('dump_file', help='German Wiktionary dump, such as %s' % DUMP_FILENAME)
    parser.add_argument('output_file', help='CoNLL-U file to write')
    parser.add_argument('--workers', type=int, default=None, help='Number of processes to parse the dump with.  Defaults to all but one of the CPUs')
    parser.add_argument('--allow_other_parser_version', action='store_true', help='Run even if the installed %s is not version %s' % (PARSER_PACKAGE, PARSER_VERSION))
    parser.add_argument('--ambiguous_file', default=None, help='Write the forms dropped for having more than one lemma to this file, one per line as form, upos, lemmas')
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args)
    processed = convert_dump(args.dump_file, args.output_file, workers=args.workers,
                             allow_other_version=args.allow_other_parser_version,
                             ambiguous_filename=args.ambiguous_file)
    print("Wrote %d entries to %s" % (len(processed), args.output_file), file=sys.stderr)


if __name__ == '__main__':
    main()
