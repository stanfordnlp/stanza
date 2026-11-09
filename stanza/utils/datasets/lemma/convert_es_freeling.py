"""
Convert the FreeLing Spanish dictionary to CoNLL-U rows for the Spanish lemmatizer.

FreeLing is an open source language analysis toolkit from TALP at
UPC.  Its Spanish dictionary comes from the Spanish Resource Grammar
at IULA, UPF, and is distributed under the LGPL-LR:

  https://github.com/TALP-UPC/FreeLing
  https://github.com/TALP-UPC/FreeLing/blob/master/COPYING

The entries are in data/es/dictionary/entries, one file per category
(MM.adj, MM.nom, MM.verb, ...), with one form, lemma, and EAGLES tag
per line:

  investigadoras investigador NCFP000

Adjectives (AQ, AO), common nouns (NC), verbs (VM, VA, VS), and
interjections (I) are converted.  The closed classes in MM.tanc are
skipped, as the treebanks cover those words and the conventions
differ from UD, such as muy -> mucho.  So are the adverbs (RG), which
are the same sort of short list: FreeLing analyzes -mente adverbs
without the dictionary.  Verbs are VERB, and also AUX for the lemmas
which UD Spanish uses as auxiliaries, such as ser, estar and poder.

These lemmas are changed to match UD Spanish:

  - Ordinals (AO) have the number as their lemma, such as
    decimotercera -> 13.  The lemma becomes the masculine singular
    ordinal, decimotercero, as in UD.  Apocopated forms get the full
    form, so primer -> primero and tercer -> tercero.

  - Feminine nouns have the masculine as their lemma when there is a
    masculine form, such as investigadoras -> investigador or
    hermanas -> hermano.  UD Spanish keeps the feminine singular as
    the lemma of a feminine noun, investigadora and hermana, so the
    lemma becomes the feminine singular form of the same FreeLing
    entry.  Adjectives keep the masculine lemma, as in UD.

  - bueno and malo are buen and mal, as in both GSD and AnCora.

  - The few adjectives with a verb as the lemma, such as
    parecida -> parecer, get the masculine singular adjective.

These entries are dropped:

  - Invariable nouns with a different lemma, which are abbreviations
    and spelling variants such as km -> kilómetro or omnibús -> ómnibus.
    UD keeps the form as the lemma.  A feminine singular abbreviation,
    such as cía -> compañía, already gets itself as the lemma.

  - Nouns and adjectives whose lemma starts differently from the form,
    ignoring accents, such as óptima -> bueno, thailandés -> tailandés,
    or email -> e-mail.  These are suppletive forms, spelling
    normalizations, abbreviations, and the occasional mistake, none of
    which match the UD lemma.

A (form, UPOS) pair with more than one lemma, such as sienta -> sentar /
sentir, is dropped entirely so that the seq2seq model makes the
decision instead of the dictionary.

The FreeLing repo keeps changing, so the first sentence of the output
records which version it came from: the git commit of the FreeLing
clone, if it is one, and a sha256 of the MM.* files.

Usage:
  python3 -m stanza.utils.datasets.lemma.convert_es_freeling FreeLing es_freeling.conllu
where FreeLing is a clone of the FreeLing repo, or its data/es/dictionary/entries directory
"""

import argparse
from collections import Counter, defaultdict
import glob
import hashlib
import logging
import os
import subprocess
import unicodedata

logger = logging.getLogger('stanza')

# the lemmas tagged AUX in UD Spanish.  ser, haber, and estar are VS or
# VA in FreeLing when they are auxiliaries, but the modals are VM
AUX_LEMMAS = frozenset(['ser', 'haber', 'estar', 'poder', 'deber'])

# adjective lemmas which UD Spanish writes as the apocopated form
ADJ_LEMMAS = {
    'bueno': 'buen',
    'malo': 'mal',
}

VERB_ENDINGS = ('ar', 'er', 'ir', 'ír')
PARTICIPLE_ENDINGS = ('ado', 'ido')

ENTRIES_SUBDIR = os.path.join("data", "es", "dictionary", "entries")

def tag_upos(tag, lemma):
    """
    Return the UPOS tags for a FreeLing EAGLES tag, or an empty list for a category which is skipped
    """
    if tag.startswith(('AQ', 'AO')):
        return ['ADJ']
    if tag.startswith('NC'):
        return ['NOUN']
    if tag.startswith('V'):
        if lemma in AUX_LEMMAS:
            return ['VERB', 'AUX']
        return ['VERB']
    if tag.startswith('I'):
        return ['INTJ']
    return []

def find_entries_dir(path):
    """
    Return the directory of MM.* files, given either a FreeLing clone or the entries directory itself
    """
    for candidate in (os.path.join(path, ENTRIES_SUBDIR), path):
        if glob.glob(os.path.join(candidate, "MM.*")):
            return candidate
    raise FileNotFoundError("Could not find the FreeLing Spanish dictionary in %s.  Expected the MM.* files in %s" % (path, os.path.join(path, ENTRIES_SUBDIR)))

def git_commit(entries_dir):
    """
    The git commit of the FreeLing clone containing entries_dir

    Returns None if entries_dir is not data/es/dictionary/entries of a
    git repo, such as a copy of the entries inside some other repo, or
    if git isn't available
    """
    def git(*args):
        result = subprocess.run(["git", "-C", entries_dir] + list(args), capture_output=True, text=True, check=True)
        return result.stdout.strip()

    try:
        toplevel = git("rev-parse", "--show-toplevel")
        if not os.path.samefile(os.path.join(toplevel, ENTRIES_SUBDIR), entries_dir):
            return None
        return git("rev-parse", "HEAD") or None
    except (OSError, subprocess.CalledProcessError):
        return None

def entries_sha256(entries_dir):
    """
    A sha256 of the names and contents of the MM.* files in entries_dir
    """
    digest = hashlib.sha256()
    for filename in sorted(glob.glob(os.path.join(entries_dir, "MM.*"))):
        digest.update(os.path.basename(filename).encode('utf-8'))
        with open(filename, 'rb') as fin:
            digest.update(fin.read())
    return digest.hexdigest()

def source_comments(entries_dir):
    """
    Comments recording which version of FreeLing a dictionary was built from
    """
    comments = []
    commit = git_commit(entries_dir)
    if commit is not None:
        comments.append(f"# freeling_commit = {commit}")
    comments.append(f"# freeling_entries_sha256 = {entries_sha256(entries_dir)}")
    return comments

def read_entries(entries_dir, skipped):
    """
    Read (form, lemma, tag) from each MM.* file in entries_dir

    Lines with anything other than three fields are counted in skipped
    """
    rows = []
    for filename in sorted(glob.glob(os.path.join(entries_dir, "MM.*"))):
        with open(filename, encoding='utf-8') as fin:
            for line in fin:
                pieces = line.split()
                if not pieces:
                    continue
                if len(pieces) != 3:
                    skipped['malformed line'] += 1
                    continue
                rows.append(tuple(pieces))
    return rows

def is_masc_sing_ordinal(tag):
    return tag.startswith('AO') and tag[3] == 'M' and tag[4] == 'S'

def is_fem_sing_noun(tag):
    return tag.startswith('NC') and tag[2] == 'F' and tag[3] == 'S'

def strip_accents(text):
    return "".join(x for x in unicodedata.normalize('NFD', text) if unicodedata.category(x) != 'Mn')

def same_start(form, lemma):
    """
    Whether the form and lemma have the same first two letters, ignoring case and accents
    """
    return strip_accents(form[:2].lower()) == strip_accents(lemma[:2].lower())

def masc_sing_adjective(form, tag):
    """
    Guess the masculine singular of an adjective from its form: parecidas -> parecido
    """
    if tag[4] == 'P' and form.endswith('s'):
        form = form[:-1]
    if tag[3] == 'F' and form.endswith('a'):
        form = form[:-1] + 'o'
    return form

def ordinal_lemma(form, tag, masc_sing):
    """
    The masculine singular ordinal for an ordinal form, given the masculine singular forms of the same number

    The masculine singular is guessed from the form (cuartas -> cuarto,
    primer -> primero) and used if FreeLing lists it for the same
    number.  Otherwise, if the number has a single masculine singular
    form ending in -o, that is the lemma.  Returns None if neither works.
    """
    guess = form
    if tag[4] == 'P' and guess.endswith('s'):
        guess = guess[:-1]
    if guess.endswith('a'):
        guess = guess[:-1] + 'o'
    elif not guess.endswith('o'):
        guess = guess + 'o'
    if guess in masc_sing:
        return guess
    if form in masc_sing and tag[3] == 'M' and tag[4] == 'S':
        return form
    candidates = [x for x in masc_sing if x.endswith('o')]
    if len(candidates) == 1:
        return candidates[0]
    return None

def feminine_lemma(form, lemma, tag, fem_sing):
    """
    The feminine singular lemma for a feminine noun, given the feminine singular forms of its FreeLing lemma

    A singular (or invariable) feminine noun is its own lemma.  A
    plural takes the feminine singular form of the same FreeLing lemma
    which it is the plural of, such as hermanas -> hermana or
    actrices -> actriz.  If the FreeLing lemma is itself feminine, or
    no feminine singular matches, the FreeLing lemma is kept.
    """
    if tag[3] != 'P':
        return form
    if lemma in fem_sing:
        return lemma
    matches = [x for x in fem_sing
               if form in (x + 's', x + 'es') or (x.endswith('z') and form == x[:-1] + 'ces')]
    if len(matches) == 1:
        return matches[0]
    return lemma

def convert_entries(rows, skipped):
    """
    Turn FreeLing (form, lemma, tag) rows into (form, upos, lemma) triples with UD lemmas
    """
    ordinals = defaultdict(set)
    fem_nouns = defaultdict(set)
    for form, lemma, tag in rows:
        if is_masc_sing_ordinal(tag):
            ordinals[lemma].add(form)
        elif is_fem_sing_noun(tag):
            fem_nouns[lemma].add(form)

    triples = []
    for form, lemma, tag in rows:
        upos_list = tag_upos(tag, lemma)
        if not upos_list:
            skipped['closed class ' + tag[:2]] += 1
            continue
        if '_' in form or '_' in lemma:
            skipped['multiword'] += 1
            continue
        if tag.startswith('AO') and lemma.isdigit():
            lemma = ordinal_lemma(form, tag, ordinals[lemma])
            if lemma is None:
                skipped['unresolved ordinal'] += 1
                continue
        elif tag.startswith('AQ') and lemma.endswith(VERB_ENDINGS) and masc_sing_adjective(form, tag).endswith(PARTICIPLE_ENDINGS):
            lemma = masc_sing_adjective(form, tag)
        elif tag.startswith('NC'):
            if tag[3] == 'N' and lemma != form:
                skipped['abbreviation or variant'] += 1
                continue
            if tag[2] == 'F':
                lemma = feminine_lemma(form, lemma, tag, fem_nouns[lemma])
        if tag.startswith(('A', 'NC')):
            if not same_start(form, lemma):
                skipped['unrelated lemma'] += 1
                continue
            lemma = ADJ_LEMMAS.get(lemma, lemma) if tag.startswith('A') else lemma
        for upos in upos_list:
            triples.append((form, upos, lemma))
    return triples

def build_lemma_dict(triples):
    """
    Map each (form, upos) to its lemma, dropping the pairs with more than one lemma

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

def write_conllu(lemma_dict, output_filename, first_comments=()):
    """
    Write {(form, upos): lemma} as single-word CoNLL-U sentences, sorted by form then UPOS

    first_comments are written as comment lines at the start of the first sentence
    """
    with open(output_filename, 'w', encoding='utf-8', newline='\n') as fout:
        for line in first_comments:
            fout.write(line + "\n")
        for (form, upos) in sorted(lemma_dict.keys()):
            # UPOS goes before the form, since UPOS never has a - in it but the form might
            fout.write(f"# sent_id = es_freeling-{upos}-{form}\n")
            fout.write(f"# text = {form}\n")
            fout.write(f"1\t{form}\t{lemma_dict[(form, upos)]}\t{upos}\t_\t_\t0\troot\t_\t_\n")
            fout.write("\n")

def write_ambiguous(ambiguous, filename):
    """
    Write the dropped (form, upos) pairs and their lemmas as a tsv
    """
    with open(filename, 'w', encoding='utf-8', newline='\n') as fout:
        for (form, upos), lemmas in sorted(ambiguous.items()):
            fout.write("%s\t%s\t%s\n" % (form, upos, " ".join(lemmas)))

def convert_directory(freeling_path, output_filename, ambiguous_filename=None):
    """
    Convert the FreeLing Spanish dictionary at freeling_path to a CoNLL-U dictionary at output_filename

    Returns the unambiguous {(form, upos): lemma} dict and the ambiguous pairs.
    """
    entries_dir = find_entries_dir(freeling_path)
    skipped = Counter()
    rows = read_entries(entries_dir, skipped)
    lemma_dict, ambiguous = build_lemma_dict(convert_entries(rows, skipped))
    write_conllu(lemma_dict, output_filename, source_comments(entries_dir))
    if ambiguous_filename:
        write_ambiguous(ambiguous, ambiguous_filename)
    logger.info("Wrote %d (form, upos, lemma) entries from %s to %s, skipping %d ambiguous (form, upos) pairs",
                len(lemma_dict), entries_dir, output_filename, len(ambiguous))
    logger.info("Rows skipped: %s", ", ".join("%s %d" % (reason, count) for reason, count in sorted(skipped.items())))
    upos_counts = Counter(upos for _, upos in lemma_dict)
    logger.info("Entries by UPOS: %s", ", ".join("%s %d" % (upos, count) for upos, count in sorted(upos_counts.items())))
    return lemma_dict, ambiguous

def build_dictionary(paths, output_filename):
    """
    Convert the FreeLing clone in STANZA_EXTERN_DIR/spanish/FreeLing to a CoNLL-U dictionary at output_filename
    """
    freeling_dir = os.path.join(paths["STANZA_EXTERN_DIR"], "spanish", "FreeLing")
    if not os.path.exists(freeling_dir):
        raise FileNotFoundError("Could not find FreeLing at %s ... please clone it with git clone https://github.com/TALP-UPC/FreeLing.git %s or use --no_external_dictionary to prepare the dataset without it" % (freeling_dir, freeling_dir))
    convert_directory(freeling_dir, output_filename)

def main(args=None):
    parser = argparse.ArgumentParser(description="Convert the FreeLing Spanish dictionary to a CoNLL-U lemma dictionary")
    parser.add_argument('input_dir', help='A clone of the FreeLing repo, or its data/es/dictionary/entries directory')
    parser.add_argument('output_file', help='Where to write the CoNLL-U dictionary')
    parser.add_argument('--ambiguous_file', default=None, help='Where to write the (form, upos) pairs dropped for having more than one lemma')
    args = parser.parse_args(args=args)
    lemma_dict, ambiguous = convert_directory(args.input_dir, args.output_file, args.ambiguous_file)
    print("Wrote %d entries to %s, skipping %d ambiguous (form, upos) pairs" % (len(lemma_dict), args.output_file, len(ambiguous)))

if __name__ == '__main__':
    main()
