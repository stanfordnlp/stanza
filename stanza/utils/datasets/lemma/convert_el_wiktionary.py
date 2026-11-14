"""
Convert Greek Wiktionary to CoNLL-U rows for the Greek lemmatizer.

The input is the kaikki.org extraction of Greek Wiktionary (el.wiktionary),
which covers every language described there:

  https://kaikki.org/elwiktionary/rawdata.html

either the whole el-extract.jsonl or a filtered copy, plain or compressed
with xz or gzip.  Only the Greek entries (lang_code el) are used.

Nouns, verbs, adjectives, and names (PROPN) are kept.  The pairs come from:

- the declension and conjugation tables of lemma entries, plus the lemma itself
- form-of entries, such as πλατειών, "γενική πληθυντικού του πλατεία",
  which point to their lemma

The dictionary is for el_combined, which is built from UD_Greek-GDT.
el_gdt is GDT alone, without the dictionary, so that it can be
compared with other pipelines.

Wiktionary's conventions differ from UD_Greek-GDT in several places,
and the pairs are rewritten to match GDT:

- A form-of sense is classified by its gloss, since many inflected
  forms have no tags:
  - a variant, such as κτίζω, "λογιότερη μορφή του χτίζω", or a
    diminutive, is its own lemma, keeping the spelling of the text
  - a participle form, "μετοχή ...", is its own lemma as an ADJ, and
    points to its verb as a VERB
  - anything else is an inflected form of its target
- An inflected form whose target is itself an inflected form, such as an
  adjective form which points to the feminine (ιαπωνικές -> ιαπωνική ->
  ιαπωνικός), follows the chain to an entry which is a lemma.  Chains
  stay within one part of speech.
- A comparative or superlative adjective is a lemma entry whose gloss is
  "συγκριτικός βαθμός του X" or similar.  GDT lemmatizes it to the
  positive X, so its forms do too.
- A verb entry lists its passive (-ομαι) and passive participle (-μένος)
  in its header.  GDT lemmatizes passive verb forms to the active verb,
  so as a VERB, a passive or participle lemma follows those links to the
  active verb (χτισμένος -> χτίζομαι -> χτίζω).  A deponent such as
  έρχομαι has no such link and keeps its lemma.  An ADJ keeps the
  participle, as in GDT.
- GDT uses -ώ rather than -άω for contracted verbs (μιλώ, not μιλάω).
  --no_contracted turns this off, for a treebank such as GUD which
  keeps -άω.
- With --rewrite_treebank, lemma substitutions are counted from a
  training treebank: when the treebank gives most of the forms of a
  Wiktionary lemma a different lemma, at least twice, the Wiktionary
  lemma is replaced, such as λέω -> λέγω for GDT.  This is a fixed
  table once the inputs are fixed.  --rewrites_file writes it out for
  review.

prepare_lemma_treebank runs check_conventions on the converted file, which
fails if the training data no longer follows the conventions above, such
as after a change in a new UD release.

Forms and lemmas with spaces, digits, markup, or polytonic accents are
skipped, as are table placeholders and transliterations.  A (form, UPOS)
with more than one lemma after the rewrites is dropped entirely, so that
the seq2seq model makes the decision instead of the dictionary.
--ambiguous_file writes out the forms which were dropped.

The output has one single-word sentence per (form, UPOS, lemma), with
the case of the forms as they are in Wiktionary.

prepare_lemma_treebank converts the dump at

  $DICTIONARY_DIR/greek/el-extract.jsonl.gz

(extern_data/dictionary by default, and .xz or uncompressed also work)
to data/lemma/el_combined.dict.conllu, counting the rewrites from the
UD_Greek-GDT training file in $UDBASE.  The rewrites are written next
to it as el_combined.dict.rewrites.tsv.  run_lemma passes the dictionary to
the lemmatizer as its --secondary_file.  To update the dictionary,
download a newer el-extract.jsonl.gz to that directory.

Greek Wiktionary is licensed under CC BY-SA 4.0.

Usage, to convert a dump by hand:
  python3 -m stanza.utils.datasets.lemma.convert_el_wiktionary el-extract.jsonl.gz el_wiki_lemmas.conllu.gz --rewrite_treebank UD_Greek-GDT/el_gdt-ud-train.conllu

An output filename ending in .gz is written with gzip.
"""

import argparse
from collections import Counter, defaultdict
import gzip
import json
import logging
import lzma
import os
import re
import sys
import unicodedata

from stanza.utils.datasets import common
from stanza.utils.datasets.lemma.convert_de_wiktionary import write_conllu, write_ambiguous
from stanza.utils.datasets.lemma.convert_es_freeling import has_feature, read_conllu_words

logger = logging.getLogger('stanza')

# The dump prepare_lemma_treebank converts, under DICTIONARY_DIR, in order of preference
DUMP_FILENAMES = ("el-extract.jsonl.gz", "el-extract.jsonl.xz", "el-extract.jsonl")
DUMP_SUBDIR = "greek"

# The treebank the dictionary follows, whose training file the rewrites
# are counted from.  This is the treebank el_combined is built from in
# build_combined_greek_dataset
REWRITE_TREEBANK = "UD_Greek-GDT"

# kaikki POS -> UPOS
POS_MAP = {
    "noun": "NOUN",
    "verb": "VERB",
    "adj":  "ADJ",
    "name": "PROPN",
}

# kaikki POS of the entries which can be in a chain for each UPOS.
# Participle forms are filed under verb, but are ADJ as well as VERB
CHAIN_POS = {
    "NOUN":  ("noun",),
    "VERB":  ("verb",),
    "ADJ":   ("adj", "verb"),
    "PROPN": ("name",),
}

TABLE_SOURCES = {"declension", "conjugation", "inflection"}

# Glosses of form-of senses which are a separate lemma of their own,
# rather than an inflected form: άλλη μορφή του, λογιότερη μορφή του,
# υποκοριστικό του, and so on
VARIANT_GLOSS = re.compile(r"μορφή\s+του|υποκοριστικό|μεγεθυντικό")
PARTICIPLE_GLOSS = re.compile(r"^μετοχή")
DEGREE_GLOSS = re.compile(r"^(?:συγκριτικός|υπερθετικός|απόλυτος υπερθετικός|σχετικός υπερθετικός)\s+βαθμός\s+του\s+([^\s,;.()]+)")

# A usable form or lemma: Greek letters, with an apostrophe or hyphen allowed
GREEK_WORD = re.compile(r"^[Ͱ-Ͽ'’΄-]+$")
GREEK_LETTER = re.compile(r"[Ͱ-Ͽ]")


def nfc(text):
    return unicodedata.normalize("NFC", text)


def usable_word(word):
    """
    Whether word can be a form or lemma: modern monotonic Greek with no spaces, digits, or markup

    Polytonic letters are in the Greek Extended block, U+1F00 to U+1FFF,
    which GREEK_WORD does not include
    """
    return bool(word) and GREEK_WORD.match(word) is not None and GREEK_LETTER.search(word) is not None


def open_dump(filename):
    """
    Open a kaikki dump as text, decompressing it if the filename ends in .xz or .gz
    """
    if filename.endswith(".xz"):
        return lzma.open(filename, "rt", encoding="utf-8")
    if filename.endswith(".gz"):
        return gzip.open(filename, "rt", encoding="utf-8")
    return open(filename, encoding="utf-8")


def read_dump(filename):
    """
    Yield the Greek entries of a kaikki dump as dicts
    """
    with open_dump(filename) as fin:
        for line in fin:
            entry = json.loads(line)
            if entry.get("lang_code") == "el":
                yield entry


def sense_gloss(sense):
    glosses = sense.get("glosses")
    return glosses[0] if glosses else ""


def entry_upos(entry):
    """
    Return the UPOS tags the forms of an entry are used with

    A participle is filed under verb, but is an ADJ as well
    """
    pos = entry.get("pos")
    if pos == "verb" and "participle" in entry.get("tags", []):
        return ("ADJ", "VERB")
    upos = POS_MAP.get(pos)
    return (upos,) if upos else ()


class GreekLemmas:
    """
    The pairs read from the Wiktionary entries, and the rewrites which turn them into GDT lemmas
    """
    def __init__(self, contracted=True):
        # (form, upos) -> raw Wiktionary lemmas
        self.pairs = defaultdict(set)
        # (word, kaikki pos) -> targets of its inflected form-of senses, for following chains
        self.inflection_links = defaultdict(set)
        # (word, kaikki pos) with a lemma entry, where chains stop
        self.lemma_entries = set()
        # passive verb or participle -> active verb, from verb headers
        self.passive_to_active = {}
        # comparative or superlative -> positive
        self.positive = {}
        # (lemma, upos) -> lemma, counted from a treebank
        self.rewrites = {}
        self.contracted = contracted
        self.skipped = Counter()

    def add_entry(self, entry):
        uposes = entry_upos(entry)
        if not uposes:
            self.skipped["pos %s" % entry.get("pos")] += 1
            return
        pos = entry["pos"]
        word = nfc(entry.get("word", ""))
        if not usable_word(word):
            self.skipped["unusable word"] += 1
            return

        if pos == "verb":
            # The header links are used even when the verb is itself a
            # form-of entry: κτίζω is a variant of χτίζω, but its header
            # still links κτισμένος to κτίζω
            for form in entry.get("forms", []):
                tags = form.get("tags", [])
                text = nfc(form.get("form", ""))
                if form.get("source") == "header" and "passive" in tags and "aorist" not in tags and usable_word(text):
                    self.passive_to_active[text] = word

        inflection_targets = set()
        participle_targets = set()
        is_variant = False
        for sense in entry.get("senses", []):
            gloss = sense_gloss(sense)
            targets = {nfc(x["word"]) for x in sense.get("form_of", []) if "word" in x}
            targets = {x for x in targets if usable_word(x) and x != word}
            if not targets:
                if pos == "adj":
                    match = DEGREE_GLOSS.match(gloss)
                    if match and usable_word(nfc(match.group(1))):
                        self.positive[word] = nfc(match.group(1))
                continue
            if VARIANT_GLOSS.search(gloss):
                is_variant = True
            elif PARTICIPLE_GLOSS.match(gloss):
                participle_targets.update(targets)
            else:
                inflection_targets.update(targets)

        if inflection_targets:
            self.inflection_links[(word, pos)].update(inflection_targets)
        if (inflection_targets or participle_targets) and not is_variant:
            self.skipped["form-of entry"] += 1
            for upos in uposes:
                self.pairs[(word, upos)].update(inflection_targets)
                if participle_targets:
                    if upos == "VERB":
                        self.pairs[(word, upos)].update(participle_targets)
                    else:
                        self.pairs[(word, upos)].add(word)
            return

        if is_variant:
            self.skipped["variant entry"] += 1
        self.skipped["lemma entry"] += 1
        self.lemma_entries.add((word, pos))
        for upos in uposes:
            self.pairs[(word, upos)].add(word)
        for form in entry.get("forms", []):
            tags = form.get("tags", [])
            text = nfc(form.get("form", ""))
            if form.get("source") not in TABLE_SOURCES or "inflection-template" in tags or "transliteration" in tags:
                continue
            if not usable_word(text):
                continue
            for upos in uposes:
                self.pairs[(text, upos)].add(word)

    def follow_chain(self, lemma, upos, max_depth=4):
        """
        Follow inflected form-of links from lemma, within the parts of speech for upos, to an entry which is a lemma
        """
        poses = CHAIN_POS[upos]
        for _ in range(max_depth):
            if any((lemma, pos) in self.lemma_entries for pos in poses):
                return lemma
            targets = set()
            for pos in poses:
                targets |= self.inflection_links.get((lemma, pos), set())
            if len(targets) != 1:
                return lemma
            lemma = next(iter(targets))
        return lemma

    def active_verb(self, lemma, max_depth=3):
        for _ in range(max_depth):
            if lemma not in self.passive_to_active:
                break
            lemma = self.passive_to_active[lemma]
        return lemma

    def gdt_lemma(self, lemma, upos, use_rewrites=True):
        """
        Rewrite one raw Wiktionary lemma to the GDT convention
        """
        lemma = self.follow_chain(lemma, upos)
        if upos == "ADJ":
            lemma = self.positive.get(lemma, lemma)
        if upos == "VERB":
            lemma = self.active_verb(lemma)
            if self.contracted and lemma.endswith("άω"):
                lemma = lemma[:-2] + "ώ"
        if use_rewrites:
            lemma = self.rewrites.get((lemma, upos), lemma)
        return lemma

    def lookup(self, form, upos, use_rewrites=True):
        """
        Return the set of GDT lemmas for form as upos, which is empty if the form is unknown
        """
        return {self.gdt_lemma(lemma, upos, use_rewrites) for lemma in self.pairs.get((form, upos), ())}

    def learn_rewrites(self, triples, min_count=2):
        """
        Count lemma rewrites from (form, upos, lemma) triples from a training treebank

        For each Wiktionary lemma which is the only lemma for a form, count
        the lemmas the treebank gives the form.  If most of the forms of a
        Wiktionary lemma get some other lemma, at least min_count times,
        that lemma replaces it.  A form which is not in the dictionary is
        looked up lowercased, since the first word of a sentence is capitalized.
        """
        counts = defaultdict(Counter)
        for form, upos, lemma in triples:
            if upos not in CHAIN_POS:
                continue
            lemmas = self.lookup(form, upos, use_rewrites=False) or self.lookup(form.lower(), upos, use_rewrites=False)
            if len(lemmas) == 1:
                counts[(next(iter(lemmas)), upos)][lemma] += 1
        self.rewrites = {}
        for key, gold in counts.items():
            best, best_count = gold.most_common(1)[0]
            if best != key[0] and best_count >= min_count and best_count * 2 > sum(gold.values()):
                self.rewrites[key] = best
        return self.rewrites


def read_treebank(filename):
    """
    Yield (form, upos, lemma) for each word of a CoNLL-U file
    """
    with open(filename, encoding="utf-8") as fin:
        for line in fin:
            if not line.strip() or line.startswith("#"):
                continue
            pieces = line.rstrip("\n").split("\t")
            if not pieces[0].isdigit():
                continue
            yield nfc(pieces[1]), pieces[3], nfc(pieces[2])


def build_lemmas(entries, treebank_triples=None, contracted=True):
    """
    Read the entries into a GreekLemmas, and count the rewrites from treebank_triples if given
    """
    lemmas = GreekLemmas(contracted=contracted)
    for entry in entries:
        lemmas.add_entry(entry)
    if treebank_triples is not None:
        lemmas.learn_rewrites(treebank_triples)
    return lemmas


def extract_lemmas(lemmas, skipped=None, ambiguous=None):
    """
    Return a sorted list of (lemma, form, upos), dropping forms with more than one lemma

    If skipped is a Counter, it is updated with the number of ambiguous
    forms dropped.  If ambiguous is a list, the dropped forms are added
    to it as (form, upos, lemmas).
    """
    processed = []
    for (form, upos) in sorted(lemmas.pairs):
        targets = lemmas.lookup(form, upos)
        targets = {x for x in targets if usable_word(x)}
        if len(targets) == 1:
            processed.append((next(iter(targets)), form, upos))
            continue
        if skipped is not None:
            skipped["ambiguous form"] += 1
        if ambiguous is not None:
            ambiguous.append((form, upos, sorted(targets)))
    processed.sort()
    return processed


def write_rewrites(rewrites, filename):
    """
    Write the rewrites as tab separated lines of Wiktionary lemma, UPOS, treebank lemma
    """
    with open(filename, 'w', encoding='utf-8', newline='\n') as fout:
        for (lemma, upos), target in sorted(rewrites.items()):
            fout.write("%s\t%s\t%s\n" % (lemma, upos, target))


def convert_dump(dump_filename, output_filename, treebank_filenames=(), contracted=True,
                 ambiguous_filename=None, rewrites_filename=None):
    """
    Convert a kaikki Greek Wiktionary dump to a CoNLL-U lemma file

    Returns the list of (lemma, form, upos) written
    """
    treebank_triples = None
    if treebank_filenames:
        treebank_triples = [triple for filename in treebank_filenames for triple in read_treebank(filename)]
    lemmas = build_lemmas(read_dump(dump_filename), treebank_triples, contracted)
    skipped = Counter()
    ambiguous = []
    processed = extract_lemmas(lemmas, skipped, ambiguous)
    write_conllu(processed, output_filename)
    if ambiguous_filename:
        write_ambiguous(ambiguous, ambiguous_filename)
    if rewrites_filename:
        write_rewrites(lemmas.rewrites, rewrites_filename)
    logger.info("Wrote %d (form, upos, lemma) entries to %s", len(processed), output_filename)
    logger.info("Entries: %s", ", ".join("%s %d" % (reason, count) for reason, count in sorted(lemmas.skipped.items())))
    logger.info("Passive links: %d  Degree links: %d  Rewrites: %d  Ambiguous forms dropped: %d",
                len(lemmas.passive_to_active), len(lemmas.positive), len(lemmas.rewrites), skipped["ambiguous form"])
    return processed


PARTICIPLE_ENDINGS = ("μένος", "μένη", "μένο", "μένοι", "μένες", "μένα", "μένου", "μένης", "μένων", "μένους")

def participle_form(form):
    return form.lower().endswith(PARTICIPLE_ENDINGS)

# The conventions which the lemma rewrites follow.  Each is a
# description, a function of a training word's (form, upos, feats,
# lemma) which selects the words the convention applies to, and the
# fraction of those words where the dictionary and the training data
# need to agree.
#
# GDT is not consistent about participles itself: some VERB participles
# have the participle as the lemma (αυξημένη -> αυξημένος), and some ADJ
# participles have the verb (χτισμένα -> χτίζω).  Their minimums are
# lower to allow for that.  A change of convention would make most of a
# category disagree, which is well below any of these.
CONVENTIONS = [
    ("passive verbs have the active lemma (δημιουργήθηκε -> δημιουργώ)",
     lambda form, upos, feats, lemma: upos == 'VERB' and has_feature(feats, 'Voice=Pass') and not has_feature(feats, 'VerbForm=Part'),
     0.85),
    ("VERB participles have the verb as the lemma (κτισμένο -> κτίζω)",
     lambda form, upos, feats, lemma: upos == 'VERB' and has_feature(feats, 'VerbForm=Part') and participle_form(form),
     0.6),
    ("ADJ participles have the participle as the lemma (αυξημένη -> αυξημένος)",
     lambda form, upos, feats, lemma: upos == 'ADJ' and participle_form(form),
     0.7),
    ("comparatives and superlatives have the positive as the lemma (μεγαλύτερο -> μεγάλος)",
     lambda form, upos, feats, lemma: upos == 'ADJ' and (has_feature(feats, 'Degree=Cmp') or has_feature(feats, 'Degree=Sup')),
     0.7),
    ("contracted verbs have the -ώ lemma (μιλάει -> μιλώ)",
     lambda form, upos, feats, lemma: upos == 'VERB' and lemma.endswith(('ώ', 'άω')),
     0.85),
]

def check_conventions(train_file, dict_file, conventions=CONVENTIONS):
    """
    Check that the dictionary and the training data agree on the words each of the conventions applies to

    The training lemma of a (form, upos) is its most frequent lemma.
    For each convention, the (form, upos) pairs in both the training
    data and the dictionary which the convention selects are compared.

    Raises ValueError if any convention has no words to compare or
    agrees on less than its minimum.  Returns a list of
    (description, agree, total) for the conventions.
    """
    dictionary = {(nfc(form), upos): nfc(lemma) for form, upos, lemma, _ in read_conllu_words(dict_file)}
    train_words = [(nfc(form), upos, nfc(lemma), feats) for form, upos, lemma, feats in read_conllu_words(train_file)]

    results = []
    failures = []
    for description, selects, min_agreement in conventions:
        lemmas = defaultdict(Counter)
        for form, upos, lemma, feats in train_words:
            if lemma != '_' and (form, upos) in dictionary and selects(form, upos, feats, lemma):
                lemmas[(form, upos)][lemma] += 1
        disagree = []
        for key, counts in sorted(lemmas.items()):
            train_lemma = counts.most_common(1)[0][0]
            if train_lemma != dictionary[key]:
                disagree.append((key[0], train_lemma, dictionary[key]))
        total = len(lemmas)
        agree = total - len(disagree)
        results.append((description, agree, total))
        logger.info("%s: the dictionary agrees with %s on %d of %d words", description, train_file, agree, total)
        if total == 0 or agree < min_agreement * total:
            examples = ", ".join("%s (%s / %s)" % x for x in disagree[:10])
            failures.append("%s: agrees on %d of %d words, needs %.0f%%.  Examples (training / dictionary): %s" % (description, agree, total, 100 * min_agreement, examples))

    if failures:
        raise ValueError("The Greek Wiktionary dictionary %s does not follow the lemma conventions of %s.  If the treebank conventions have changed, update the rules in stanza/utils/datasets/lemma/convert_el_wiktionary.py to match and convert the dump again.\n  %s" % (dict_file, train_file, "\n  ".join(failures)))
    return results


def find_dump(dictionary_dir):
    """
    Return the path of the Greek Wiktionary dump in dictionary_dir/greek, or raise FileNotFoundError
    """
    dump_dir = os.path.join(dictionary_dir, DUMP_SUBDIR)
    for filename in DUMP_FILENAMES:
        path = os.path.join(dump_dir, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Could not find the Greek Wiktionary dump at %s ... please download el-extract.jsonl.gz from https://kaikki.org/elwiktionary/rawdata.html to %s, or use --no_external_dictionary to prepare the dataset without it" % (os.path.join(dump_dir, DUMP_FILENAMES[0]), dump_dir))


def rewrites_filename(output_filename):
    """
    el_combined.dict.conllu -> el_combined.dict.rewrites.tsv
    """
    return os.path.splitext(output_filename)[0] + ".rewrites.tsv"


def build_dictionary(paths, output_filename):
    """
    Convert the Greek Wiktionary dump in DICTIONARY_DIR to output_filename, with rewrites counted from the GDT training file

    The rewrites are written next to output_filename, so that each
    conversion leaves a record of them.
    """
    dump_filename = find_dump(paths["DICTIONARY_DIR"])
    train_filename = common.find_treebank_dataset_file(REWRITE_TREEBANK, paths["UDBASE"], "train", "conllu", fail=True)
    logger.info("Converting %s to %s, with rewrites counted from %s", dump_filename, output_filename, train_filename)
    convert_dump(dump_filename, output_filename, [train_filename], rewrites_filename=rewrites_filename(output_filename))


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Convert the kaikki.org extraction of Greek Wiktionary to a CoNLL-U lemma file")
    parser.add_argument('dump_file', help='kaikki el-extract.jsonl, plain or compressed with xz or gzip')
    parser.add_argument('output_file', help='CoNLL-U file to write, gzipped if it ends in .gz')
    parser.add_argument('--rewrite_treebank', action='append', default=[],
                        help='Training treebank to count lemma rewrites from, such as el_gdt-ud-train.conllu.  Can be given more than once')
    parser.add_argument('--no_contracted', dest='contracted', action='store_false', default=True,
                        help="Keep -άω verb lemmas rather than rewriting them to -ώ as GDT does")
    parser.add_argument('--ambiguous_file', default=None, help='Write the forms dropped for having more than one lemma to this file, one per line as form, upos, lemmas')
    parser.add_argument('--rewrites_file', default=None, help='Write the rewrites counted from the treebank to this file, one per line as Wiktionary lemma, upos, treebank lemma')
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args)
    processed = convert_dump(args.dump_file, args.output_file, args.rewrite_treebank, args.contracted,
                             ambiguous_filename=args.ambiguous_file, rewrites_filename=args.rewrites_file)
    print("Wrote %d entries to %s" % (len(processed), args.output_file), file=sys.stderr)


if __name__ == '__main__':
    main()
