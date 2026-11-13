"""
Tests for converting the kaikki.org extraction of Greek Wiktionary to a lemma file

The entries are small hand-written dicts in the shape of the kaikki
JSONL, with only the fields the converter reads.
"""

from collections import Counter
import gzip
import json
import lzma

import pytest

from stanza.utils.conll import CoNLL
from stanza.utils.datasets.lemma import convert_el_wiktionary

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def entry(word, pos, table=(), header=(), senses=None, tags=None, lang_code="el"):
    """
    Build a kaikki entry

    table is a list of forms from the declension or conjugation table,
    header is a list of (form, tags) from the header, and senses is a
    list of (gloss, form_of target, tags) for form-of senses, with a
    target of None for a sense which is not a form-of
    """
    forms = [{"form": form, "tags": list(form_tags), "source": "header"} for form, form_tags in header]
    source = "conjugation" if pos == "verb" else "declension"
    forms.extend({"form": form, "tags": ["singular"], "source": source} for form in table)
    result = {"word": word, "pos": pos, "lang_code": lang_code, "forms": forms, "senses": []}
    for gloss, target, sense_tags in (senses or [("ορισμός", None, ())]):
        sense = {"glosses": [gloss], "tags": list(sense_tags)}
        if target is not None:
            sense["form_of"] = [{"word": target}]
        result["senses"].append(sense)
    if tags:
        result["tags"] = list(tags)
    return result

def form_of(word, pos, gloss, target, tags=(), entry_tags=("form-of",)):
    return entry(word, pos, senses=[(gloss, target, tags)], tags=entry_tags)

def convert(entries, treebank_triples=None, contracted=True, skipped=None, ambiguous=None):
    lemmas = convert_el_wiktionary.build_lemmas(entries, treebank_triples, contracted)
    return convert_el_wiktionary.extract_lemmas(lemmas, skipped, ambiguous)

def lemma_of(processed, form, upos):
    lemmas = [lemma for lemma, x, y in processed if x == form and y == upos]
    assert len(lemmas) <= 1
    return lemmas[0] if lemmas else None

NIKI = entry("παραφροσύνη", "noun", table=["παραφροσύνη", "παραφροσύνες", "παραφροσύνης", "παραφροσυνών"])

def test_table_forms():
    """
    The table forms of a lemma entry are used, but not table placeholders or transliterations
    """
    noun = entry("παραφροσύνη", "noun", table=["παραφροσύνες", "παραφροσύνης"])
    noun["forms"].append({"form": "el-κλίση-'νίκη'", "tags": ["inflection-template"], "source": "declension"})
    noun["forms"].append({"form": "parafrosyni", "tags": ["transliteration"], "source": "declension"})
    processed = convert([noun])
    assert processed == [("παραφροσύνη", form, "NOUN") for form in sorted(["παραφροσύνη", "παραφροσύνες", "παραφροσύνης"])]

def test_form_of_entry():
    processed = convert([form_of("πλατειών", "noun", "γενική πληθυντικού του πλατεία", "πλατεία", ["genitive", "plural"])])
    assert processed == [("πλατεία", "πλατειών", "NOUN")]

def test_chain():
    """
    An adjective form which points to the feminine follows the chain to the masculine lemma
    """
    entries = [entry("ιαπωνικός", "adj", table=["ιαπωνικού"]),
               form_of("ιαπωνική", "adj", "ονομαστική ενικού, θηλυκού γένους του ιαπωνικός", "ιαπωνικός", ["feminine"]),
               form_of("ιαπωνικές", "adj", "ονομαστική πληθυντικού του ιαπωνική", "ιαπωνική")]
    processed = convert(entries)
    assert lemma_of(processed, "ιαπωνικές", "ADJ") == "ιαπωνικός"
    assert lemma_of(processed, "ιαπωνική", "ADJ") == "ιαπωνικός"

def test_chain_stays_in_pos():
    """
    A chain stops at a lemma entry and stays within one part of speech

    πρωτεύουσα is a noun, and also a form of the participle πρωτεύων
    """
    entries = [entry("πρωτεύουσα", "noun", table=["πρωτεύουσες"]),
               entry("πρωτεύων", "verb", tags=["participle"]),
               form_of("πρωτεύουσα", "verb", "ονομαστική ενικού, θηλυκού γένους του πρωτεύων", "πρωτεύων",
                       ["feminine"], entry_tags=["form-of", "participle"])]
    processed = convert(entries)
    assert lemma_of(processed, "πρωτεύουσα", "NOUN") == "πρωτεύουσα"
    assert lemma_of(processed, "πρωτεύουσα", "ADJ") == "πρωτεύων"

def test_variant():
    """
    A variant is its own lemma, keeping the spelling of the text, and its table is used
    """
    entries = [entry("χτίζω", "verb", table=["χτίζει"]),
               entry("κτίζω", "verb", table=["κτίζει"], senses=[("λογιότερη μορφή του χτίζω", "χτίζω", ())])]
    processed = convert(entries)
    assert lemma_of(processed, "κτίζω", "VERB") == "κτίζω"
    assert lemma_of(processed, "κτίζει", "VERB") == "κτίζω"
    assert lemma_of(processed, "χτίζει", "VERB") == "χτίζω"

def test_participles():
    """
    A participle is its own lemma as an ADJ, and goes to the active verb as a VERB
    """
    entries = [entry("αυξάνω", "verb", header=[("αυξάνομαι", ["passive"]), ("αυξημένος", ["passive", "participle"])]),
               entry("αυξημένος", "verb", tags=["participle"]),
               form_of("αυξημένη", "verb", "ονομαστική ενικού, θηλυκού γένους του αυξημένος", "αυξημένος",
                       ["feminine"], entry_tags=["form-of", "participle"]),
               entry("αναπτύσσω", "verb", header=[("αναπτύσσομαι", ["passive"])]),
               form_of("αναπτυσσόμενος", "verb", "μετοχή ενεστώτα του αναπτύσσομαι", "αναπτύσσομαι",
                       entry_tags=["form-of", "participle"])]
    processed = convert(entries)
    assert lemma_of(processed, "αυξημένη", "ADJ") == "αυξημένος"
    assert lemma_of(processed, "αυξημένη", "VERB") == "αυξάνω"
    assert lemma_of(processed, "αυξημένος", "VERB") == "αυξάνω"
    assert lemma_of(processed, "αναπτυσσόμενος", "ADJ") == "αναπτυσσόμενος"
    assert lemma_of(processed, "αναπτυσσόμενος", "VERB") == "αναπτύσσω"

def test_passive():
    """
    Passive forms go to the active verb, following links to a fixed point, and deponents keep their lemma
    """
    entries = [entry("χτίζω", "verb", header=[("χτίζομαι", ["passive"]), ("χτίστηκα", ["passive", "aorist"])]),
               entry("χτίζομαι", "verb", header=[("χτισμένος", ["passive", "participle"])]),
               form_of("χτίστηκε", "verb", "γ' ενικό οριστικής αορίστου του ρήματος χτίζομαι", "χτίζομαι"),
               entry("χτισμένος", "verb", tags=["participle"]),
               entry("έρχομαι", "verb", table=["έρχεται"])]
    processed = convert(entries)
    assert lemma_of(processed, "χτίστηκε", "VERB") == "χτίζω"
    assert lemma_of(processed, "χτισμένος", "VERB") == "χτίζω"
    assert lemma_of(processed, "χτισμένος", "ADJ") == "χτισμένος"
    assert lemma_of(processed, "έρχεται", "VERB") == "έρχομαι"
    # the aorist in the header is not a lemma
    assert lemma_of(processed, "χτίστηκα", "VERB") is None

def test_variant_passive_links():
    """
    The header links of a variant are used, even though the variant is a form-of entry
    """
    entries = [entry("κτίζω", "verb", header=[("κτισμένος", ["passive", "participle"])],
                     senses=[("λογιότερη μορφή του χτίζω", "χτίζω", ())]),
               entry("κτισμένος", "verb", tags=["participle"])]
    processed = convert(entries)
    assert lemma_of(processed, "κτισμένος", "VERB") == "κτίζω"

def test_degree():
    """
    Comparatives and superlatives go to the positive, unless the positive is polytonic
    """
    entries = [entry("μεγάλος", "adj"),
               entry("μεγαλύτερος", "adj", table=["μεγαλύτερη"], senses=[("συγκριτικός βαθμός του μεγάλος", None, ())]),
               form_of("μεγαλύτερο", "adj", "ουδέτερο του μεγαλύτερος", "μεγαλύτερος", ["neuter"]),
               entry("ελάχιστος", "adj", senses=[("υπερθετικός βαθμός του λίγος, ο πιο λίγος", None, ())]),
               entry("άριστος", "adj", senses=[("υπερθετικός βαθμός του ἀγαθός", None, ())])]
    processed = convert(entries)
    assert lemma_of(processed, "μεγαλύτερη", "ADJ") == "μεγάλος"
    assert lemma_of(processed, "μεγαλύτερο", "ADJ") == "μεγάλος"
    assert lemma_of(processed, "ελάχιστος", "ADJ") == "λίγος"
    assert lemma_of(processed, "άριστος", "ADJ") == "άριστος"

def test_contracted():
    entries = [entry("μιλάω", "verb", table=["μιλάει"])]
    assert lemma_of(convert(entries), "μιλάει", "VERB") == "μιλώ"
    assert lemma_of(convert(entries, contracted=False), "μιλάει", "VERB") == "μιλάω"

def test_ambiguous_forms():
    """
    A form with two lemmas for the same UPOS is dropped
    """
    entries = [entry("νέο", "noun", table=["νέα"]),
               entry("νέα", "noun", table=["νέες"])]
    skipped = Counter()
    ambiguous = []
    processed = convert(entries, skipped=skipped, ambiguous=ambiguous)
    assert lemma_of(processed, "νέα", "NOUN") is None
    assert lemma_of(processed, "νέες", "NOUN") == "νέα"
    assert skipped["ambiguous form"] == 1
    assert ambiguous == [("νέα", "NOUN", ["νέα", "νέο"])]

def test_names():
    """
    Names are PROPN, so they do not collide with forms of nouns
    """
    entries = [entry("σταυρός", "noun", table=["σταυρού"]),
               entry("Σταυρός", "name", table=["Σταυρού"])]
    processed = convert(entries)
    assert lemma_of(processed, "Σταυρού", "PROPN") == "Σταυρός"
    assert lemma_of(processed, "σταυρού", "NOUN") == "σταυρός"

def test_unusable_words():
    """
    Forms with spaces, digits, Latin letters, or polytonic accents are skipped, as are other parts of speech
    """
    entries = [entry("έρχομαι", "verb", table=["θα έρχομαι", "ερχόμαστε1", "erchomai", "ἔρχομαι", "έρχεσαι"]),
               entry("και", "conj")]
    skipped = Counter()
    lemmas = convert_el_wiktionary.build_lemmas(entries)
    processed = convert_el_wiktionary.extract_lemmas(lemmas, skipped)
    assert processed == [("έρχομαι", "έρχεσαι", "VERB"), ("έρχομαι", "έρχομαι", "VERB")]
    assert lemmas.skipped["pos conj"] == 1

def test_learn_rewrites():
    """
    A lemma which the treebank mostly gives a different lemma, at least twice, is rewritten
    """
    entries = [entry("λέω", "verb", table=["λέει", "είπα"]),
               entry("τρώω", "verb", table=["τρώει"])]
    triples = [("λέει", "VERB", "λέγω"), ("Είπα", "VERB", "λέγω"), ("τρώει", "VERB", "τρώγω")]
    lemmas = convert_el_wiktionary.build_lemmas(entries, triples)
    # Είπα is found lowercased, and τρώγω is only seen once
    assert lemmas.rewrites == {("λέω", "VERB"): "λέγω"}
    processed = convert_el_wiktionary.extract_lemmas(lemmas)
    assert lemma_of(processed, "είπα", "VERB") == "λέγω"
    assert lemma_of(processed, "τρώει", "VERB") == "τρώω"

def write_dump(path, entries):
    with lzma.open(path, "wt", encoding="utf-8") as fout:
        for item in entries:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

def test_read_dump(tmp_path):
    """
    Only Greek entries are read, from a plain or compressed dump
    """
    dump = tmp_path / "el-extract.jsonl.xz"
    write_dump(dump, [NIKI, entry("casa", "noun", lang_code="it")])
    assert [x["word"] for x in convert_el_wiktionary.read_dump(str(dump))] == ["παραφροσύνη"]

def test_convert_dump(tmp_path):
    dump = tmp_path / "el-extract.jsonl.xz"
    write_dump(dump, [entry("λέω", "verb", table=["λέει"]), entry("νέο", "noun", table=["νέα"]), entry("νέα", "noun")])
    treebank = tmp_path / "train.conllu"
    treebank.write_text("1\tλέει\tλέγω\tVERB\t_\t_\t0\troot\t_\t_\n\n2\tλέω\tλέγω\tVERB\t_\t_\t0\troot\t_\t_\n\n", encoding="utf-8")
    output = tmp_path / "el_wiki_lemmas.conllu.gz"
    processed = convert_el_wiktionary.convert_dump(str(dump), str(output), [str(treebank)],
                                                   ambiguous_filename=str(tmp_path / "ambiguous.tsv"),
                                                   rewrites_filename=str(tmp_path / "rewrites.tsv"))
    # νέα is dropped as ambiguous between νέο and νέα
    assert processed == [("λέγω", "λέει", "VERB"), ("λέγω", "λέω", "VERB"), ("νέο", "νέο", "NOUN")]
    with gzip.open(output, "rt", encoding="utf-8") as fin:
        doc = CoNLL.conll2doc(input_str=fin.read())
    assert [(w.text, w.lemma, w.upos) for s in doc.sentences for w in s.words] == [("λέει", "λέγω", "VERB"), ("λέω", "λέγω", "VERB"), ("νέο", "νέο", "NOUN")]
    assert (tmp_path / "rewrites.tsv").read_text(encoding="utf-8") == "λέω\tVERB\tλέγω\n"
    assert (tmp_path / "ambiguous.tsv").read_text(encoding="utf-8") == "νέα\tNOUN\tνέα νέο\n"

def test_check_conventions(tmp_path):
    """
    The check passes when the dictionary agrees with the training data, and fails when a convention has changed
    """
    train = tmp_path / "train.conllu"
    train.write_text("1\tχτίστηκε\tχτίζω\tVERB\t_\tVoice=Pass\t0\troot\t_\t_\n"
                     "2\tμεγαλύτερο\tμεγάλος\tADJ\t_\tDegree=Cmp\t1\tamod\t_\t_\n"
                     "3\tκτισμένο\tκτίζω\tVERB\t_\tVerbForm=Part\t1\tacl\t_\t_\n"
                     "4\tαυξημένη\tαυξημένος\tADJ\t_\t_\t1\tamod\t_\t_\n"
                     "5\tμιλάει\tμιλώ\tVERB\t_\t_\t1\tconj\t_\t_\n\n", encoding="utf-8")
    good = [("χτίζω", "χτίστηκε", "VERB"), ("μεγάλος", "μεγαλύτερο", "ADJ"), ("κτίζω", "κτισμένο", "VERB"),
            ("αυξημένος", "αυξημένη", "ADJ"), ("μιλώ", "μιλάει", "VERB")]
    dict_file = tmp_path / "dict.conllu"
    convert_el_wiktionary.write_conllu(good, str(dict_file))
    results = convert_el_wiktionary.check_conventions(str(train), str(dict_file))
    assert [(agree, total) for _, agree, total in results] == [(1, 1)] * 5

    # as if the treebank had switched to -άω lemmas
    bad = [x if x[1] != "μιλάει" else ("μιλάω", "μιλάει", "VERB") for x in good]
    convert_el_wiktionary.write_conllu(bad, str(dict_file))
    with pytest.raises(ValueError, match="contracted verbs"):
        convert_el_wiktionary.check_conventions(str(train), str(dict_file))

def test_prepare_dictionary(tmp_path):
    """
    prepare_lemma_treebank converts the dump in DICTIONARY_DIR to el_gdt.dict.conllu, with rewrites from the GDT training file
    """
    from stanza.utils.datasets import prepare_lemma_treebank
    paths = {"DICTIONARY_DIR": str(tmp_path / "dictionary"),
             "UDBASE": str(tmp_path / "ud"),
             "LEMMA_DATA_DIR": str(tmp_path / "data" / "lemma")}
    with pytest.raises(FileNotFoundError, match="el-extract.jsonl.gz"):
        prepare_lemma_treebank.prepare_external_dictionary(paths, "el_gdt")

    dump_dir = tmp_path / "dictionary" / "greek"
    dump_dir.mkdir(parents=True)
    with gzip.open(dump_dir / "el-extract.jsonl.gz", "wt", encoding="utf-8") as fout:
        for item in (entry("λέω", "verb", table=["λέει"]), NIKI, entry("casa", "noun", lang_code="it")):
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    # the dump is there, but the treebank to count the rewrites from is not
    with pytest.raises(FileNotFoundError, match="UD_Greek-GDT"):
        prepare_lemma_treebank.prepare_external_dictionary(paths, "el_gdt")

    gdt_dir = tmp_path / "ud" / "UD_Greek-GDT"
    gdt_dir.mkdir(parents=True)
    (gdt_dir / "el_gdt-ud-train.conllu").write_text("1\tλέει\tλέγω\tVERB\t_\t_\t0\troot\t_\t_\n\n"
                                                     "1\tλέω\tλέγω\tVERB\t_\t_\t0\troot\t_\t_\n\n", encoding="utf-8")
    dict_file = prepare_lemma_treebank.prepare_external_dictionary(paths, "el_gdt")
    assert dict_file == str(tmp_path / "data" / "lemma" / "el_gdt.dict.conllu")
    words = [(w.text, w.lemma, w.upos) for s in CoNLL.conll2doc(input_file=dict_file).sentences for w in s.words]
    assert ("λέει", "λέγω", "VERB") in words
    assert ("παραφροσύνες", "παραφροσύνη", "NOUN") in words
    assert len(words) == 6
    rewrites = tmp_path / "data" / "lemma" / "el_gdt.dict.rewrites.tsv"
    assert rewrites.read_text(encoding="utf-8") == "λέω\tVERB\tλέγω\n"
