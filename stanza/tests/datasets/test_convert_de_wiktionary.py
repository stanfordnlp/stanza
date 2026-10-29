"""
Tests for converting German Wiktionary to a lemma file

Most of the tests use stand-ins for the parser's entries, so they run
without wiktionary-de-parser, which needs Python 3.13.  The tests at
the end run the real parser on a few pages if it is installed.
"""

from collections import Counter
from types import SimpleNamespace

import pytest

from stanza.utils.conll import CoNLL
from stanza.utils.datasets.lemma import convert_de_wiktionary

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def fake_entry(page_name, pos, lemma=None, inflection=None, reference=None, language_code='de'):
    """
    Build something shaped like a wiktionary_de_parser ParsedEntry

    pos is one POS name or a list of them, and reference is None, or a
    (type, target) pair such as ("inflected", "lieben")
    """
    if isinstance(pos, str):
        pos = [pos]
    if reference is not None:
        reference = SimpleNamespace(type=reference[0], target=reference[1])
        if lemma is None:
            lemma = reference.target
    if lemma is None:
        lemma = page_name
    return SimpleNamespace(page_name=page_name, lemma=lemma, language_code=language_code,
                           pos=[SimpleNamespace(pos=x, subtypes=()) for x in pos],
                           inflection=inflection, reference=reference)

LIEBEN = fake_entry("lieben", "Verb", inflection={
    'present_1sg': 'liebe', 'present_2sg': 'liebst', 'present_3sg': 'liebt',
    'preterite_1sg': 'liebte', 'participle_2': 'geliebt', 'subjunctive_2_1sg': 'liebte',
    'imperative_singular': 'liebe', 'imperative_plural': 'liebt', 'auxiliary': 'haben'})

def test_verb_table():
    """
    The forms in the table are used, plus the infinitive, but not the auxiliary verb
    """
    processed = convert_de_wiktionary.extract_lemmas([LIEBEN])
    assert processed == [("lieben", form, "VERB") for form in sorted(["geliebt", "liebe", "lieben", "liebst", "liebt", "liebte"])]

def test_inflected_entries():
    """
    Inflected verb pages are used, inflected noun pages are not
    """
    entries = [fake_entry("möchtest", "Verb", reference=("inflected", "mögen")),
               fake_entry("Häuser", "Substantiv", reference=("inflected", "Haus"))]
    assert convert_de_wiktionary.extract_lemmas(entries) == [("mögen", "möchtest", "VERB")]

def test_variant_spelling():
    """
    A variant spelling is its own lemma
    """
    entry = fake_entry("Geografie", "Substantiv", reference=("variant", "Geographie"),
                       inflection={'gender': 'f', 'nominative_singular': 'Geografie', 'nominative_plural': 'Geografien'})
    assert convert_de_wiktionary.extract_lemmas([entry]) == [("Geografie", "Geografie", "NOUN"), ("Geografie", "Geografien", "NOUN")]

def test_non_form_fields():
    """
    gender, stem, and no_plural are not forms
    """
    entry = fake_entry("Tausendfache", "Substantiv", inflection={'gender': 'n', 'stem': 'Tausendfache', 'no_plural': 'ja'})
    assert convert_de_wiktionary.extract_lemmas([entry]) == [("Tausendfache", "Tausendfache", "NOUN")]

def test_eszett():
    """
    Forms with ß also get a version with ss, with the original lemma
    """
    entry = fake_entry("Fuß", "Substantiv", inflection={'nominative_singular': 'Fuß', 'nominative_plural': 'Füße', 'dative_plural': 'Füßen'})
    forms = [form for _, form, _ in convert_de_wiktionary.extract_lemmas([entry])]
    assert forms == sorted(["Fuß", "Fuss", "Füße", "Füsse", "Füßen", "Füssen"])

def test_ambiguous_forms():
    """
    A form with two lemmas for the same POS is dropped from both, including through ß -> ss
    """
    entries = [fake_entry("Maß", "Substantiv", inflection={'nominative_singular': 'Maß', 'nominative_plural': 'Maße'}),
               fake_entry("Masse", "Substantiv", inflection={'nominative_singular': 'Masse', 'nominative_plural': 'Massen'})]
    skipped = Counter()
    processed = convert_de_wiktionary.extract_lemmas(entries, skipped)
    assert ("Maß", "Masse", "NOUN") not in processed
    assert ("Masse", "Masse", "NOUN") not in processed
    assert ("Maß", "Maße", "NOUN") in processed
    assert ("Masse", "Massen", "NOUN") in processed
    assert skipped['ambiguous form'] == 1

def test_skipped_entries():
    entries = [fake_entry("instrument", "Substantiv", language_code='en'),
               fake_entry("und", "Konjunktion"),
               fake_entry("Abend", ["Substantiv", "Toponym"]),
               fake_entry("falscher Hase", "Substantiv"),
               fake_entry("liebst", "Verb", reference=("inflected", "lieben")),
               fake_entry("Gen", "Substantiv", inflection={'nominative_singular': '[[Gen]]', 'nominative_plural': 'Gene'})]
    skipped = Counter()
    processed = convert_de_wiktionary.extract_lemmas(entries, skipped)
    assert processed == [("Gen", "Gen", "NOUN"), ("Gen", "Gene", "NOUN"), ("lieben", "liebst", "VERB")]
    assert skipped == Counter({'language': 1, 'pos': 1, 'multiple pos': 1, 'multiword': 1})

def test_leaked_fields():
    """
    If the parser renames a field, its values show up as forms, which should fail loudly
    """
    entry = fake_entry("Abend", "Substantiv", inflection={'genus': 'm', 'nominative_singular': 'Abend'})
    with pytest.raises(ValueError):
        convert_de_wiktionary.extract_lemmas([entry])
    entry = fake_entry("lieben", "Verb", inflection={'hilfsverb': 'sein', 'present_1sg': 'liebe'})
    with pytest.raises(ValueError):
        convert_de_wiktionary.extract_lemmas([entry])

def test_write_conllu(tmp_path):
    output_file = tmp_path / "de_wiki_lemmas.conllu"
    convert_de_wiktionary.write_conllu(convert_de_wiktionary.extract_lemmas([LIEBEN]), str(output_file))
    doc = CoNLL.conll2doc(input_file=str(output_file))
    assert len(doc.sentences) == 6
    assert doc.sentences[0].sent_id == "geliebt_lieben_VERB"
    assert [(word.text, word.lemma, word.upos) for sentence in doc.sentences for word in sentence.words][0] == ("geliebt", "lieben", "VERB")

def test_parser_version(monkeypatch):
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.0.1")
    with pytest.raises(ImportError):
        convert_de_wiktionary.check_parser_version()
    convert_de_wiktionary.check_parser_version(allow_other_version=True)

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(ImportError):
        convert_de_wiktionary.check_parser_version(allow_other_version=True)

# Pages in the shape of the German Wiktionary wikitext, for the real parser
PAGES = {
    "lieben": """== lieben ({{Sprache|Deutsch}}) ==
=== {{Wortart|Verb|Deutsch}} ===

{{Deutsch Verb Übersicht
|Präsens_ich=liebe
|Präsens_du=liebst
|Präsens_er, sie, es=liebt
|Präteritum_ich=liebte
|Partizip II=geliebt
|Konjunktiv II_ich=liebte
|Imperativ Singular=liebe
|Imperativ Plural=liebt
|Hilfsverb=haben
}}

{{Bedeutungen}}
:[1] jemanden lieben
""",
    "liebst": """== liebst ({{Sprache|Deutsch}}) ==
=== {{Wortart|Konjugierte Form|Deutsch}} ===

{{Grammatische Merkmale}}
*2. Person Singular Indikativ Präsens Aktiv des Verbs '''[[lieben]]'''

{{Grundformverweis Konj|lieben}}
""",
    "Häuser": """== Häuser ({{Sprache|Deutsch}}) ==
=== {{Wortart|Deklinierte Form|Deutsch}} ===

{{Grammatische Merkmale}}
* Nominativ Plural des Substantivs '''[[Haus]]'''

{{Grundformverweis Dekl|Haus}}
""",
    "Fuß": """== Fuß ({{Sprache|Deutsch}}) ==
=== {{Wortart|Substantiv|Deutsch}}, {{m}} ===

{{Deutsch Substantiv Übersicht
|Genus=m
|Nominativ Singular=Fuß
|Nominativ Plural=Füße
|Genitiv Singular=Fußes
|Genitiv Plural=Füße
|Dativ Singular=Fuß
|Dativ Plural=Füßen
|Akkusativ Singular=Fuß
|Akkusativ Plural=Füße
}}
""",
    "Tausendfache": """== Tausendfache ({{Sprache|Deutsch}}) ==
=== {{Wortart|Substantiv|Deutsch}}, {{n}}, adjektivische Deklination ===

{{Deutsch adjektivisch Übersicht
|Genus=n
|Stamm=Tausendfache|kein Plural=ja
}}
""",
    "pittoresk": """== pittoresk ({{Sprache|Deutsch}}) ==
=== {{Wortart|Adjektiv|Deutsch}} ===

{{Deutsch Adjektiv Übersicht
|Positiv=pittoresk
|Komparativ=pittoresker
|Superlativ=pittoreskesten
}}
""",
}

def test_real_parser():
    """
    Run the pinned parser on a few pages, checking that its output is still what the converter expects
    """
    pytest.importorskip("wiktionary_de_parser")
    from wiktionary_de_parser import WiktionaryParser, WiktionaryPage
    convert_de_wiktionary.check_parser_version()

    parser = WiktionaryParser()
    entries = []
    for page_id, (name, wikitext) in enumerate(PAGES.items()):
        page = WiktionaryPage(page_id=page_id, name=name, wikitext=wikitext)
        entries.extend(parser.parse(entry) for entry in parser.entries(page))

    processed = convert_de_wiktionary.extract_lemmas(entries)
    by_lemma = {}
    for lemma, form, upos in processed:
        by_lemma.setdefault((lemma, upos), set()).add(form)
    assert by_lemma == {
        ("lieben", "VERB"): {"geliebt", "liebe", "lieben", "liebst", "liebt", "liebte"},
        ("Fuß", "NOUN"): {"Fuß", "Fuss", "Füße", "Füsse", "Fußes", "Fusses", "Füßen", "Füssen"},
        ("Tausendfache", "NOUN"): {"Tausendfache"},
        ("pittoresk", "ADJ"): {"pittoresk", "pittoresker", "pittoreskesten"},
    }
