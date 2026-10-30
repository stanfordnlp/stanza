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

def fake_entry(page_name, pos, lemma=None, inflection=None, reference=None, language_code='de', subtypes=()):
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
                           pos=[SimpleNamespace(pos=x, subtypes=tuple(subtypes)) for x in pos],
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
    gender and no_plural are not forms, and are counted as unused fields
    """
    entry = fake_entry("Abend", "Substantiv", inflection={'gender': 'm', 'nominative_singular': 'Abend', 'no_plural': 'ja'})
    unused_fields = Counter()
    assert convert_de_wiktionary.extract_lemmas([entry], unused_fields=unused_fields) == [("Abend", "Abend", "NOUN")]
    assert unused_fields == Counter({'gender': 1, 'no': 1})

def test_adjectival_nouns():
    """
    A noun with adjectival declension uses its stem as the lemma, with the endings added for the forms
    """
    entry = fake_entry("Abgeordneter", "Substantiv", inflection={'gender': 'm', 'stem': 'Abgeordnete'})
    unused_fields = Counter()
    processed = convert_de_wiktionary.extract_lemmas([entry], unused_fields=unused_fields)
    assert unused_fields == Counter({'gender': 1})
    assert processed == [("Abgeordnete", form, "NOUN") for form in ["Abgeordnete", "Abgeordnetem", "Abgeordneten", "Abgeordneter"]]

    entry = fake_entry("Tausendfaches", "Substantiv", inflection={'gender': 'n', 'stem': 'Tausendfache', 'no_plural': 'ja'})
    processed = convert_de_wiktionary.extract_lemmas([entry])
    assert processed == [("Tausendfache", form, "NOUN") for form in ["Tausendfache", "Tausendfachem", "Tausendfachen", "Tausendfaches"]]

def test_untranslated_fields():
    """
    Fields with a * attached are left in German by the parser

    brechen can take either auxiliary, so its table has Hilfsverb=haben
    and Hilfsverb*=sein, and the second one is not a form of brechen.
    Alternative forms such as Genitiv Singular* are still forms.
    """
    entry = fake_entry("brechen", "Verb", inflection={'present_1sg': 'breche', 'participle_2': 'gebrochen',
                                                       'auxiliary': 'haben', 'hilfsverb*': 'sein'})
    unused_fields = Counter()
    processed = convert_de_wiktionary.extract_lemmas([entry], unused_fields=unused_fields)
    assert processed == [("brechen", form, "VERB") for form in ["breche", "brechen", "gebrochen"]]
    assert unused_fields == Counter({'auxiliary': 1, 'hilfsverb': 1})

    entry = fake_entry("Abend", "Substantiv", inflection={'nominative_singular': 'Abend', 'genitive_singular': 'Abends',
                                                           'genitive_singular*': 'Abendes', 'genus*': 'm'})
    processed = convert_de_wiktionary.extract_lemmas([entry])
    assert processed == [("Abend", form, "NOUN") for form in ["Abend", "Abendes", "Abends"]]

def test_field_type():
    assert convert_de_wiktionary.field_type("hilfsverb*") == "hilfsverb"
    assert convert_de_wiktionary.field_type("genitive_singular*") == "genitive"
    assert convert_de_wiktionary.field_type("gender_1") == "gender"
    assert convert_de_wiktionary.field_type("present_3sg") == "present"

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

    ambiguous = []
    convert_de_wiktionary.extract_lemmas(entries, ambiguous=ambiguous)
    assert ambiguous == [("Masse", "NOUN", ["Masse", "Maß"])]

def test_names():
    """
    Names are PROPN, so they do not collide with forms of nouns
    """
    entries = [fake_entry("Aal", "Substantiv", inflection={'nominative_singular': 'Aal', 'nominative_plural': 'Aale', 'dative_plural': 'Aalen'}),
               fake_entry("Aalen", "Substantiv", subtypes=["Toponym"], inflection={'nominative_singular': 'Aalen', 'genitive_singular': 'Aalens'})]
    skipped = Counter()
    processed = convert_de_wiktionary.extract_lemmas(entries, skipped)
    assert processed == [("Aal", "Aal", "NOUN"), ("Aal", "Aale", "NOUN"), ("Aal", "Aalen", "NOUN"),
                         ("Aalen", "Aalen", "PROPN"), ("Aalen", "Aalens", "PROPN")]
    assert skipped['ambiguous form'] == 0

def test_variant_forms():
    """
    A form shared by an old spelling and the lemma it points to goes to that lemma
    """
    entries = [fake_entry("Geschoss", "Substantiv", inflection={'nominative_singular': 'Geschoss', 'nominative_plural': 'Geschosse', 'genitive_singular': 'Geschosses'}),
               fake_entry("Geschoß", "Substantiv", reference=("variant", "Geschoss"),
                          inflection={'nominative_singular': 'Geschoß', 'nominative_plural': 'Geschosse', 'genitive_singular': 'Geschosses'})]
    skipped = Counter()
    processed = convert_de_wiktionary.extract_lemmas(entries, skipped)
    # Geschoss is also the ss version of Geschoß
    assert processed == [("Geschoss", "Geschoss", "NOUN"), ("Geschoss", "Geschosse", "NOUN"), ("Geschoss", "Geschosses", "NOUN"),
                         ("Geschoß", "Geschoß", "NOUN")]
    assert skipped['variant form'] == 3
    assert skipped['ambiguous form'] == 0

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
    A gender or an auxiliary verb in a field which should hold forms fails loudly
    """
    entry = fake_entry("Abend", "Substantiv", inflection={'nominative_singular': 'Abend', 'nominative_plural': 'm'})
    with pytest.raises(ValueError):
        convert_de_wiktionary.extract_lemmas([entry])
    entry = fake_entry("lieben", "Verb", inflection={'present_1sg': 'liebe', 'participle_2': 'sein'})
    with pytest.raises(ValueError):
        convert_de_wiktionary.extract_lemmas([entry])
    # sein is a form of sein, though
    entry = fake_entry("sein", "Verb", inflection={'present_1sg': 'bin', 'present_1pl': 'sind', 'auxiliary': 'sein'})
    assert [form for _, form, _ in convert_de_wiktionary.extract_lemmas([entry])] == ["bin", "sein", "sind"]

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
    "brechen": """== brechen ({{Sprache|Deutsch}}) ==
=== {{Wortart|Verb|Deutsch}} ===

{{Deutsch Verb Übersicht
|Präsens_ich=breche
|Präsens_du=brichst
|Präsens_er, sie, es=bricht
|Präteritum_ich=brach
|Partizip II=gebrochen
|Konjunktiv II_ich=bräche
|Imperativ Singular=brich
|Imperativ Plural=brecht
|Hilfsverb=haben
|Hilfsverb*=sein
}}
""",
    "Abend": """== Abend ({{Sprache|Deutsch}}) ==
=== {{Wortart|Substantiv|Deutsch}}, {{m}} ===

{{Deutsch Substantiv Übersicht
|Genus=m
|Nominativ Singular=Abend
|Nominativ Plural=Abende
|Genitiv Singular=Abends
|Genitiv Singular*=Abendes
|Genitiv Plural=Abende
|Dativ Singular=Abend
|Dativ Plural=Abenden
|Akkusativ Singular=Abend
|Akkusativ Plural=Abende
}}
""",
    "Geschoß": """== Geschoß ({{Sprache|Deutsch}}) ==
=== {{Wortart|Substantiv|Deutsch}}, {{n}} ===

{{Alte Schreibweise|Geschoss|Reform 1996}}

{{Deutsch Substantiv Übersicht
|Genus=n
|Nominativ Singular=Geschoß
|Nominativ Plural=Geschosse
|Genitiv Singular=Geschosses
|Dativ Plural=Geschossen
}}
""",
    "Geschoss": """== Geschoss ({{Sprache|Deutsch}}) ==
=== {{Wortart|Substantiv|Deutsch}}, {{n}} ===

{{Deutsch Substantiv Übersicht
|Genus=n
|Nominativ Singular=Geschoss
|Nominativ Plural=Geschosse
|Genitiv Singular=Geschosses
|Dativ Plural=Geschossen
}}
""",
    "Abgeordneter": """== Abgeordneter ({{Sprache|Deutsch}}) ==
=== {{Wortart|Substantiv|Deutsch}}, {{m}}, adjektivische Deklination ===

{{Deutsch adjektivisch Übersicht
|Genus=m
|Stamm=Abgeordnete
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
        ("brechen", "VERB"): {"breche", "brechen", "brichst", "bricht", "brach", "gebrochen", "bräche", "brich", "brecht"},
        ("Abend", "NOUN"): {"Abend", "Abende", "Abends", "Abendes", "Abenden"},
        ("Fuß", "NOUN"): {"Fuß", "Fuss", "Füße", "Füsse", "Fußes", "Fusses", "Füßen", "Füssen"},
        ("Tausendfache", "NOUN"): {"Tausendfache", "Tausendfachen", "Tausendfaches", "Tausendfachem"},
        ("pittoresk", "ADJ"): {"pittoresk", "pittoresker", "pittoreskesten"},
        ("Geschoss", "NOUN"): {"Geschoss", "Geschosse", "Geschosses", "Geschossen"},
        ("Geschoß", "NOUN"): {"Geschoß"},
        ("Abgeordnete", "NOUN"): {"Abgeordnete", "Abgeordneten", "Abgeordneter", "Abgeordnetem"},
    }
