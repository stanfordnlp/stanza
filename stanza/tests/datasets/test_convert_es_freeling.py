"""
Tests for converting the FreeLing Spanish dictionary to a lemma dictionary
"""

import os
import shutil
import subprocess

import pytest

from stanza.utils.conll import CoNLL
from stanza.utils.datasets import prepare_lemma_treebank
from stanza.utils.datasets.lemma import convert_es_freeling

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

# A few lines of each FreeLing file, in its format: form lemma tag
SAMPLE = {
    "MM.adj": """
abacial abacial AQ0CS00
abaciales abacial AQ0CP00
conservadora conservador AQ0FS00
buena bueno AQ0FS00
óptima bueno AQ0FS00
parecidas parecer AQ0FP00
ocular ocular AQ0CS00
primer 1 AO0MS00
primero 1 AO0MS00
primera 1 AO0FS00
cuarto 4 AO0MS00
cuartas 4 AO0FP00
decimotercero 13 AO0MS00
decimotercera 13 AO0FS00
séptimo 7 AO0MS00
sétimo 7 AO0MS00
sétima 7 AO0FS00
""",
    "MM.nom": """
casa casa NCFS000
casas casa NCFP000
hermano hermano NCMS000
hermana hermano NCFS000
hermanas hermano NCFP000
actriz actriz NCFS000
actrices actriz NCFP000
cazadora cazador NCFS000
cazadora cazadora NCFS000
crisis crisis NCFN000
estudiantes estudiante NCCP000
km kilómetro NCMN000
cía compañía NCFS000
""",
    "MM.verb": """
cantaba cantar VMII3S0
puede poder VMIP3S0
sienta sentar VMIP3S0
sienta sentir VMSP3S0
fue ir VMIS3S0
era erar VMIP3S0
""",
    "MM.vaux": """
fue ser VSIS3S0
era ser VSII3S0
estado estar VAP00SM
""",
    "MM.adv": """
muy muy RG
""",
    "MM.int": """
abur abur I
""",
    "MM.tanc": """
a a SP
adonde adonde PR00000
""",
}

EXPECTED = {
    ("abacial", "ADJ"): "abacial",
    ("abaciales", "ADJ"): "abacial",
    # adjectives keep the masculine lemma
    ("conservadora", "ADJ"): "conservador",
    ("buena", "ADJ"): "buen",
    ("parecidas", "ADJ"): "parecido",
    ("ocular", "ADJ"): "ocular",
    # ordinals
    ("primer", "ADJ"): "primero",
    ("primero", "ADJ"): "primero",
    ("primera", "ADJ"): "primero",
    ("cuarto", "ADJ"): "cuarto",
    ("cuartas", "ADJ"): "cuarto",
    ("decimotercero", "ADJ"): "decimotercero",
    ("decimotercera", "ADJ"): "decimotercero",
    ("séptimo", "ADJ"): "séptimo",
    ("sétimo", "ADJ"): "sétimo",
    ("sétima", "ADJ"): "sétimo",
    # feminine nouns keep the feminine lemma
    ("casa", "NOUN"): "casa",
    ("casas", "NOUN"): "casa",
    ("hermano", "NOUN"): "hermano",
    ("hermana", "NOUN"): "hermana",
    ("hermanas", "NOUN"): "hermana",
    ("actriz", "NOUN"): "actriz",
    ("actrices", "NOUN"): "actriz",
    ("cazadora", "NOUN"): "cazadora",
    ("crisis", "NOUN"): "crisis",
    ("estudiantes", "NOUN"): "estudiante",
    # a feminine singular noun is its own lemma, even an abbreviation
    ("cía", "NOUN"): "cía",
    # verbs, with AUX for the auxiliaries
    ("cantaba", "VERB"): "cantar",
    ("puede", "VERB"): "poder",
    ("puede", "AUX"): "poder",
    ("fue", "AUX"): "ser",
    ("era", "AUX"): "ser",
    ("estado", "VERB"): "estar",
    ("estado", "AUX"): "estar",
    ("abur", "INTJ"): "abur",
}

EXPECTED_AMBIGUOUS = {
    ("sienta", "VERB"): ["sentar", "sentir"],
    ("fue", "VERB"): ["ir", "ser"],
    ("era", "VERB"): ["erar", "ser"],
}

def write_sample(directory):
    entries_dir = directory / "data" / "es" / "dictionary" / "entries"
    entries_dir.mkdir(parents=True)
    for filename, text in SAMPLE.items():
        (entries_dir / filename).write_text(text.lstrip(), encoding="utf-8")
    return entries_dir

def test_convert_directory(tmp_path):
    freeling_dir = tmp_path / "FreeLing"
    write_sample(freeling_dir)
    output_file = tmp_path / "es_freeling.conllu"
    ambiguous_file = tmp_path / "ambiguous.tsv"
    lemma_dict, ambiguous = convert_es_freeling.convert_directory(str(freeling_dir), str(output_file), str(ambiguous_file))
    assert lemma_dict == EXPECTED
    assert ambiguous == EXPECTED_AMBIGUOUS

    doc = CoNLL.conll2doc(input_file=str(output_file))
    words = {(word.text, word.upos): word.lemma for sentence in doc.sentences for word in sentence.words}
    assert words == EXPECTED
    sent_ids = [sentence.sent_id for sentence in doc.sentences]
    assert sent_ids[0] == "es_freeling-ADJ-abacial"
    # not a git clone, so only the hash of the entries is recorded
    first_comments = [x for x in doc.sentences[0].comments if not x.startswith(("# sent_id", "# text"))]
    assert first_comments == ["# freeling_entries_sha256 = %s" % convert_es_freeling.entries_sha256(str(freeling_dir / "data" / "es" / "dictionary" / "entries"))]
    assert len(set(sent_ids)) == len(sent_ids)

    assert ambiguous_file.read_text(encoding="utf-8").splitlines() == ["era\tVERB\terar ser", "fue\tVERB\tir ser", "sienta\tVERB\tsentar sentir"]

def test_entries_dir(tmp_path):
    """
    The converter takes either a FreeLing clone or the entries directory itself
    """
    entries_dir = write_sample(tmp_path / "FreeLing")
    lemma_dict, _ = convert_es_freeling.convert_directory(str(entries_dir), str(tmp_path / "out.conllu"))
    assert lemma_dict == EXPECTED

    with pytest.raises(FileNotFoundError):
        convert_es_freeling.convert_directory(str(tmp_path), str(tmp_path / "out.conllu"))

def test_build_dictionary(tmp_path):
    paths = {"STANZA_EXTERN_DIR": str(tmp_path / "extern")}
    output_file = tmp_path / "es_combined.dict.conllu"
    with pytest.raises(FileNotFoundError):
        convert_es_freeling.build_dictionary(paths, str(output_file))

    write_sample(tmp_path / "extern" / "spanish" / "FreeLing")
    convert_es_freeling.build_dictionary(paths, str(output_file))
    assert os.path.exists(output_file)

def test_ordinal_lemma():
    masc_sing = {"séptimo", "sétimo"}
    assert convert_es_freeling.ordinal_lemma("séptimas", "AO0FP00", masc_sing) == "séptimo"
    assert convert_es_freeling.ordinal_lemma("sétimo", "AO0MS00", masc_sing) == "sétimo"
    # no way to choose between the two
    assert convert_es_freeling.ordinal_lemma("sept", "AO0MS00", masc_sing) is None

def test_feminine_lemma():
    assert convert_es_freeling.feminine_lemma("hermanas", "hermano", "NCFP000", {"hermana"}) == "hermana"
    assert convert_es_freeling.feminine_lemma("hermana", "hermano", "NCFS000", {"hermana"}) == "hermana"
    assert convert_es_freeling.feminine_lemma("narices", "nariz", "NCFP000", {"nariz"}) == "nariz"
    # nothing matches, so the FreeLing lemma stays
    assert convert_es_freeling.feminine_lemma("afueras", "afuera", "NCFP000", set()) == "afuera"

@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_source_comments(tmp_path):
    """
    A git clone of FreeLing records its commit, and a copy of the entries inside some other repo does not
    """
    def git(directory, *args):
        result = subprocess.run(["git", "-C", str(directory)] + list(args), capture_output=True, text=True, check=True)
        return result.stdout.strip()

    freeling_dir = tmp_path / "FreeLing"
    entries_dir = write_sample(freeling_dir)
    git(freeling_dir, "init", "-q")
    git(freeling_dir, "add", ".")
    git(freeling_dir, "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-q", "-m", "sample")
    commit = git(freeling_dir, "rev-parse", "HEAD")
    sha = convert_es_freeling.entries_sha256(str(entries_dir))

    assert convert_es_freeling.source_comments(str(entries_dir)) == ["# freeling_commit = %s" % commit, "# freeling_entries_sha256 = %s" % sha]

    # the same entries somewhere else in the repo are not FreeLing's
    other_dir = freeling_dir / "copy"
    shutil.copytree(entries_dir, other_dir)
    assert convert_es_freeling.source_comments(str(other_dir)) == ["# freeling_entries_sha256 = %s" % sha]

# training words as (form, upos, lemma, feats)
TRAIN_WORDS = [
    ("investigadoras", "NOUN", "investigadora", "Gender=Fem|Number=Plur"),
    ("investigadoras", "NOUN", "investigadora", "Gender=Fem|Number=Plur"),
    ("jugadora",       "NOUN", "jugadora",      "Gender=Fem|Number=Sing"),
    ("conservadoras",  "ADJ",  "conservador",   "Gender=Fem|Number=Plur"),
    ("hermanas",       "NOUN", "hermana",       "Gender=Fem|Number=Plur"),
    ("casas",          "NOUN", "casa",          "Gender=Fem|Number=Plur"),
    ("primeras",       "ADJ",  "primero",       "Gender=Fem|Number=Plur|NumType=Ord"),
    ("buena",          "ADJ",  "buen",          "Gender=Fem|Number=Sing"),
    # not in the dictionary, so not compared
    ("ganadoras",      "NOUN", "ganador",       "Gender=Fem|Number=Plur"),
]

DICTIONARY = [
    ("investigadoras", "NOUN", "investigadora"),
    ("jugadora",       "NOUN", "jugadora"),
    ("conservadoras",  "ADJ",  "conservador"),
    ("hermanas",       "NOUN", "hermana"),
    ("casas",          "NOUN", "casa"),
    ("primeras",       "ADJ",  "primero"),
    ("buena",          "ADJ",  "buen"),
]

def write_words(filename, words):
    lines = []
    for form, upos, lemma, feats in words:
        lines.append(f"1\t{form}\t{lemma}\t{upos}\t_\t{feats}\t0\troot\t_\t_")
        lines.append("")
    filename.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(filename)

def test_check_conventions(tmp_path):
    train_file = write_words(tmp_path / "train.conllu", TRAIN_WORDS)
    dict_file = write_words(tmp_path / "dict.conllu", [(form, upos, lemma, "_") for form, upos, lemma in DICTIONARY])
    results = convert_es_freeling.check_conventions(train_file, dict_file)
    assert [(agree, total) for _, agree, total in results] == [(2, 2), (1, 1), (4, 4), (1, 1), (1, 1)]

def test_check_conventions_changed(tmp_path):
    """
    If the training data changes a convention, such as -dor for the -dora nouns, the check fails
    """
    train_words = [(form, upos, lemma[:-1] if form.endswith(("dora", "doras")) and upos == "NOUN" else lemma, feats)
                   for form, upos, lemma, feats in TRAIN_WORDS]
    train_file = write_words(tmp_path / "train.conllu", train_words)
    dict_file = write_words(tmp_path / "dict.conllu", [(form, upos, lemma, "_") for form, upos, lemma in DICTIONARY])
    with pytest.raises(ValueError, match="-dora nouns have the -dora lemma .* agrees on 0 of 2 words.*investigadoras \\(investigador / investigadora\\)"):
        convert_es_freeling.check_conventions(train_file, dict_file)

    # with a lower threshold, the same data passes
    convert_es_freeling.check_conventions(train_file, dict_file, min_agreement=0.0)

def test_check_conventions_no_words(tmp_path):
    """
    A convention with no words to compare fails, as there is nothing to show it still holds
    """
    train_file = write_words(tmp_path / "train.conllu", [x for x in TRAIN_WORDS if "NumType=Ord" not in x[3]])
    dict_file = write_words(tmp_path / "dict.conllu", [(form, upos, lemma, "_") for form, upos, lemma in DICTIONARY])
    with pytest.raises(ValueError, match="ordinals .* agrees on 0 of 0 words"):
        convert_es_freeling.check_conventions(train_file, dict_file)

def test_prepare_check():
    assert prepare_lemma_treebank.EXTERNAL_DICTIONARY_CHECKS["es_combined"] is convert_es_freeling.check_conventions
