"""
Tests for converting the Sloleks lexicon to a lemma dictionary
"""

import pytest

from stanza.utils.conll import CoNLL
from stanza.utils.datasets.lemma import convert_sl_sloleks

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

# A reduced version of the Sloleks 3.1 XML, with only the pieces the converter reads
SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<lexicon>
<entry>
  <head>
    <headword><lemma>miza</lemma></headword>
    <grammar><category>noun</category></grammar>
  </head>
  <body>
    <wordForm><orthographyList><orthography norm="standard"><form>miza</form></orthography></orthographyList></wordForm>
    <wordForm><orthographyList><orthography norm="standard"><form>mize</form></orthography></orthographyList></wordForm>
    <wordForm><orthographyList><orthography norm="standard"><form>mizi</form></orthography></orthographyList></wordForm>
  </body>
</entry>
<entry>
  <head>
    <headword><lemma>mizen</lemma></headword>
    <grammar><category>adjective</category></grammar>
  </head>
  <body>
    <wordForm><orthographyList><orthography norm="standard"><form>mizni</form></orthography></orthographyList></wordForm>
  </body>
</entry>
<entry>
  <head>
    <headword><lemma>mizi</lemma></headword>
    <grammar><category>noun</category></grammar>
  </head>
  <body>
    <wordForm><orthographyList><orthography norm="standard"><form>mizi</form></orthography></orthographyList></wordForm>
  </body>
</entry>
</lexicon>
"""

def test_sent_ids(tmp_path):
    sloleks_dir = tmp_path / "Sloleks.3.1"
    sloleks_dir.mkdir()
    (sloleks_dir / "sloleks_3.1_001.xml").write_text(SAMPLE, encoding="utf-8")
    output_file = tmp_path / "sl_combined.dict.conllu"
    convert_sl_sloleks.convert_directory(str(sloleks_dir), str(output_file))

    doc = CoNLL.conll2doc(input_file=str(output_file))
    sent_ids = [sentence.sent_id for sentence in doc.sentences]
    # mizi appears twice as a NOUN, with two different lemmas
    assert sent_ids == ["sl_sloleks-NOUN-miza-miza",
                        "sl_sloleks-NOUN-mize-miza",
                        "sl_sloleks-NOUN-mizi-miza",
                        "sl_sloleks-ADJ-mizni-mizen",
                        "sl_sloleks-NOUN-mizi-mizi"]

def test_repeated_sent_ids():
    build_id = convert_sl_sloleks.sent_ids()
    assert build_id("a-b", "NOUN", "c") == "sl_sloleks-NOUN-a-b-c"
    assert build_id("a", "NOUN", "b-c") == "sl_sloleks-NOUN-a-b-c-2"
    assert build_id("a b", "NOUN", "a b") == "sl_sloleks-NOUN-a_b-a_b"

def test_missing_sloleks(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_sl_sloleks.build_dictionary({"STANZA_EXTERN_DIR": str(tmp_path)}, str(tmp_path / "out.conllu"))
