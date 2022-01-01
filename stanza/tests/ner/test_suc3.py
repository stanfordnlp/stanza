"""
Tests the conversion code for the SUC3 NER dataset
"""

import os
import tempfile
from zipfile import ZipFile

import pytest
pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

import stanza.utils.datasets.ner.suc_conll_to_iob as suc_conll_to_iob

TEST_CONLL = """
1	Den	den	PN	PN	UTR|SIN|DEF|SUB/OBJ	_	_	_	_	O	_	ac01b-030:2328
2	Gud	Gud	PM	PM	NOM	_	_	_	_	B	myth	ac01b-030:2329
3	giver	giva	VB	VB	PRS|AKT	_	_	_	_	O	_	ac01b-030:2330
4	ämbetet	ämbete	NN	NN	NEU|SIN|DEF|NOM	_	_	_	_	O	_	ac01b-030:2331
5	får	få	VB	VB	PRS|AKT	_	_	_	_	O	_	ac01b-030:2332
6	också	också	AB	AB		_	_	_	_	O	_	ac01b-030:2333
7	förståndet	förstånd	NN	NN	NEU|SIN|DEF|NOM	_	_	_	_	O	_	ac01b-030:2334
8	.	.	MAD	MAD		_	_	_	_	O	_	ac01b-030:2335

1	Han	han	PN	PN	UTR|SIN|DEF|SUB	_	_	_	_	O	_	aa01a-017:227
2	berättar	berätta	VB	VB	PRS|AKT	_	_	_	_	O	_	aa01a-017:228
3	anekdoten	anekdot	NN	NN	UTR|SIN|DEF|NOM	_	_	_	_	O	_	aa01a-017:229
4	som	som	HP	HP	-|-|-	_	_	_	_	O	_	aa01a-017:230
5	FN-medlaren	FN-medlare	NN	NN	UTR|SIN|DEF|NOM	_	_	_	_	O	_	aa01a-017:231
6	Brian	Brian	PM	PM	NOM	_	_	_	_	B	person	aa01a-017:232
7	Urquhart	Urquhart	PM	PM	NOM	_	_	_	_	I	person	aa01a-017:233
8	myntat	mynta	VB	VB	SUP|AKT	_	_	_	_	O	_	aa01a-017:234
9	:	:	MAD	MAD		_	_	_	_	O	_	aa01a-017:235
"""

EXPECTED_IOB = """
Den	O
Gud	B-myth
giver	O
ämbetet	O
får	O
också	O
förståndet	O
.	O

Han	O
berättar	O
anekdoten	O
som	O
FN-medlaren	O
Brian	B-person
Urquhart	I-person
myntat	O
:	O
"""

def test_read_zip():
    """
    Test creating a fake zip file, then converting it to an .iob file
    """
    with tempfile.TemporaryDirectory() as tempdir:
        zip_name = os.path.join(tempdir, "test.zip")
        in_filename = "conll"
        with ZipFile(zip_name, "w") as zout:
            with zout.open(in_filename, "w") as fout:
                fout.write(TEST_CONLL.encode())

        out_filename = "iob"
        num = suc_conll_to_iob.extract_from_zip(zip_name, in_filename, out_filename)
        assert num == 2

        with open(out_filename) as fin:
            result = fin.read()
        assert EXPECTED_IOB.strip() == result.strip()

def test_read_raw():
    """
    Test a direct text file conversion w/o the zip file
    """
    with tempfile.TemporaryDirectory() as tempdir:
        in_filename = os.path.join(tempdir, "test.txt")
        with open(in_filename, "w", encoding="utf-8") as fout:
            fout.write(TEST_CONLL)

        out_filename = "iob"
        with open(in_filename, encoding="utf-8") as fin, open(out_filename, "w", encoding="utf-8") as fout:
            num = suc_conll_to_iob.extract(fin, fout)
        assert num == 2

        with open(out_filename) as fin:
            result = fin.read()
        assert EXPECTED_IOB.strip() == result.strip()
