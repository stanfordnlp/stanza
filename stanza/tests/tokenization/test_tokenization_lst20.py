import os
import tempfile

import pytest

import stanza
from stanza.tests import *

from stanza.utils.datasets.common import convert_conllu_to_txt
from stanza.utils.datasets.tokenization.convert_th_lst20 import read_document
from stanza.utils.datasets.tokenization.process_thai_tokenization import write_section

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

SMALL_LST_SAMPLE="""
สุรยุทธ์	NN	B_PER	B_CLS
ยัน	VV	O	I_CLS
ปฏิเสธ	VV	O	I_CLS
ลงนาม	VV	O	I_CLS
_	PU	O	I_CLS
MOU	NN	O	I_CLS
_	PU	O	I_CLS
กับ	PS	O	I_CLS
อียู	NN	B_ORG	I_CLS
ไม่	NG	O	I_CLS
กระทบ	VV	O	I_CLS
สัมพันธ์	NN	O	E_CLS

1	NU	B_DTM	B_CLS
_	PU	I_DTM	I_CLS
กันยายน	NN	I_DTM	I_CLS
_	PU	I_DTM	I_CLS
2550	NU	E_DTM	I_CLS
_	PU	O	I_CLS
12:21	NU	B_DTM	I_CLS
_	PU	I_DTM	I_CLS
น.	CL	E_DTM	E_CLS

ผู้สื่อข่าว	NN	O	B_CLS
รายงาน	VV	O	I_CLS
เพิ่มเติม	VV	O	I_CLS
ว่า	CC	O	E_CLS
_	PU	O	O
จาก	PS	O	B_CLS
การ	FX	O	I_CLS
ลง	VV	O	I_CLS
พื้นที่	NN	O	I_CLS
พบ	VV	O	I_CLS
ว่า	CC	O	E_CLS
""".strip()

EXPECTED_CONLLU="""
1	สุรยุทธ์	_	_	_	_	0	root	0:root	SpaceAfter=No|NewPar=Yes
2	ยัน	_	_	_	_	1	dep	1:dep	SpaceAfter=No
3	ปฏิเสธ	_	_	_	_	2	dep	2:dep	SpaceAfter=No
4	ลงนาม	_	_	_	_	3	dep	3:dep	_
5	MOU	_	_	_	_	4	dep	4:dep	_
6	กับ	_	_	_	_	5	dep	5:dep	SpaceAfter=No
7	อียู	_	_	_	_	6	dep	6:dep	SpaceAfter=No
8	ไม่	_	_	_	_	7	dep	7:dep	SpaceAfter=No
9	กระทบ	_	_	_	_	8	dep	8:dep	SpaceAfter=No
10	สัมพันธ์	_	_	_	_	9	dep	9:dep	SpaceAfter=No

1	1	_	_	_	_	0	root	0:root	_
2	กันยายน	_	_	_	_	1	dep	1:dep	_
3	2550	_	_	_	_	2	dep	2:dep	_
4	12:21	_	_	_	_	3	dep	3:dep	_
5	น.	_	_	_	_	4	dep	4:dep	SpaceAfter=No

1	ผู้สื่อข่าว	_	_	_	_	0	root	0:root	SpaceAfter=No
2	รายงาน	_	_	_	_	1	dep	1:dep	SpaceAfter=No
3	เพิ่มเติม	_	_	_	_	2	dep	2:dep	SpaceAfter=No
4	ว่า	_	_	_	_	3	dep	3:dep	_
5	จาก	_	_	_	_	4	dep	4:dep	SpaceAfter=No
6	การ	_	_	_	_	5	dep	5:dep	SpaceAfter=No
7	ลง	_	_	_	_	6	dep	6:dep	SpaceAfter=No
8	พื้นที่	_	_	_	_	7	dep	7:dep	SpaceAfter=No
9	พบ	_	_	_	_	8	dep	8:dep	SpaceAfter=No
10	ว่า	_	_	_	_	9	dep	9:dep	SpaceAfter=No
""".strip()

# Note: these DO NOT line up perfectly (in an emacs window, at least)
# because Thai characters have a length greater than 1.
# The lengths of the words are:
#   สุรยุทธ์    8
#      ยัน    3
#   ปฏิเสธ    6
#   ลงนาม    5
#     MOU    3
#      กับ    3
#      อียู    4
#      ไม่    3
#   กระทบ    5
#   สัมพันธ์    8
#       1    1
#  กันยายน    7
#    2550    4
#   12:21    5
#      น.    2
#  ผู้สื่อข่าว   11
#  รายงาน    6
#  เพิ่มเติม    9
#      ว่า    3
#     จาก    3
#     การ    3
#      ลง    2
#     พื้นที่    7
#      พบ    2
#      ว่า    3
EXPECTED_TXT    =   "สุรยุทธ์ยันปฏิเสธลงนาม MOU กับอียูไม่กระทบสัมพันธ์1 กันยายน 2550 12:21 น.ผู้สื่อข่าวรายงานเพิ่มเติมว่า จากการลงพื้นที่พบว่า\n\n"
EXPECTED_LABELS =   "000000010010000010000100010001000100100001000000021000000010000100000100200000000001000001000000001001000100101000000101002\n\n"
# counting spaces    1234567812312345612345_123_123123412312345123456781_1234567_1234_12345_12123456789AB123456123456789123_12312312123456712123

# note that the word splits go on the final letter of the word in the
# UD conllu datasets, so that is what we mimic here
# for example, from EWT:
# Al-Zaman : American forces killed Shaikh Abdullah
# 0110000101000000001000000100000010000001000000001

def check_results(documents, expected_conllu, expected_txt, expected_labels):
    with tempfile.TemporaryDirectory() as output_dir:
        write_section(output_dir, "lst20", "train", documents)
        with open(os.path.join(output_dir, "th_lst20.train.gold.conllu")) as fin:
            conllu = fin.read().strip()
        with open(os.path.join(output_dir, "th_lst20.train.txt")) as fin:
            txt = fin.read()
        with open(os.path.join(output_dir, "th_lst20-ud-train.toklabels")) as fin:
            labels = fin.read()
        assert conllu == expected_conllu
        assert txt == expected_txt
        assert labels == expected_labels

        assert len(txt) == len(labels)


def test_small():
    """
    A small test just to verify that the output is being produced as we want

    Note that there currently are no spaces after the first sentence.
    Apparently this is wrong, but weirdly, doing that makes the model even worse.
    """
    lines = SMALL_LST_SAMPLE.strip().split("\n")
    documents = read_document(lines, spaces_after=False, split_clauses=False)
    check_results(documents, EXPECTED_CONLLU, EXPECTED_TXT, EXPECTED_LABELS)

EXPECTED_SPACE_CONLLU="""
1	สุรยุทธ์	_	_	_	_	0	root	0:root	SpaceAfter=No|NewPar=Yes
2	ยัน	_	_	_	_	1	dep	1:dep	SpaceAfter=No
3	ปฏิเสธ	_	_	_	_	2	dep	2:dep	SpaceAfter=No
4	ลงนาม	_	_	_	_	3	dep	3:dep	_
5	MOU	_	_	_	_	4	dep	4:dep	_
6	กับ	_	_	_	_	5	dep	5:dep	SpaceAfter=No
7	อียู	_	_	_	_	6	dep	6:dep	SpaceAfter=No
8	ไม่	_	_	_	_	7	dep	7:dep	SpaceAfter=No
9	กระทบ	_	_	_	_	8	dep	8:dep	SpaceAfter=No
10	สัมพันธ์	_	_	_	_	9	dep	9:dep	_

1	1	_	_	_	_	0	root	0:root	_
2	กันยายน	_	_	_	_	1	dep	1:dep	_
3	2550	_	_	_	_	2	dep	2:dep	_
4	12:21	_	_	_	_	3	dep	3:dep	_
5	น.	_	_	_	_	4	dep	4:dep	_

1	ผู้สื่อข่าว	_	_	_	_	0	root	0:root	SpaceAfter=No
2	รายงาน	_	_	_	_	1	dep	1:dep	SpaceAfter=No
3	เพิ่มเติม	_	_	_	_	2	dep	2:dep	SpaceAfter=No
4	ว่า	_	_	_	_	3	dep	3:dep	_
5	จาก	_	_	_	_	4	dep	4:dep	SpaceAfter=No
6	การ	_	_	_	_	5	dep	5:dep	SpaceAfter=No
7	ลง	_	_	_	_	6	dep	6:dep	SpaceAfter=No
8	พื้นที่	_	_	_	_	7	dep	7:dep	SpaceAfter=No
9	พบ	_	_	_	_	8	dep	8:dep	SpaceAfter=No
10	ว่า	_	_	_	_	9	dep	9:dep	_
""".strip()

EXPECTED_SPACE_TXT    =   "สุรยุทธ์ยันปฏิเสธลงนาม MOU กับอียูไม่กระทบสัมพันธ์ 1 กันยายน 2550 12:21 น. ผู้สื่อข่าวรายงานเพิ่มเติมว่า จากการลงพื้นที่พบว่า\n\n"
EXPECTED_SPACE_LABELS =   "00000001001000001000010001000100010010000100000002010000000100001000001002000000000001000001000000001001000100101000000101002\n\n"

def test_space_after():
    """
    This version of the test adds the space after attribute
    """
    lines = SMALL_LST_SAMPLE.strip().split("\n")
    documents = read_document(lines, spaces_after=True, split_clauses=False)
    check_results(documents, EXPECTED_SPACE_CONLLU, EXPECTED_SPACE_TXT, EXPECTED_SPACE_LABELS)


EXPECTED_CLAUSE_CONLLU="""
1	สุรยุทธ์	_	_	_	_	0	root	0:root	SpaceAfter=No|NewPar=Yes
2	ยัน	_	_	_	_	1	dep	1:dep	SpaceAfter=No
3	ปฏิเสธ	_	_	_	_	2	dep	2:dep	SpaceAfter=No
4	ลงนาม	_	_	_	_	3	dep	3:dep	_
5	MOU	_	_	_	_	4	dep	4:dep	_
6	กับ	_	_	_	_	5	dep	5:dep	SpaceAfter=No
7	อียู	_	_	_	_	6	dep	6:dep	SpaceAfter=No
8	ไม่	_	_	_	_	7	dep	7:dep	SpaceAfter=No
9	กระทบ	_	_	_	_	8	dep	8:dep	SpaceAfter=No
10	สัมพันธ์	_	_	_	_	9	dep	9:dep	_

1	1	_	_	_	_	0	root	0:root	_
2	กันยายน	_	_	_	_	1	dep	1:dep	_
3	2550	_	_	_	_	2	dep	2:dep	_
4	12:21	_	_	_	_	3	dep	3:dep	_
5	น.	_	_	_	_	4	dep	4:dep	_

1	ผู้สื่อข่าว	_	_	_	_	0	root	0:root	SpaceAfter=No
2	รายงาน	_	_	_	_	1	dep	1:dep	SpaceAfter=No
3	เพิ่มเติม	_	_	_	_	2	dep	2:dep	SpaceAfter=No
4	ว่า	_	_	_	_	3	dep	3:dep	_

1	จาก	_	_	_	_	0	root	0:root	SpaceAfter=No
2	การ	_	_	_	_	1	dep	1:dep	SpaceAfter=No
3	ลง	_	_	_	_	2	dep	2:dep	SpaceAfter=No
4	พื้นที่	_	_	_	_	3	dep	3:dep	SpaceAfter=No
5	พบ	_	_	_	_	4	dep	4:dep	SpaceAfter=No
6	ว่า	_	_	_	_	5	dep	5:dep	_
""".strip()

EXPECTED_CLAUSE_TXT    =   "สุรยุทธ์ยันปฏิเสธลงนาม MOU กับอียูไม่กระทบสัมพันธ์ 1 กันยายน 2550 12:21 น. ผู้สื่อข่าวรายงานเพิ่มเติมว่า จากการลงพื้นที่พบว่า\n\n"
EXPECTED_CLAUSE_LABELS =   "00000001001000001000010001000100010010000100000002010000000100001000001002000000000001000001000000001002000100101000000101002\n\n"


def test_split_clause():
    """
    This version of the test also resplits on spaces between clauses
    """
    lines = SMALL_LST_SAMPLE.strip().split("\n")
    documents = read_document(lines, spaces_after=True, split_clauses=True)
    check_results(documents, EXPECTED_CLAUSE_CONLLU, EXPECTED_CLAUSE_TXT, EXPECTED_CLAUSE_LABELS)

if __name__ == "__main__":
    lines = SMALL_LST_SAMPLE.strip().split("\n")
    documents = read_document(lines, spaces_after=False, split_clauses=False)

    write_section("foo", "lst20", "train", documents)
