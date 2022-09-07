import os
import tempfile

import pytest

import xml.etree.ElementTree as ET

import stanza
from stanza.tests import *

from stanza.utils.datasets.common import convert_conllu_to_txt
from stanza.utils.datasets.tokenization.convert_th_orchid import parse_xml
from stanza.utils.datasets.tokenization.process_thai_tokenization import write_section

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]


SMALL_DOC="""
<corpus>
<document TPublisher="ศูนย์เทคโนโลยีอิเล็กทรอนิกส์และคอมพิวเตอร์แห่งชาติ, กระทรวงวิทยาศาสตร์ เทคโนโลยีและการพลังงาน" EPublisher="National Electronics and Computer Technology Center, Ministry of Science, Technology and Energy" TInbook="การประชุมทางวิชาการ ครั้งที่ 1, โครงการวิจัยและพัฒนาอิเล็กทรอนิกส์และคอมพิวเตอร์, ปีงบประมาณ 2531, เล่ม 1" TTitle="การประชุมทางวิชาการ ครั้งที่ 1" Year="1989" EInbook="The 1st Annual Conference, Electronics and Computer Research and Development Project, Fiscal Year 1988, Book 1" ETitle="[1st Annual Conference]">
<paragraph id="1" line_num="12">
<sentence id="1" line_num = "13" raw_txt = "การประชุมทางวิชาการ ครั้งที่ 1">
<word surface="การ" pos="FIXN"/>
<word surface="ประชุม" pos="VACT"/>
<word surface="ทาง" pos="NCMN"/>
<word surface="วิชาการ" pos="NCMN"/>
<word surface="&lt;space&gt;" pos="PUNC"/>
<word surface="ครั้ง" pos="CFQC"/>
<word surface="ที่ 1" pos="DONM"/>
</sentence>
<sentence id="2" line_num = "23" raw_txt = "โครงการวิจัยและพัฒนาอิเล็กทรอนิกส์และคอมพิวเตอร์">
<word surface="โครงการวิจัยและพัฒนา" pos="NCMN"/>
<word surface="อิเล็กทรอนิกส์" pos="NCMN"/>
<word surface="และ" pos="JCRG"/>
<word surface="คอมพิวเตอร์" pos="NCMN"/>
</sentence>
</paragraph>
<paragraph id="3" line_num="51">
<sentence id="1" line_num = "52" raw_txt = "วันที่ 15-16 สิงหาคม 2532">
<word surface="วัน" pos="NCMN"/>
<word surface="ที่ 15" pos="DONM"/>
<word surface="&lt;minus&gt;" pos="PUNC"/>
<word surface="16" pos="DONM"/>
<word surface="&lt;space&gt;" pos="PUNC"/>
<word surface="สิงหาคม" pos="NCMN"/>
<word surface="&lt;space&gt;" pos="PUNC"/>
<word surface="2532" pos="NCNM"/>
</sentence>
</paragraph>
</document>
</corpus>
"""


EXPECTED_RESULTS="""
1	การ	_	_	_	_	0	root	0:root	SpaceAfter=No|NewPar=Yes
2	ประชุม	_	_	_	_	1	dep	1:dep	SpaceAfter=No
3	ทาง	_	_	_	_	2	dep	2:dep	SpaceAfter=No
4	วิชาการ	_	_	_	_	3	dep	3:dep	_
5	ครั้ง	_	_	_	_	4	dep	4:dep	SpaceAfter=No
6	ที่ 1	_	_	_	_	5	dep	5:dep	_

1	โครงการวิจัยและพัฒนา	_	_	_	_	0	root	0:root	SpaceAfter=No
2	อิเล็กทรอนิกส์	_	_	_	_	1	dep	1:dep	SpaceAfter=No
3	และ	_	_	_	_	2	dep	2:dep	SpaceAfter=No
4	คอมพิวเตอร์	_	_	_	_	3	dep	3:dep	_

1	วัน	_	_	_	_	0	root	0:root	SpaceAfter=No|NewPar=Yes
2	ที่ 15	_	_	_	_	1	dep	1:dep	SpaceAfter=No
3	-	_	_	_	_	2	dep	2:dep	SpaceAfter=No
4	16	_	_	_	_	3	dep	3:dep	_
5	สิงหาคม	_	_	_	_	4	dep	4:dep	_
6	2532	_	_	_	_	5	dep	5:dep	_
""".strip()

EXPECTED_TEXT="""การประชุมทางวิชาการ ครั้งที่ 1 โครงการวิจัยและพัฒนาอิเล็กทรอนิกส์และคอมพิวเตอร์

วันที่ 15-16 สิงหาคม 2532

"""

EXPECTED_LABELS="""0010000010010000001000001000020000000000000000000010000000000000100100000000002

0010000011010000000100002

"""

def check_results(documents, expected_conllu, expected_txt, expected_labels):
    with tempfile.TemporaryDirectory() as output_dir:
        write_section(output_dir, "orchid", "train", documents)
        with open(os.path.join(output_dir, "th_orchid.train.gold.conllu")) as fin:
            conllu = fin.read().strip()
        with open(os.path.join(output_dir, "th_orchid.train.txt")) as fin:
            txt = fin.read()
        with open(os.path.join(output_dir, "th_orchid-ud-train.toklabels")) as fin:
            labels = fin.read()
        assert conllu == expected_conllu
        assert txt == expected_txt
        assert labels == expected_labels

        assert len(txt) == len(labels)

def test_orchid():
    tree = ET.ElementTree(ET.fromstring(SMALL_DOC))
    documents = parse_xml(tree)
    check_results(documents, EXPECTED_RESULTS, EXPECTED_TEXT, EXPECTED_LABELS)

