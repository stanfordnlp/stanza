"""
Test some of the methods in the vi_quad dataset

Uses a small section of the dataset as a test
"""

import pytest

from stanza.utils.datasets.constituency import selftrain_vi_quad

from stanza.tests import *

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

SAMPLE_TEXT = """
{"version": "1.1", "data": [{"title": "Ph\u1ea1m V\u0103n \u0110\u1ed3ng", "paragraphs": [{"qas": [{"question": "T\u00ean g\u1ecdi n\u00e0o \u0111\u01b0\u1ee3c Ph\u1ea1m V\u0103n \u0110\u1ed3ng s\u1eed d\u1ee5ng khi l\u00e0m Ph\u00f3 ch\u1ee7 nhi\u1ec7m c\u01a1 quan Bi\u1ec7n s\u1ef1 x\u1ee9 t\u1ea1i Qu\u1ebf L\u00e2m?", "answers": [{"answer_start": 507, "text": "L\u00e2m B\u00e1 Ki\u1ec7t"}], "id": "uit_01__05272_0_1"}, {"question": "Ph\u1ea1m V\u0103n \u0110\u1ed3ng gi\u1eef ch\u1ee9c v\u1ee5 g\u00ec trong b\u1ed9 m\u00e1y Nh\u00e0 n\u01b0\u1edbc C\u1ed9ng h\u00f2a X\u00e3 h\u1ed9i ch\u1ee7 ngh\u0129a Vi\u1ec7t Nam?", "answers": [{"answer_start": 60, "text": "Th\u1ee7 t\u01b0\u1edbng"}], "id": "uit_01__05272_0_2"}, {"question": "Giai \u0111o\u1ea1n n\u0103m 1955-1976, Ph\u1ea1m V\u0103n \u0110\u1ed3ng n\u1eafm gi\u1eef ch\u1ee9c v\u1ee5 g\u00ec?", "answers": [{"answer_start": 245, "text": "Th\u1ee7 t\u01b0\u1edbng Ch\u00ednh ph\u1ee7 Vi\u1ec7t Nam D\u00e2n ch\u1ee7 C\u1ed9ng h\u00f2a"}], "id": "uit_01__05272_0_3"}], "context": "Ph\u1ea1m V\u0103n \u0110\u1ed3ng (1 th\u00e1ng 3 n\u0103m 1906 \u2013 29 th\u00e1ng 4 n\u0103m 2000) l\u00e0 Th\u1ee7 t\u01b0\u1edbng \u0111\u1ea7u ti\u00ean c\u1ee7a n\u01b0\u1edbc C\u1ed9ng h\u00f2a X\u00e3 h\u1ed9i ch\u1ee7 ngh\u0129a Vi\u1ec7t Nam t\u1eeb n\u0103m 1976 (t\u1eeb n\u0103m 1981 g\u1ecdi l\u00e0 Ch\u1ee7 t\u1ecbch H\u1ed9i \u0111\u1ed3ng B\u1ed9 tr\u01b0\u1edfng) cho \u0111\u1ebfn khi ngh\u1ec9 h\u01b0u n\u0103m 1987. Tr\u01b0\u1edbc \u0111\u00f3 \u00f4ng t\u1eebng gi\u1eef ch\u1ee9c v\u1ee5 Th\u1ee7 t\u01b0\u1edbng Ch\u00ednh ph\u1ee7 Vi\u1ec7t Nam D\u00e2n ch\u1ee7 C\u1ed9ng h\u00f2a t\u1eeb n\u0103m 1955 \u0111\u1ebfn n\u0103m 1976. \u00d4ng l\u00e0 v\u1ecb Th\u1ee7 t\u01b0\u1edbng Vi\u1ec7t Nam t\u1ea1i v\u1ecb l\u00e2u nh\u1ea5t (1955\u20131987). \u00d4ng l\u00e0 h\u1ecdc tr\u00f2, c\u1ed9ng s\u1ef1 c\u1ee7a Ch\u1ee7 t\u1ecbch H\u1ed3 Ch\u00ed Minh. \u00d4ng c\u00f3 t\u00ean g\u1ecdi th\u00e2n m\u1eadt l\u00e0 T\u00f4, \u0111\u00e2y t\u1eebng l\u00e0 b\u00ed danh c\u1ee7a \u00f4ng. \u00d4ng c\u00f2n c\u00f3 t\u00ean g\u1ecdi l\u00e0 L\u00e2m B\u00e1 Ki\u1ec7t khi l\u00e0m Ph\u00f3 ch\u1ee7 nhi\u1ec7m c\u01a1 quan Bi\u1ec7n s\u1ef1 x\u1ee9 t\u1ea1i Qu\u1ebf L\u00e2m (Ch\u1ee7 nhi\u1ec7m l\u00e0 H\u1ed3 H\u1ecdc L\u00e3m)."}, {"qas": [{"question": "S\u1ef1 ki\u1ec7n quan tr\u1ecdng n\u00e0o \u0111\u00e3 di\u1ec5n ra v\u00e0o ng\u00e0y 20/7/1954?", "answers": [{"answer_start": 364, "text": "b\u1ea3n Hi\u1ec7p \u0111\u1ecbnh \u0111\u00ecnh ch\u1ec9 chi\u1ebfn s\u1ef1 \u1edf Vi\u1ec7t Nam, Campuchia v\u00e0 L\u00e0o \u0111\u00e3 \u0111\u01b0\u1ee3c k\u00fd k\u1ebft th\u1eeba nh\u1eadn t\u00f4n tr\u1ecdng \u0111\u1ed9c l\u1eadp, ch\u1ee7 quy\u1ec1n, c\u1ee7a n\u01b0\u1edbc Vi\u1ec7t Nam, L\u00e0o v\u00e0 Campuchia"}], "id": "uit_01__05272_1_1"}, {"question": "Ch\u1ee9c v\u1ee5 m\u00e0 Ph\u1ea1m V\u0103n \u0110\u1ed3ng \u0111\u1ea3m nhi\u1ec7m t\u1ea1i H\u1ed9i ngh\u1ecb Gen\u00e8ve v\u1ec1 \u0110\u00f4ng D\u01b0\u01a1ng?", "answers": [{"answer_start": 33, "text": "Tr\u01b0\u1edfng ph\u00e1i \u0111o\u00e0n Ch\u00ednh ph\u1ee7"}], "id": "uit_01__05272_1_2"}, {"question": "H\u1ed9i ngh\u1ecb Gen\u00e8ve v\u1ec1 \u0110\u00f4ng D\u01b0\u01a1ng c\u00f3 t\u00ednh ch\u1ea5t nh\u01b0 th\u1ebf n\u00e0o?", "answers": [{"answer_start": 262, "text": "r\u1ea5t c\u0103ng th\u1eb3ng v\u00e0 ph\u1ee9c t\u1ea1p"}], "id": "uit_01__05272_1_3"}]}]}]}
"""

EXPECTED = ['Tên gọi nào được Phạm Văn Đồng sử dụng khi làm Phó chủ nhiệm cơ quan Biện sự xứ tại Quế Lâm?', 'Phạm Văn Đồng giữ chức vụ gì trong bộ máy Nhà nước Cộng hòa Xã hội chủ nghĩa Việt Nam?', 'Giai đoạn năm 1955-1976, Phạm Văn Đồng nắm giữ chức vụ gì?', 'Sự kiện quan trọng nào đã diễn ra vào ngày 20/7/1954?', 'Chức vụ mà Phạm Văn Đồng đảm nhiệm tại Hội nghị Genève về Đông Dương?', 'Hội nghị Genève về Đông Dương có tính chất như thế nào?']

def test_read_file():
    results = selftrain_vi_quad.parse_quad(SAMPLE_TEXT)
    assert results == EXPECTED
