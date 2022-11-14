import pytest
import os

from stanza.utils.datasets.vietnamese import renormalize

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def test_replace_all():
    text     = "SỌAmple tụy test file"
    expected = "SOẠmple tuỵ test file"

    assert renormalize.replace_all(text) == expected

def test_replace_file(tmp_path):
    text     = "SỌAmple tụy test file"
    expected = "SOẠmple tuỵ test file"

    orig = tmp_path / "orig.txt"
    converted = tmp_path / "converted.txt"

    with open(orig, "w", encoding="utf-8") as fout:
        for i in range(10):
            fout.write(text)
            fout.write("\n")

    renormalize.convert_file(orig, converted)

    assert os.path.exists(converted)
    with open(converted, encoding="utf-8") as fin:
        lines = fin.readlines()

    assert len(lines) == 10
    for i in lines:
        assert i.strip() == expected
        
