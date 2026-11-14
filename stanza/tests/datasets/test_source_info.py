"""
Tests for the comments recording which version of a lexicon a lemma dictionary came from
"""

import datetime
import hashlib
import os

import pytest

from stanza.utils.datasets.lemma import source_info

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

# 2026-01-15 12:00 UTC
NOON_JAN_15 = datetime.datetime(2026, 1, 15, 12, tzinfo=datetime.timezone.utc).timestamp()
# 2026-03-02 12:00 UTC
NOON_MAR_2 = datetime.datetime(2026, 3, 2, 12, tzinfo=datetime.timezone.utc).timestamp()

def write_files(tmp_path):
    first = tmp_path / "a.xml"
    first.write_bytes(b"first")
    second = tmp_path / "b.xml"
    second.write_bytes(b"second")
    os.utime(first, (NOON_JAN_15, NOON_JAN_15))
    os.utime(second, (NOON_MAR_2, NOON_MAR_2))
    return str(first), str(second)

def test_files_sha256(tmp_path):
    first, second = write_files(tmp_path)
    expected = hashlib.sha256(b"a.xml" + b"first" + b"b.xml" + b"second").hexdigest()
    assert source_info.files_sha256([first, second]) == expected
    # the order of the files matters
    assert source_info.files_sha256([second, first]) != expected

def test_modified_date(tmp_path):
    first, second = write_files(tmp_path)
    assert source_info.modified_date([first]) == "2026-01-15"
    # the latest of the files
    assert source_info.modified_date([first, second]) == "2026-03-02"
    assert source_info.modified_date([]) is None

def test_file_comments(tmp_path):
    first, second = write_files(tmp_path)
    assert source_info.file_comments("sloleks", [first, second]) == [
        "# sloleks_sha256 = %s" % source_info.files_sha256([first, second]),
        "# sloleks_modified = 2026-03-02",
    ]
    assert source_info.file_comments("sloleks", []) == ["# sloleks_sha256 = %s" % hashlib.sha256().hexdigest()]
