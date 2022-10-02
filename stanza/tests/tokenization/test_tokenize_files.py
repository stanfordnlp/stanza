import pytest

from stanza.models.tokenization import tokenize_files
from stanza.tests import TEST_MODELS_DIR

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

EXPECTED = """
This is a test . This is a second sentence .
I took my daughter ice skating
""".lstrip()

def test_tokenize_files(tmp_path):
    input_file = tmp_path / "input.txt"
    with open(input_file, "w") as fout:
        fout.write("This is a test.  This is a second sentence.\n\nI took my daughter ice skating")

    output_file = tmp_path / "output.txt"
    tokenize_files.main([str(input_file), "--lang", "en", "--output_file", str(output_file), "--model_dir", TEST_MODELS_DIR])

    with open(output_file) as fin:
        text = fin.read()

    assert EXPECTED == text
