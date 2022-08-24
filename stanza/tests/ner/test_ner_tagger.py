"""
Basic testing of the NER tagger.
"""

import os
import pytest
import stanza

from stanza.tests import *
from stanza.models import ner_tagger
from stanza.models.ner.scorer import score_by_token, score_by_entity
from stanza.utils.confusion import confusion_to_macro_f1
import stanza.utils.datasets.ner.prepare_ner_file as prepare_ner_file
from stanza.utils.training.run_ner import build_pretrain_args

pytestmark = pytest.mark.pipeline

EN_DOC = "Chris Manning is a good man. He works in Stanford University."

EN_DOC_GOLD = """
<Span text=Chris Manning;type=PERSON;start_char=0;end_char=13>
<Span text=Stanford University;type=ORG;start_char=41;end_char=60>
""".strip()

EN_BIO = """
Chris B-PERSON
Manning E-PERSON
is O
a O
good O
man O
. O

He O
works O
in O
Stanford B-ORG
University E-ORG
. O
""".strip().replace(" ", "\t")

EN_EXPECTED_OUTPUT = """
Chris B-PERSON B-PERSON
Manning E-PERSON E-PERSON
is O O
a O O
good O O
man O O
. O O

He O O
works O O
in O O
Stanford B-ORG B-ORG
University E-ORG E-ORG
. O O
""".strip().replace(" ", "\t")


def test_ner():
    nlp = stanza.Pipeline(**{'processors': 'tokenize,ner', 'dir': TEST_MODELS_DIR, 'lang': 'en', 'logging_level': 'error'})
    doc = nlp(EN_DOC)
    assert EN_DOC_GOLD == '\n'.join([ent.pretty_print() for ent in doc.ents])

def test_evaluate(tmp_path):
    """
    This simple example should have a 1.0 f1 for the ontonote model
    """
    model_path = os.path.join(TEST_MODELS_DIR, "en", "ner", "ontonotes.pt")
    assert os.path.exists(model_path), "This model should be downloaded as part of setup.py"

    os.makedirs(tmp_path, exist_ok=True)

    test_bio_filename = tmp_path / "test.bio"
    test_json_filename = tmp_path / "test.json"
    test_output_filename = tmp_path / "output.bio"
    with open(test_bio_filename, "w", encoding="utf-8") as fout:
        fout.write(EN_BIO)

    prepare_ner_file.process_dataset(test_bio_filename, test_json_filename)

    args = ["--save_name", str(model_path),
            "--eval_file", str(test_json_filename),
            "--eval_output_file", str(test_output_filename),
            "--mode", "predict"]
    args = args + build_pretrain_args("en", "ontonotes", model_dir=TEST_MODELS_DIR)
    args = ner_tagger.parse_args(args=args)
    confusion = ner_tagger.evaluate(args)
    assert confusion_to_macro_f1(confusion) == pytest.approx(1.0)

    with open(test_output_filename, encoding="utf-8") as fin:
        results = fin.read().strip()

    assert results == EN_EXPECTED_OUTPUT
    print(results)
