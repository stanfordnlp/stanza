"""
Test some simple conversions of NER bio files
"""

import pytest

import json

from stanza.models.common.doc import Document
from stanza.utils.datasets.ner.prepare_ner_file import process_dataset

BIO_1 = """
Jennifer	B-PERSON
Sh'reyan	I-PERSON
has	O
lovely	O
antennae	O
""".strip()

BIO_2 = """
but	O
I	O
don't	O
like	O
the	O
way	O
Jennifer	B-PERSON
treated	O
Beckett	B-PERSON
on	O
the	O
Cerritos	B-LOCATION
""".strip()

def check_json_file(doc, raw_text, expected_sentences, expected_tokens):
    raw_sentences = raw_text.strip().split("\n\n")
    assert len(raw_sentences) == expected_sentences
    if isinstance(expected_tokens, int):
        expected_tokens = [expected_tokens]
    for raw_sentence, expected_len in zip(raw_sentences, expected_tokens):
        assert len(raw_sentence.strip().split("\n")) == expected_len

    assert len(doc.sentences) == expected_sentences
    for sentence, expected_len in zip(doc.sentences, expected_tokens):
        assert len(sentence.tokens) == expected_len
    for sentence, raw_sentence in zip(doc.sentences, raw_sentences):
        for token, line in zip(sentence.tokens, raw_sentence.strip().split("\n")):
            word, tag = line.strip().split()
            assert token.text == word
            assert token.ner == tag

def write_and_convert(tmp_path, raw_text):
    bio_file = tmp_path / "test.bio"
    with open(bio_file, "w", encoding="utf-8") as fout:
        fout.write(raw_text)

    json_file = tmp_path / "json.bio"
    process_dataset(bio_file, json_file)

    with open(json_file) as fin:
        doc = Document(json.load(fin))

    return doc

def run_test(tmp_path, raw_text, expected_sentences, expected_tokens):
    doc = write_and_convert(tmp_path, raw_text)
    check_json_file(doc, raw_text, expected_sentences, expected_tokens)

def test_simple(tmp_path):
    run_test(tmp_path, BIO_1, 1, 5)

def test_ner_at_end(tmp_path):
    run_test(tmp_path, BIO_2, 1, 12)

def test_two_sentences(tmp_path):
    raw_text = BIO_1 + "\n\n" + BIO_2
    run_test(tmp_path, raw_text, 2, [5, 12])
