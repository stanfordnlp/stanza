"""
Run through the various text processing methods for using the parser on text files / directories

Uses a simple tree where the parser should always get it right, but things could potentially go wrong
"""

import glob
import os
import pytest

from stanza import Pipeline

from stanza.models.constituency import text_processing
from stanza.models.constituency import tree_reader
from stanza.tests import TEST_MODELS_DIR

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

@pytest.fixture(scope="module")
def pipeline():
    return Pipeline(dir=TEST_MODELS_DIR, lang="en", processors="tokenize, pos, constituency", tokenize_pretokenized=True)

def test_read_tokenized_file(tmp_path):
    filename = str(tmp_path / "test_input.txt")
    with open(filename, "w") as fout:
        # test that the underscore token comes back with spaces
        fout.write("This is a_small test\nLine two\n")
    text, ids = text_processing.read_tokenized_file(filename)
    assert text == [['This', 'is', 'a small', 'test'], ['Line', 'two']]
    assert ids == [None, None]

def test_parse_tokenized_sentences(pipeline):
    con_processor = pipeline.processors["constituency"]
    model = con_processor._model
    args = model.args

    sentences = [["This", "is", "a", "test"]]
    trees = text_processing.parse_tokenized_sentences(args, model, [pipeline], sentences)
    predictions = [x.predictions for x in trees]
    assert len(predictions) == 1
    scored_trees = predictions[0]
    assert len(scored_trees) == 1
    result = "{}".format(scored_trees[0].tree)
    expected = "(ROOT (S (NP (DT This)) (VP (VBZ is) (NP (DT a) (NN test)))))"
    assert result == expected

def test_parse_text(tmp_path, pipeline):
    con_processor = pipeline.processors["constituency"]
    model = con_processor._model
    args = model.args

    raw_file = str(tmp_path / "test_input.txt")
    with open(raw_file, "w") as fout:
        fout.write("This is a test\nThis is another test\n")
    output_file = str(tmp_path / "test_output.txt")
    text_processing.parse_text(args, model, [pipeline], tokenized_file=raw_file, predict_file=output_file)

    trees = tree_reader.read_treebank(output_file)
    trees = ["{}".format(x) for x in trees]
    expected_trees = ["(ROOT (S (NP (DT This)) (VP (VBZ is) (NP (DT a) (NN test)))))",
                      "(ROOT (S (NP (DT This)) (VP (VBZ is) (NP (DT another) (NN test)))))"]
    assert trees == expected_trees

def test_parse_dir(tmp_path, pipeline):
    con_processor = pipeline.processors["constituency"]
    model = con_processor._model
    args = model.args

    raw_dir = str(tmp_path / "input")
    os.makedirs(raw_dir)
    raw_f1 = str(tmp_path / "input" / "f1.txt")
    raw_f2 = str(tmp_path / "input" / "f2.txt")
    output_dir = str(tmp_path / "output")

    with open(raw_f1, "w") as fout:
        fout.write("This is a test")
    with open(raw_f2, "w") as fout:
        fout.write("This is another test")

    text_processing.parse_dir(args, model, [pipeline], raw_dir, output_dir)
    output_files = sorted(glob.glob(os.path.join(output_dir, "*")))
    expected_trees = ["(ROOT (S (NP (DT This)) (VP (VBZ is) (NP (DT a) (NN test)))))",
                      "(ROOT (S (NP (DT This)) (VP (VBZ is) (NP (DT another) (NN test)))))"]
    for output_file, expected_tree in zip(output_files, expected_trees):
        trees = tree_reader.read_treebank(output_file)
        assert len(trees) == 1
        assert "{}".format(trees[0]) == expected_tree

def test_parse_text(tmp_path, pipeline):
    con_processor = pipeline.processors["constituency"]
    model = con_processor._model
    args = dict(model.args)

    model_path = con_processor._config['model_path']

    raw_file = str(tmp_path / "test_input.txt")
    with open(raw_file, "w") as fout:
        fout.write("This is a test\nThis is another test\n")
    output_file = str(tmp_path / "test_output.txt")

    args['tokenized_file'] = raw_file
    args['predict_file'] = output_file

    text_processing.load_model_parse_text(args, model_path, [pipeline])
    trees = tree_reader.read_treebank(output_file)
    trees = ["{}".format(x) for x in trees]
    expected_trees = ["(ROOT (S (NP (DT This)) (VP (VBZ is) (NP (DT a) (NN test)))))",
                      "(ROOT (S (NP (DT This)) (VP (VBZ is) (NP (DT another) (NN test)))))"]
    assert trees == expected_trees
