"""
Add a simple test of the Ensemble's inference path

This just reuses one model several times - that should still check the main loop, at least
"""

import pytest

from stanza import Pipeline
from stanza.models.constituency import text_processing
from stanza.models.constituency import tree_reader
from stanza.models.constituency.ensemble import Ensemble, EnsembleTrainer
from stanza.models.constituency.text_processing import parse_tokenized_sentences

from stanza.tests import TEST_MODELS_DIR

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]


@pytest.fixture(scope="module")
def pipeline():
    return Pipeline(dir=TEST_MODELS_DIR, lang="en", processors="tokenize, pos, constituency", tokenize_pretokenized=True)

@pytest.fixture(scope="module")
def saved_ensemble(tmp_path_factory, pipeline):
    tmp_path = tmp_path_factory.mktemp("ensemble")

    # test the ensemble by reusing the same parser multiple times
    con_processor = pipeline.processors["constituency"]
    model = con_processor._model
    args = dict(model.args)
    foundation_cache = pipeline.foundation_cache

    model_path = con_processor._config['model_path']
    # reuse the same model 3 times just to make sure the code paths are working
    filenames = [model_path, model_path, model_path]

    ensemble = EnsembleTrainer.from_files(args, filenames, foundation_cache=foundation_cache)
    save_path = tmp_path / "ensemble.pt"

    ensemble.save(save_path)
    return ensemble, save_path, args, foundation_cache

def check_basic_predictions(trees):
    predictions = [x.predictions for x in trees]
    assert len(predictions) == 2
    assert all(len(x) == 1 for x in predictions)
    trees = [x[0].tree for x in predictions]
    result = ["{}".format(tree) for tree in trees]
    expected = ["(ROOT (S (NP (DT This)) (VP (VBZ is) (NP (DT a) (NN test)))))",
                "(ROOT (S (NP (DT This)) (VP (VBZ is) (NP (DT another) (NN test)))))"]
    assert result == expected

def test_ensemble_inference(pipeline):
    # test the ensemble by reusing the same parser multiple times
    con_processor = pipeline.processors["constituency"]
    model = con_processor._model
    args = dict(model.args)
    foundation_cache = pipeline.foundation_cache

    model_path = con_processor._config['model_path']
    # reuse the same model 3 times just to make sure the code paths are working
    filenames = [model_path, model_path, model_path]

    ensemble = EnsembleTrainer.from_files(args, filenames, foundation_cache=foundation_cache)
    ensemble = ensemble.model
    sentences = [["This", "is", "a", "test"], ["This", "is", "another", "test"]]
    trees = parse_tokenized_sentences(args, ensemble, [pipeline], sentences)
    check_basic_predictions(trees)

def test_ensemble_save(saved_ensemble):
    """
    Depending on the saved_ensemble fixture should be enough to ensure
    that the ensemble was correctly saved

    (loading is tested separately)
    """

def test_ensemble_save_load(pipeline, saved_ensemble):
    _, save_path, args, foundation_cache = saved_ensemble
    ensemble = EnsembleTrainer.load(save_path, args, foundation_cache=foundation_cache)
    sentences = [["This", "is", "a", "test"], ["This", "is", "another", "test"]]
    trees = parse_tokenized_sentences(args, ensemble.model, [pipeline], sentences)
    check_basic_predictions(trees)

def test_parse_text(tmp_path, pipeline, saved_ensemble):
    _, model_path, args, foundation_cache = saved_ensemble

    raw_file = str(tmp_path / "test_input.txt")
    with open(raw_file, "w") as fout:
        fout.write("This is a test\nThis is another test\n")
    output_file = str(tmp_path / "test_output.txt")

    args = dict(args)
    args['tokenized_file'] = raw_file
    args['predict_file'] = output_file

    text_processing.load_model_parse_text(args, model_path, [pipeline])
    trees = tree_reader.read_treebank(output_file)
    trees = ["{}".format(x) for x in trees]
    expected_trees = ["(ROOT (S (NP (DT This)) (VP (VBZ is) (NP (DT a) (NN test)))))",
                      "(ROOT (S (NP (DT This)) (VP (VBZ is) (NP (DT another) (NN test)))))"]
    assert trees == expected_trees

def test_pipeline(saved_ensemble):
    _, model_path, _, foundation_cache = saved_ensemble
    nlp = Pipeline("en", processors="tokenize,pos,constituency", constituency_model_path=str(model_path), foundation_cache=foundation_cache, download_method=None)
    doc = nlp("This is a test")
    tree = "{}".format(doc.sentences[0].constituency)
    assert tree == "(ROOT (S (NP (DT This)) (VP (VBZ is) (NP (DT a) (NN test)))))"
