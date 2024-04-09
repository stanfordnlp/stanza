"""
Add a simple test of the Ensemble's inference path

This just reuses one model several times - that should still check the main loop, at least
"""

import pytest

from stanza import Pipeline
from stanza.models.constituency.ensemble import Ensemble
from stanza.models.constituency.text_processing import parse_tokenized_sentences

from stanza.tests import TEST_MODELS_DIR

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]


@pytest.fixture(scope="module")
def pipeline():
    return Pipeline(dir=TEST_MODELS_DIR, lang="en", processors="tokenize, pos, constituency", tokenize_pretokenized=True)

def test_ensemble_inference(pipeline):
    # test the ensemble by reusing the same parser multiple times
    con_processor = pipeline.processors["constituency"]
    model = con_processor._model
    args = dict(model.args)
    foundation_cache = pipeline.foundation_cache

    model_path = con_processor._config['model_path']
    # reuse the same model 3 times just to make sure the code paths are working
    filenames = [model_path, model_path, model_path]

    ensemble = Ensemble(filenames, args, foundation_cache)
    sentences = [["This", "is", "a", "test"], ["This", "is", "another", "test"]]
    trees = parse_tokenized_sentences(args, ensemble, [pipeline], sentences)

    predictions = [x.predictions for x in trees]
    assert len(predictions) == 2
    assert all(len(x) == 1 for x in predictions)
    trees = [x[0].tree for x in predictions]
    result = ["{}".format(tree) for tree in trees]
    expected = ["(ROOT (S (NP (DT This)) (VP (VBZ is) (NP (DT a) (NN test)))))",
                "(ROOT (S (NP (DT This)) (VP (VBZ is) (NP (DT another) (NN test)))))"]
    assert result == expected
