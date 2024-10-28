import pytest

from stanza import Pipeline
from stanza.utils.conll import CoNLL
from stanza.tests import TEST_MODELS_DIR

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def test_from_conllu():
    """
    If the doc does not have the entire text available, make sure it still safely processes the text

    Test case supplied from user - see issue #1428
    """
    pipe = Pipeline("en", dir=TEST_MODELS_DIR, processors="tokenize,ner", download_method=None)
    doc = pipe("In February, I traveled to Seattle.  Dr. Pritchett gave me a new hip")
    ents = [x.text for x in doc.ents]
    # the default NER model ought to find these three
    assert ents == ['February', 'Seattle', 'Pritchett']

    doc_conllu = "{:C}\n\n".format(doc)
    doc = CoNLL.conll2doc(input_str=doc_conllu)
    pipe = Pipeline("en", dir=TEST_MODELS_DIR, processors="tokenize,ner", tokenize_pretokenized=True, download_method=None)
    pipe(doc)
    ents = [x.text for x in doc.ents]
    # this should still work when processed from a CoNLLu document
    # the bug previously caused a crash because the text to construct
    # the entities was not available, since the Document wouldn't have
    # the entire document text available
    assert ents == ['February', 'Seattle', 'Pritchett']
