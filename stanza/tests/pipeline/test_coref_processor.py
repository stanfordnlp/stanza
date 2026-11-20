import pytest
import torch

from stanza.models.common.doc import Document
from stanza.models.coref.const import CorefResult
from stanza.pipeline.coref_processor import CorefProcessor

pytestmark = pytest.mark.pipeline

def test_zero_anaphora_uses_original_span_head():
    class Model:
        def build_doc(self, doc):
            return doc

        def run(self, doc):
            # The zero-scored head's span sorts before the overt mention.
            return CorefResult(
                word_clusters=[[1, 4]],
                span_clusters=[[(1, 2), (0, 5)]],
                zero_scores=torch.tensor([-1.0, 1.0]),
            )

    words = "Yesterday Alice said she left".split()
    document = Document([[{"id": i, "text": word} for i, word in enumerate(words, 1)]])
    processor = CorefProcessor.__new__(CorefProcessor)
    processor._model = Model()
    processor._use_zeros = True

    chain, = processor.process(document).coref

    assert chain.representative_text == "Alice"
    assert {(mention.start_word, mention.end_word) for mention in chain.mentions} == {
        (1, 2), ((4, 1), (4, 1)),
    }
