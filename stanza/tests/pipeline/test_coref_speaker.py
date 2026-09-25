from stanza.models.common.doc import Document
from stanza.models.coref.const import CorefResult
from stanza.pipeline.coref_processor import CorefProcessor


class RecordingModel:
    def build_doc(self, doc):
        self.doc = doc
        return doc

    def run(self, doc):
        assert doc is self.doc
        return CorefResult(word_clusters=[], span_clusters=[], zero_scores=None)


def test_pipeline_passes_speakers_to_coref_model():
    document = Document([
        [{"id": 1, "text": "Hello"}, {"id": 2, "text": "Alice"}],
        [{"id": 1, "text": "Hi"}],
        [{"id": 1, "text": "Narration"}],
    ])
    document.sentences[0].speaker = "Bob"
    document.sentences[1].speaker = "Alice"

    processor = CorefProcessor.__new__(CorefProcessor)
    processor._model = RecordingModel()
    processor._use_zeros = False
    processor.process(document)

    assert processor._model.doc["speaker"] == ["Bob", "Bob", "Alice", "_"]
