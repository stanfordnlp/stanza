import json
from pathlib import Path
from types import SimpleNamespace

import torch

from stanza.models.coref.dataset import CorefDataset
from stanza.models.coref.pairwise_encoder import PairwiseEncoder
from stanza.utils.conll import CoNLL
from stanza.utils.datasets.coref.convert_udcoref import process_documents


GUM_DEV = Path("extern_data/coref/en_gum/en_gum-corefud-dev.conllu")


class IdentityTokenizer:
    def tokenize(self, word):
        return [word]


def test_prepare_udcoref_speakers():
    docs = CoNLL.conll2multi_docs(GUM_DEV, ignore_gapping=False)
    doc = next(doc for doc in docs
               if doc.sentences[0].doc_id == "GUM_conversation_grounded")

    prepared = process_documents([(doc, doc.sentences[0].doc_id, "en")])[0]
    expected = [sentence.speaker for sentence in doc.sentences
                for word in sentence.all_words if word.text != "_"]

    assert prepared["speaker"] == expected
    assert prepared["speaker"][:8] == ["Kendra"] * 6 + ["Sabrina"] * 2


def test_training_data_uses_speaker_features(tmp_path):
    prepared = [{
        "document_id": "speakers",
        "cased_words": ["I", "agree", "I"],
        "speaker": ["Alice", "Alice", "Bob"],
        "span_clusters": [],
        "word_clusters": [],
        "head2span": [],
    }]
    data_path = tmp_path / "coref.json"
    data_path.write_text(json.dumps(prepared), encoding="utf-8")
    dataset = CorefDataset(data_path, SimpleNamespace(bert_model="test"), IdentityTokenizer())

    encoder = PairwiseEncoder(SimpleNamespace(
        embedding_size=1, dropout_rate=0.0, full_pairwise=True,
    ))
    encoder.distance_emb.weight.data.zero_()
    encoder.speaker_emb.weight.data.copy_(torch.tensor([[10.0], [20.0]]))
    pairwise = encoder(torch.tensor([[0], [0], [1]]), dataset[0])

    # The first two pairs have the same speaker; the last pair does not.
    assert pairwise[:, 0, 0].tolist() == [20.0, 20.0, 10.0]
