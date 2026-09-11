import json
from types import SimpleNamespace

import torch

from stanza.models.coref.dataset import CorefDataset
from stanza.models.coref.pairwise_encoder import PairwiseEncoder
from stanza.utils.conll import CoNLL
from stanza.utils.datasets.coref.convert_udcoref import process_documents


SPEAKER_CONLLU = """\
# newdoc id = speaker_test
# sent_id = speaker_test-1
# speaker = Alice
# text = Hello
1\tHello\thello\tINTJ\tUH\t_\t0\troot\t0:root\t_
1.1\t_\t_\tPRON\tPRP\t_\t1\tdep\t1:dep\t_

# sent_id = speaker_test-2
# speaker = Bob
# text = Hi
1\tHi\thi\tINTJ\tUH\t_\t0\troot\t0:root\t_
"""


class IdentityTokenizer:
    def tokenize(self, word):
        return [word]


def test_prepare_udcoref_speakers():
    doc = CoNLL.conll2multi_docs(input_str=SPEAKER_CONLLU, ignore_gapping=False)[0]

    prepared = process_documents([(doc, "speaker_test", "en")])[0]
    expected = [sentence.speaker for sentence in doc.sentences
                for word in sentence.all_words if word.text != "_"]

    assert prepared["speaker"] == expected
    assert prepared["speaker"] == ["Alice", "Bob"]


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
