import json
import pytest

from stanza.models import ner_tagger
from stanza.models.common.doc import Document
from stanza.models.ner.data import DataLoader
from stanza.tests import TEST_WORKING_DIR

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]


ONE_SENTENCE = """
[
 [
  {
   "text": "EU",
   "ner": "B-ORG"
  },
  {
   "text": "rejects",
   "ner": "O"
  },
  {
   "text": "German",
   "ner": "B-MISC"
  },
  {
   "text": "call",
   "ner": "O"
  },
  {
   "text": "to",
   "ner": "O"
  },
  {
   "text": "boycott",
   "ner": "O"
  },
  {
   "text": "Mox",
   "ner": "B-MISC"
  },
  {
   "text": "Opal",
   "ner": "I-MISC"
  },
  {
   "text": ".",
   "ner": "O"
  }
 ]
]
"""

@pytest.fixture(scope="module")
def pretrain_file():
    return f'{TEST_WORKING_DIR}/in/tiny_emb.pt'


@pytest.fixture(scope="module")
def one_sentence_json_path(tmpdir_factory):
    filename = tmpdir_factory.mktemp('data').join("sentence.json")
    with open(filename, 'w') as fout:
        fout.write(ONE_SENTENCE)
    return filename


def test_build_vocab(pretrain_file, one_sentence_json_path, tmp_path):
    """
    Test that when loading a data file, we get back 
    """
    args = ner_tagger.parse_args(["--wordvec_pretrain_file", pretrain_file])
    pt = ner_tagger.load_pretrain(args)

    with open(one_sentence_json_path) as fin:
        train_doc = Document(json.load(fin))

    train_batch = DataLoader(train_doc, args['batch_size'], args, pt, vocab=None, evaluation=False, scheme=args.get('train_scheme'), max_batch_words=args['max_batch_words'])

    vocab = train_batch.vocab
    pt_words = list(vocab['word'])
    assert pt_words == ['<PAD>', '<UNK>', '<EMPTY>', '<ROOT>', 'unban', 'mox', 'opal']
    delta_words = list(vocab['delta'])
    assert delta_words == ['<PAD>', '<UNK>', '<EMPTY>', '<ROOT>', 'eu', 'rejects', 'german', 'call', 'to', 'boycott', 'mox', 'opal', '.']
    tags = list(vocab['tag'])
    assert tags == [['<PAD>'], ['<UNK>'], [], ['<ROOT>'], ['S-ORG'], ['O'], ['S-MISC'], ['B-MISC'], ['E-MISC']]


def test_build_vocab_ignore_repeats(pretrain_file, one_sentence_json_path, tmp_path):
    """
    Test that when loading a data file, we get back 
    """
    args = ner_tagger.parse_args(["--wordvec_pretrain_file", pretrain_file, "--emb_finetune_known_only"])
    pt = ner_tagger.load_pretrain(args)

    with open(one_sentence_json_path) as fin:
        train_doc = Document(json.load(fin))

    train_batch = DataLoader(train_doc, args['batch_size'], args, pt, vocab=None, evaluation=False, scheme=args.get('train_scheme'), max_batch_words=args['max_batch_words'])

    vocab = train_batch.vocab
    pt_words = list(vocab['word'])
    assert pt_words == ['<PAD>', '<UNK>', '<EMPTY>', '<ROOT>', 'unban', 'mox', 'opal']
    delta_words = list(vocab['delta'])
    assert delta_words == ['<PAD>', '<UNK>', '<EMPTY>', '<ROOT>', 'mox', 'opal']
    tags = list(vocab['tag'])
    assert tags == [['<PAD>'], ['<UNK>'], [], ['<ROOT>'], ['S-ORG'], ['O'], ['S-MISC'], ['B-MISC'], ['E-MISC']]
