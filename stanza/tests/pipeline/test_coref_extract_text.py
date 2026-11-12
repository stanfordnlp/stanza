import pytest
from stanza.models.common.doc import Document
from stanza.models.coref.const import CorefResult
from stanza.pipeline.coref_processor import CorefProcessor, extract_text


class MockCorefModel:
    """Mock model that returns predefined clusters for testing CorefProcessor."""
    def __init__(self, span_clusters, word_clusters=None):
        self.span_clusters = span_clusters
        self.word_clusters = word_clusters if word_clusters is not None else [[s[0] for s in c] for c in span_clusters]

    def build_doc(self, doc):
        return doc

    def run(self, doc):
        return CorefResult(word_clusters=self.word_clusters, span_clusters=self.span_clusters, zero_scores=None)


def test_extract_text_preserves_spacing_with_punctuation():
    """Verify that extract_text respects token spaces_after for punctuation and dates."""
    doc = Document([
        [
            {'id': 1, 'text': 'July', 'start_char': 0, 'end_char': 4},
            {'id': 2, 'text': '4', 'start_char': 5, 'end_char': 6},
            {'id': 3, 'text': ',', 'start_char': 6, 'end_char': 7},
            {'id': 4, 'text': '1976', 'start_char': 8, 'end_char': 12},
        ]
    ], text='July 4, 1976')
    doc.mark_whitespace()

    # extract 'July 4, 1976'
    assert extract_text(doc, 0, 0, 4) == 'July 4, 1976'

    # extract '4, 1976'
    assert extract_text(doc, 0, 1, 4) == '4, 1976'


def test_extract_text_abbreviations_and_clitics():
    """Verify that extract_text handles abbreviations and possessive clitics properly."""
    doc = Document([
        [
            {'id': 1, 'text': 'Washington', 'start_char': 0, 'end_char': 10},
            {'id': 2, 'text': ',', 'start_char': 10, 'end_char': 11},
            {'id': 3, 'text': 'D.C.', 'start_char': 12, 'end_char': 16},
            {'id': 4, 'text': "'s", 'start_char': 16, 'end_char': 18},
        ]
    ], text="Washington, D.C.'s")
    doc.mark_whitespace()

    assert extract_text(doc, 0, 0, 3) == 'Washington, D.C.'
    assert extract_text(doc, 0, 0, 4) == "Washington, D.C.'s"


def test_extract_text_with_misc_space_after():
    """Verify that tokens constructed from CoNLL-U style SpaceAfter=No are handled."""
    doc = Document([
        [
            {'id': 1, 'text': 'Dr', 'misc': 'SpaceAfter=No'},
            {'id': 2, 'text': '.'},
            {'id': 3, 'text': 'Smith'},
        ]
    ])
    assert extract_text(doc, 0, 0, 2) == 'Dr.'
    assert extract_text(doc, 0, 0, 3) == 'Dr. Smith'


def test_extract_text_multi_word_tokens():
    """Verify extract_text when spans contain full or partial multi-word tokens."""
    # French 'des pommes' -> Token 1-2 ('des'), Word 1 ('de'), Word 2 ('les'), Token 3 ('pommes')
    doc = Document([
        [
            {'id': (1, 2), 'text': 'des', 'start_char': 0, 'end_char': 3, 'spaces_after': ' '},
            {'id': 1, 'text': 'de'},
            {'id': 2, 'text': 'les'},
            {'id': 3, 'text': 'pommes', 'start_char': 4, 'end_char': 10, 'spaces_after': ''},
        ]
    ], text='des pommes')
    doc.mark_whitespace()

    # Entire MWT + next token
    assert extract_text(doc, 0, 0, 3) == 'des pommes'

    # Partial MWT: only Word 1 ('de')
    assert extract_text(doc, 0, 0, 1) == 'de'

    # Partial MWT: only Word 2 ('les') + next token ('pommes')
    assert extract_text(doc, 0, 1, 3) == 'les pommes'


def test_coref_processor_representative_text_spacing():
    """Verify that CorefProcessor sets the representative_text with accurate spacing."""
    # "He was born on July 4, 1976. That date was historic."
    doc = Document([
        [
            {'id': 1, 'text': 'He', 'start_char': 0, 'end_char': 2},
            {'id': 2, 'text': 'was', 'start_char': 3, 'end_char': 6},
            {'id': 3, 'text': 'born', 'start_char': 7, 'end_char': 11},
            {'id': 4, 'text': 'on', 'start_char': 12, 'end_char': 14},
            {'id': 5, 'text': 'July', 'start_char': 15, 'end_char': 19},
            {'id': 6, 'text': '4', 'start_char': 20, 'end_char': 21},
            {'id': 7, 'text': ',', 'start_char': 21, 'end_char': 22},
            {'id': 8, 'text': '1976', 'start_char': 23, 'end_char': 27},
            {'id': 9, 'text': '.', 'start_char': 27, 'end_char': 28},
        ],
        [
            {'id': 1, 'text': 'That', 'start_char': 29, 'end_char': 33},
            {'id': 2, 'text': 'date', 'start_char': 34, 'end_char': 38},
            {'id': 3, 'text': 'was', 'start_char': 39, 'end_char': 42},
            {'id': 4, 'text': 'historic', 'start_char': 43, 'end_char': 51},
            {'id': 5, 'text': '.', 'start_char': 51, 'end_char': 52},
        ]
    ], text='He was born on July 4, 1976. That date was historic.')
    doc.mark_whitespace()

    # Cluster connecting "July 4, 1976" (sentence 0, words 4 to 8) and "That date" (sentence 1, words 0 to 2)
    # The first sentence has 9 words (0..8).
    # Span indices across flattened document words:
    # Sent 0: word 4 to 8 is indices [4, 8]
    # Sent 1: word 0 to 2 is indices [9, 11]
    clusters = [
        [[4, 8], [9, 11]]
    ]

    processor = CorefProcessor.__new__(CorefProcessor)
    processor._model = MockCorefModel(span_clusters=clusters)
    processor._use_zeros = False
    processor.process(doc)

    assert len(doc.coref) == 1
    chain = doc.coref[0]
    # The representative text must be "July 4, 1976" without an extra space before the comma
    assert chain.representative_text == 'July 4, 1976'
