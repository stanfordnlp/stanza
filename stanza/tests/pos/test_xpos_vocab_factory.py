"""
Test some pieces of the depparse dataloader
"""
import pytest

from stanza.models.pos.data import DataLoader
from stanza.models.pos.xpos_vocab_factory import xpos_vocab_factory
from stanza.models.pos.xpos_vocab_utils import XPOSDescription, XPOSType, build_xpos_vocab, choose_simplest_factory
from stanza.utils.conll import CoNLL
from stanza.models.pos.vocab import WordVocab, XPOSVocab

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

EN_EXAMPLE="""
1	Sh'reyan	Sh'reyan	PROPN	NNP%(tag)s	Number=Sing	3	nmod:poss	3:nmod:poss	_
2	's	's	PART	POS%(tag)s	_	1	case	1:case	_
3	antennae	antenna	NOUN%(tag)s	NNS	Number=Plur	6	nsubj	6:nsubj	_
4	are	be	VERB	VBP%(tag)s	Mood=Ind|Tense=Pres|VerbForm=Fin	6	cop	6:cop	_
5	hella	hella	ADV	RB%(tag)s	_	6	advmod	6:advmod	_
6	thicc	thicc	ADJ	JJ%(tag)s	Degree=Pos	0	root	0:root	_
"""

def build_data(iterations, tag):
    """
    build N copies of the english text above, with a lambda function applied for the tag suffices

    for example:
      lambda x: "" means the suffices are all blank (NNP, POS, NNS, etc) for each iteration
      lambda x: "-%d" % x means they go (NNP-0, NNP-1, NNP-2, etc) for the first word's tag
    """
    texts = [EN_EXAMPLE % {"tag": tag(i)} for i in range(iterations)]
    text = "\n\n".join(texts)
    doc = CoNLL.conll2doc(input_str=text)
    data = DataLoader.load_doc(doc)
    return data
    

def test_basic_en_ewt():
    """
    en_ewt is currently the basic vocab

    note that this may change if the dataset is drastically relabeled in the future
    """
    data = build_data(1, lambda x: "")
    vocab = xpos_vocab_factory(data, "en_ewt")
    assert isinstance(vocab, WordVocab)


def test_basic_en_unknown():
    """
    With only 6 tags, it should use a basic vocab for an unknown dataset
    """
    data = build_data(10, lambda x: "")
    vocab = xpos_vocab_factory(data, "en_unknown")
    assert isinstance(vocab, WordVocab)



def test_dash_en_unknown():
    """
    With this many different tags, it should choose to reduce it to the base xpos removing the -
    """
    data = build_data(10, lambda x: "-%d" % x)
    vocab = xpos_vocab_factory(data, "en_unknown")
    assert isinstance(vocab, XPOSVocab)
    assert vocab.sep == "-"


