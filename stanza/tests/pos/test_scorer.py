"""
Test the per-tag and per-feature scores from the POS scorer
"""

import logging

import pytest

from stanza.models.pos import scorer
from stanza.models import tagger
from stanza.utils.conll import CoNLL

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

GOLD = """
# text = Kom binnen!
1	Kom	komen	VERB	WW|pv|tgw|ev	Mood=Imp|VerbForm=Fin	0	root	0:root	_
2	binnen	binnen	ADV	BW	_	1	compound:prt	1:compound:prt	SpaceAfter=No
3	!	!	PUNCT	LET	_	1	punct	1:punct	_

# text = Hij komt binnen.
1	Hij	hij	PRON	VNW|pers|pron|nomin|vol|3|ev|masc	Case=Nom|Person=3|PronType=Prs	2	nsubj	2:nsubj	_
2	komt	komen	VERB	WW|pv|tgw|met-t	Mood=Ind|Number=Sing|Tense=Pres|VerbForm=Fin	0	root	0:root	_
3	binnen	binnen	ADV	BW	_	2	compound:prt	2:compound:prt	SpaceAfter=No
4	.	.	PUNCT	LET	_	2	punct	2:punct	_
""".lstrip()

def build_docs():
    gold = CoNLL.conll2doc(input_str=GOLD)
    pred = CoNLL.conll2doc(input_str=GOLD)
    # the classic mistake: the imperative tagged as indicative
    pred.sentences[0].words[0].feats = "Mood=Ind|Number=Sing|Tense=Pres|VerbForm=Fin"
    pred.sentences[0].words[0].xpos = "WW|pv|tgw|ev|x"
    pred.sentences[1].words[2].upos = "ADP"
    return pred, gold

def test_parse_feats():
    assert scorer.parse_feats(None) == {}
    assert scorer.parse_feats("_") == {}
    assert scorer.parse_feats("Mood=Imp|VerbForm=Fin") == {"Mood": "Imp", "VerbForm": "Fin"}
    assert scorer.parse_feats("PronType=Int,Rel") == {"PronType": "Int,Rel"}
    assert scorer.parse_feats("Foo") == {"Foo": "_"}

def test_count_upos():
    pred, gold = build_docs()
    tp, fp, fn = scorer.count_tag_f1(pred, gold, "upos")
    assert tp["ADV"] == 1
    assert fn["ADV"] == 1
    assert fp["ADP"] == 1
    assert tp["VERB"] == 2
    assert fp["VERB"] == 0

def test_count_features():
    pred, gold = build_docs()
    tp, fp, fn, confusion = scorer.count_feature_scores(pred, gold)
    assert tp["Mood=Imp"] == 0
    assert fn["Mood=Imp"] == 1
    assert fp["Mood=Ind"] == 1
    assert tp["Mood=Ind"] == 1
    # VerbForm=Fin is right on both verbs even though Mood is wrong on one
    assert tp["VerbForm=Fin"] == 2
    assert fn["VerbForm=Fin"] == 0
    # Number and Tense were added to the imperative
    assert fp["Number=Sing"] == 1
    assert fp["Tense=Pres"] == 1

    # every word is in each feature's confusion matrix
    assert sum(sum(row.values()) for row in confusion["Mood"].values()) == 7
    assert confusion["Mood"]["Imp"]["Ind"] == 1
    assert confusion["Mood"]["Ind"]["Ind"] == 1
    assert confusion["Mood"]["_"]["_"] == 5
    assert confusion["Number"]["_"]["Sing"] == 1

def test_feature_confusion_blank_first():
    pred, gold = build_docs()
    _, _, _, confusion = scorer.count_feature_scores(pred, gold)
    text = scorer.format_feature_confusion(confusion["Mood"])
    header = text.split("\n")[0].split()
    assert header == ["t\\p", "_", "Imp", "Ind"]

def test_score_named_tags_log(caplog):
    pred, gold = build_docs()
    with caplog.at_level(logging.INFO, logger="stanza"):
        scorer.score_named_tags(pred, gold, ["Mood"])
    text = caplog.text
    assert "F1 scores for each UPOS" in text
    assert "F1 scores for each XPOS" in text
    assert "Mood=Imp: p 0.0000 r 0.0000 f1 0.0000 (1 actual)" in text
    assert "Confusion matrix for feature Mood" in text
    assert "Confusion matrix for feature Number" not in text

def test_score_named_tags_no_confusion(caplog):
    pred, gold = build_docs()
    with caplog.at_level(logging.INFO, logger="stanza"):
        scorer.score_named_tags(pred, gold, [])
    assert "F1 scores for each feature value" in caplog.text
    assert "Confusion matrix" not in caplog.text

def test_score_named_tags_no_xpos(caplog):
    pred, gold = build_docs()
    for doc in (pred, gold):
        for sentence in doc.sentences:
            for word in sentence.words:
                word.xpos = None
    with caplog.at_level(logging.INFO, logger="stanza"):
        scorer.score_named_tags(pred, gold)
    assert "F1 scores for each UPOS" in caplog.text
    assert "F1 scores for each XPOS" not in caplog.text

def test_score_named_tags_mismatch(caplog):
    pred, gold = build_docs()
    gold.sentences[1].words.pop()
    with caplog.at_level(logging.INFO, logger="stanza"):
        scorer.score_named_tags(pred, gold)
    assert "sentence length mismatch" in caplog.text
    assert "F1 scores for each UPOS" not in caplog.text

def test_feature_confusion_args():
    args = tagger.parse_args([])
    assert args['feature_confusion'] is None
    args = tagger.parse_args(['--feature_confusion', 'Mood,VerbForm'])
    assert args['feature_confusion'] == ['Mood', 'VerbForm']
    args = tagger.parse_args(['--no_feature_confusion'])
    assert args['feature_confusion'] == []
