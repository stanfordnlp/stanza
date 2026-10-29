"""
Tests for comparing the predictions of two lemmatizers
"""

import pytest

from stanza.utils.lemma import compare_lemma_predictions

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def conllu(entries):
    lines = []
    for form, upos, lemma in entries:
        lines.append(f"1\t{form}\t{lemma}\t{upos}\t_\t_\t0\troot\t_\t_")
        lines.append("")
    return "\n".join(lines) + "\n"

TRAIN = [("dogs", "NOUN", "dog"), ("ran", "VERB", "run")]
LEXICON = [("housewives", "NOUN", "housewife")]
GOLD = [("dogs", "NOUN", "dog"), ("housewives", "NOUN", "housewife"), ("dives", "NOUN", "dive"), ("knives", "NOUN", "knife"), ("ran", "VERB", "run")]
PRED_A = [("dogs", "NOUN", "dog"), ("housewives", "NOUN", "housewife"), ("dives", "NOUN", "dife"), ("knives", "NOUN", "knife"), ("ran", "VERB", "run")]
PRED_B = [("dogs", "NOUN", "dog"), ("housewives", "NOUN", "housewife"), ("dives", "NOUN", "dive"), ("knives", "NOUN", "knive"), ("ran", "VERB", "run")]

def test_compare(tmp_path):
    files = {}
    for name, entries in [("train", TRAIN), ("lexicon", LEXICON), ("gold", GOLD), ("a", PRED_A), ("b", PRED_B)]:
        files[name] = tmp_path / ("%s.conllu" % name)
        files[name].write_text(conllu(entries), encoding="utf-8")

    gold_words = compare_lemma_predictions.read_words(str(files["gold"]))
    categories = compare_lemma_predictions.categorize(gold_words, TRAIN, LEXICON)
    assert categories == ["train", "lexicon", "seq2seq", "seq2seq", "train"]

    totals, differences = compare_lemma_predictions.compare(gold_words, PRED_A, PRED_B, categories)
    assert totals["seq2seq"] == {"words": 2, "a": 1, "b": 1}
    assert totals["all"] == {"words": 5, "a": 4, "b": 4}
    assert differences == {("seq2seq", "dives", "NOUN", "dive", "dife", "dive"): 1,
                           ("seq2seq", "knives", "NOUN", "knife", "knife", "knive"): 1}

    output = tmp_path / "compare.md"
    compare_lemma_predictions.main(["--gold", str(files["gold"]), "--pred_a", str(files["a"]), "--pred_b", str(files["b"]),
                                    "--train", str(files["train"]), "--lexicon", str(files["lexicon"]), "--output", str(output)])
    report = output.read_text(encoding="utf-8")
    assert "| seq2seq | 2 | 50.00 | 50.00 | +0 words |" in report
    assert "## seq2seq: B better on 1, worse on 1" in report

def test_misaligned():
    with pytest.raises(ValueError):
        compare_lemma_predictions.compare(GOLD, PRED_A[:4], PRED_B, ["train"] * 5)
    with pytest.raises(ValueError):
        compare_lemma_predictions.compare(GOLD, [("cats", "NOUN", "cat")] + PRED_A[1:], PRED_B, ["train"] * 5)
