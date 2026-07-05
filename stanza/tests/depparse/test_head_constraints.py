"""
Test the head-constraint violation detection and repair logic:
heads that end up with more than one nsubj/csubj, or more than one obj,
which the arc-factored graph parser and Chu-Liu-Edmonds do not enforce
on their own.
"""

import numpy as np
import pytest

from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root
from stanza.models.common.vocab import VOCAB_PREFIX_SIZE
from stanza.models.pos.vocab import WordVocab
from stanza.models.depparse.head_constraints import (
    find_head_constraint_violations,
    resolve_head_constraint_violations,
)

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def make_deprel_vocab(labels):
    """
    Builds a real deprel WordVocab (the same class stanza.models.depparse.data
    uses for the deprel vocab) from a plain list of label strings, so these
    tests exercise the actual Vocab interface -- unit2id/unmap, and the real
    VOCAB_PREFIX_SIZE offset -- rather than an approximation of it.
    """
    data = [[(label,) for label in labels]]
    return WordVocab(data, idx=0, cutoff=0)


def test_find_head_constraint_violations_basic():
    """ A clean tree (one nsubj, one obj) has no violations """
    tree = [0, 3, 3, 0]
    deprels = ["nsubj", "obj", "root"]
    assert find_head_constraint_violations(tree, deprels) == []

def test_find_head_constraint_violations_double_obj():
    """ Two objs on the same head is a violation """
    tree = [0, 1, 1]
    deprels = ["obj", "obj"]
    violations = find_head_constraint_violations(tree, deprels)
    assert len(violations) == 1
    head_idx, group_name, deps = violations[0]
    assert head_idx == 1
    assert group_name == "obj"
    assert deps == [1, 2]

def test_find_head_constraint_violations_nsubj_csubj_mixed():
    """ An nsubj and a csubj on the same head are still a single 'subj' violation """
    tree = [0, 3, 3, 0]
    deprels = ["nsubj", "csubj", "root"]
    violations = find_head_constraint_violations(tree, deprels)
    assert len(violations) == 1
    assert violations[0][1] == "subj"

def test_find_head_constraint_violations_outer_exception():
    """
    nsubj:outer/csubj:outer is a SEPARATE group from plain nsubj/csubj, so a
    head with one of each is legal -- the "outer" clausal subject of a
    copula construction, eg:
      "The thing to keep in mind is that X, Y, and Z are more dangerous..."
    where "thing" is nsubj:outer of "likely" and "nationalists" is a
    separate, ordinary nsubj of that same head.
    """
    tree = [0, 19, 19, 0]
    deprels = ["nsubj:outer", "nsubj", "root"]
    assert find_head_constraint_violations(tree, deprels) == []

    # but two nsubj:outer on the same head IS a violation
    deprels_violating = ["nsubj:outer", "nsubj:outer", "root"]
    violations = find_head_constraint_violations(tree, deprels_violating)
    assert len(violations) == 1
    assert violations[0][1] == "subj_outer"


def test_resolve_label_swap():
    """
    "I gave the cat a fish": the model slightly (and wrongly) prefers
    obj(gave, cat) over iobj(gave, cat), while obj(gave, fish) is genuinely
    correct. Since the arc "cat -> gave" is highly confident on its own
    (cat has no plausible alternative head), the correct repair is to
    relabel cat to iobj IN PLACE, not to reroute it to a different head.
    """
    vocab = make_deprel_vocab(["root", "nsubj", "obj", "iobj", "det", "punct"])
    n = 7  # 0=root, 1=I, 2=gave, 3=the, 4=cat, 5=a, 6=fish
    scores = np.full((n, n), -30.0)
    scores[0, 0] = 0
    scores[2, 0] = -1.0
    scores[1, 2] = -1.0
    scores[3, 4] = -1.0
    scores[5, 6] = -1.0
    # cat's only plausible head is "gave"; everywhere else is a bad arc
    scores[4, 2] = -1.0
    for h in (0, 1, 3, 5, 6):
        scores[4, h] = -20.0
    # fish's only plausible head is also "gave"
    scores[6, 2] = -1.2
    for h in (0, 1, 3, 4, 5):
        scores[6, h] = -20.0

    label_log_probs = np.full((n, n, len(vocab) - VOCAB_PREFIX_SIZE), -10.0)
    def set_label(dep, head, label, score):
        label_log_probs[dep, head, vocab.unit2id(label) - VOCAB_PREFIX_SIZE] = score
    set_label(2, 0, "root", -0.1)
    set_label(1, 2, "nsubj", -0.1)
    set_label(3, 4, "det", -0.1)
    set_label(5, 6, "det", -0.1)
    set_label(4, 2, "obj", -0.5)   # cat's best (wrong) label
    set_label(4, 2, "iobj", -0.6)  # cat's correct label, a close second
    set_label(6, 2, "obj", -0.2)   # fish's best (correct) label
    set_label(6, 2, "iobj", -5.0)  # fish's alternative is clearly worse

    tree = chuliu_edmonds_one_root(scores.copy())
    fixed_tree, raw_ids = resolve_head_constraint_violations(scores, label_log_probs, tree, vocab)
    labels = vocab.unmap([r + VOCAB_PREFIX_SIZE for r in raw_ids])

    np.testing.assert_array_equal(fixed_tree, tree)  # structure is unchanged
    assert labels[3] == "iobj"  # cat: relabeled
    assert labels[5] == "obj"   # fish: unchanged
    assert find_head_constraint_violations(fixed_tree, labels) == []

def test_resolve_structural_reroute():
    """
    When relabeling in place would be a bad option (no decent non-group
    label exists for one of the offending dependents on its current arc),
    but a genuinely good alternative HEAD exists, the repair should reroute
    that dependent structurally instead of forcing a poor label choice.
    """
    vocab = make_deprel_vocab(["root", "nsubj", "csubj", "obj", "advmod"])
    n = 5  # 0=root, 1=verb, 2=subjA, 3=subjB, 4=otherhead
    scores = np.full((n, n), -30.0)
    scores[0, 0] = 0
    scores[1, 0] = -1.0
    scores[2, 1] = -1.0
    scores[3, 1] = -1.05   # subjB's arc to verb is decent
    scores[3, 4] = -1.1    # subjB's arc to otherhead is nearly as good
    scores[4, 1] = -1.0

    label_log_probs = np.full((n, n, len(vocab) - VOCAB_PREFIX_SIZE), -10.0)
    def set_label(dep, head, label, score):
        label_log_probs[dep, head, vocab.unit2id(label) - VOCAB_PREFIX_SIZE] = score
    set_label(1, 0, "root", -0.1)
    set_label(2, 1, "nsubj", -0.1)
    set_label(3, 1, "nsubj", -0.15)     # subjB's current (conflicting) label
    set_label(3, 1, "advmod", -8.0)     # subjB's only non-group option here is terrible
    set_label(3, 4, "obj", -0.2)        # subjB's label if rerouted: great
    set_label(4, 1, "obj", -0.1)

    tree = chuliu_edmonds_one_root(scores.copy())
    fixed_tree, raw_ids = resolve_head_constraint_violations(scores, label_log_probs, tree, vocab)
    labels = vocab.unmap([r + VOCAB_PREFIX_SIZE for r in raw_ids])

    assert fixed_tree[3] == 4  # subjB rerouted to otherhead
    assert find_head_constraint_violations(fixed_tree, labels) == []

def test_resolve_valency_three():
    """ A head with THREE competing objs, not just two, should still resolve cleanly """
    vocab = make_deprel_vocab(["root", "nsubj", "csubj", "obj", "advmod", "obl"])
    n = 6  # 0=root, 1/2/3=competing objs, 4=verb, 5=harmless obl dependent
    scores = np.full((n, n), -30.0)
    scores[0, 0] = 0
    scores[1, 4] = -0.5; scores[1, 5] = -3.0; scores[1, 0] = -10.0
    scores[2, 4] = -1.0; scores[2, 5] = -1.2; scores[2, 0] = -10.0
    scores[3, 4] = -1.5; scores[3, 5] = -1.6; scores[3, 0] = -10.0
    scores[4, 0] = -1.0
    for h in (1, 2, 3, 5):
        scores[4, h] = -10.0
    scores[5, 4] = -1.0
    for h in (0, 1, 2, 3):
        scores[5, h] = -10.0

    label_log_probs = np.full((n, n, len(vocab) - VOCAB_PREFIX_SIZE), -10.0)
    def set_label(dep, head, label, score):
        label_log_probs[dep, head, vocab.unit2id(label) - VOCAB_PREFIX_SIZE] = score
    set_label(1, 4, "obj", -0.2); set_label(1, 5, "advmod", -1.0); set_label(1, 0, "root", -0.1)
    set_label(2, 4, "obj", -0.3); set_label(2, 5, "advmod", -1.5); set_label(2, 0, "root", -0.1)
    set_label(3, 4, "obj", -0.4); set_label(3, 5, "advmod", -1.8); set_label(3, 0, "root", -0.1)
    set_label(4, 0, "root", -0.1)
    set_label(5, 4, "obl", -0.1)

    tree = chuliu_edmonds_one_root(scores.copy())
    initial_labels = vocab.unmap([r + VOCAB_PREFIX_SIZE for r in
                                 [int(np.argmax(label_log_probs[j, tree[j]])) for j in range(1, len(tree))]])
    assert len(find_head_constraint_violations(tree, initial_labels)) == 1  # sanity check on the fixture

    fixed_tree, raw_ids = resolve_head_constraint_violations(scores, label_log_probs, tree, vocab)
    labels = vocab.unmap([r + VOCAB_PREFIX_SIZE for r in raw_ids])
    assert find_head_constraint_violations(fixed_tree, labels) == []

def test_resolve_clean_sentence_is_a_noop():
    """ A sentence with no violations should come back completely unchanged """
    vocab = make_deprel_vocab(["root", "nsubj", "obj"])
    n = 4
    scores = np.full((n, n), -30.0)
    scores[0, 0] = 0
    scores[1, 3] = -1.0
    scores[2, 3] = -1.1
    scores[3, 0] = -1.0

    label_log_probs = np.full((n, n, len(vocab) - VOCAB_PREFIX_SIZE), -10.0)
    def set_label(dep, head, label, score):
        label_log_probs[dep, head, vocab.unit2id(label) - VOCAB_PREFIX_SIZE] = score
    set_label(1, 3, "nsubj", -0.1)
    set_label(2, 3, "obj", -0.1)
    set_label(3, 0, "root", -0.1)

    tree = chuliu_edmonds_one_root(scores.copy())
    fixed_tree, raw_ids = resolve_head_constraint_violations(scores, label_log_probs, tree, vocab)
    labels = vocab.unmap([r + VOCAB_PREFIX_SIZE for r in raw_ids])

    np.testing.assert_array_equal(fixed_tree, tree)
    assert labels == ["nsubj", "obj", "root"]

def test_resolve_two_independent_violations_dont_oscillate():
    """
    A sentence with two SEPARATE violations in different clauses: one that
    needs a label swap ("I gave the cat a fish") and one, in a different
    clause, that needs a structural reroute. Earlier versions of this
    repair recomputed every word's label from scratch each round with no
    memory of prior decisions, which caused the two fixes to repeatedly
    undo each other (oscillating between states) instead of both holding.
    This confirms both fixes survive together, in the same final tree.
    """
    vocab = make_deprel_vocab(["root", "nsubj", "csubj", "obj", "iobj", "det", "advmod"])
    # 1=I 2=gave 3=the 4=cat 5=a 6=fish | 7=verb2 8=subjA 9=subjB 10=otherhead
    n = 11
    scores = np.full((n, n), -30.0)
    scores[0, 0] = 0
    scores[2, 0] = -1.0; scores[1, 2] = -1.0; scores[3, 4] = -1.0; scores[5, 6] = -1.0
    scores[4, 2] = -1.0
    for h in (0, 1, 3, 5, 6):
        scores[4, h] = -20.0
    scores[6, 2] = -1.2
    for h in (0, 1, 3, 4, 5):
        scores[6, h] = -20.0
    scores[7, 0] = -1.0
    scores[8, 7] = -1.0
    scores[9, 7] = -1.05; scores[9, 10] = -1.1
    scores[10, 7] = -1.0

    label_log_probs = np.full((n, n, len(vocab) - VOCAB_PREFIX_SIZE), -10.0)
    def set_label(dep, head, label, score):
        label_log_probs[dep, head, vocab.unit2id(label) - VOCAB_PREFIX_SIZE] = score
    set_label(2, 0, "root", -0.1)
    set_label(1, 2, "nsubj", -0.1)
    set_label(3, 4, "det", -0.1)
    set_label(5, 6, "det", -0.1)
    set_label(4, 2, "obj", -0.5)
    set_label(4, 2, "iobj", -0.6)
    set_label(6, 2, "obj", -0.2)
    set_label(6, 2, "iobj", -5.0)
    set_label(7, 0, "root", -0.1)
    set_label(8, 7, "nsubj", -0.1)
    set_label(9, 7, "nsubj", -0.15)
    set_label(9, 7, "advmod", -8.0)
    set_label(9, 10, "obj", -0.2)
    set_label(10, 7, "obj", -0.1)

    tree = chuliu_edmonds_one_root(scores.copy())
    fixed_tree, raw_ids = resolve_head_constraint_violations(scores, label_log_probs, tree, vocab)
    labels = vocab.unmap([r + VOCAB_PREFIX_SIZE for r in raw_ids])

    assert labels[3] == "iobj"     # cat: label-swap fix survives
    assert labels[5] == "obj"      # fish: unaffected, unchanged
    assert fixed_tree[9] == 10     # subjB: reroute fix survives
    assert find_head_constraint_violations(fixed_tree, labels) == []
