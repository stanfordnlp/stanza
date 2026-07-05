"""
Detects and repairs head-constraint violations in a decoded dependency tree:
heads that end up with more than one nsubj/csubj (or nsubj:pass/csubj:pass),
or more than one obj. These aren't enforced by the arc-factored graph parser
or by Chu-Liu-Edmonds itself (unlike the single-root constraint, which is
enforced by reweighting root arc costs -- see chuliu_edmonds_one_root), since
they depend on which LABEL ends up chosen for each arc, not just the tree
structure.

SINGLETON_DEPREL_GROUPS/DEPREL_TO_GROUP below are the canonical definitions
of which deprels are mutually exclusive on a single head; stanza.models.depparse.scorer
imports them from here to count how often violations occur across a corpus.
"""

from collections import defaultdict

import numpy as np

from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root
from stanza.models.common.vocab import VOCAB_PREFIX_SIZE

# groups of deprels which are mutually exclusive per head -- a single head
# should never have more than one edge belonging to the same group.
#
# nsubj:pass / csubj:pass collapse into the ordinary subj group (a head
# still shouldn't have two of nsubj/csubj/nsubj:pass/csubj:pass combined).
#
# nsubj:outer / csubj:outer are their own group instead, since a head can
# legitimately have both a plain nsubj/csubj AND an nsubj:outer/csubj:outer
# at once -- this is the "outer" clausal subject of a copula construction, eg:
#   "The thing to keep in mind is that X, Y, and Z are more dangerous..."
# where "thing" is nsubj:outer of "likely" and "nationalists" is a separate,
# ordinary nsubj of the same head ("likely").
SINGLETON_DEPREL_GROUPS = {
    "subj": {"nsubj", "csubj", "nsubj:pass", "csubj:pass"},
    "subj_outer": {"nsubj:outer", "csubj:outer"},
    "obj": {"obj"},
}

# reverse lookup built from SINGLETON_DEPREL_GROUPS: exact deprel -> group name
DEPREL_TO_GROUP = {deprel: group_name
                    for group_name, deprels in SINGLETON_DEPREL_GROUPS.items()
                    for deprel in deprels}

def find_head_constraint_violations(tree, deprel_strs):
    """
    Finds heads with more than one dependent in the same SINGLETON_DEPREL_GROUPS
    group (eg two nsubj, or an nsubj + csubj, or two obj).

    Returns a list of (head_idx, group_name, [dependent_word_indices]) for
    each violating (head, group) combination.
    """
    deps_per_head = defaultdict(lambda: defaultdict(list))
    for j, deprel in enumerate(deprel_strs):
        word_idx = j + 1
        group_name = DEPREL_TO_GROUP.get(deprel)
        if group_name is not None:
            deps_per_head[tree[word_idx]][group_name].append(word_idx)
    return [(head_idx, group_name, deps)
            for head_idx, groups in deps_per_head.items()
            for group_name, deps in groups.items()
            if len(deps) > 1]

def label_for_arc(label_log_probs, deprel_id_to_group, dep, head, exclude_groups=None):
    """
    Best raw label id (and its log-prob) for a specific (dependent, head) arc.
    exclude_groups, if given, is a set of SINGLETON_DEPREL_GROUPS names whose
    labels are not eligible here.
    """
    row = label_log_probs[dep, head]
    if not exclude_groups:
        best_id = int(np.argmax(row))
        return best_id, row[best_id]
    best_id, best_score = None, -np.inf
    for label_id, score in enumerate(row):
        if deprel_id_to_group[label_id] in exclude_groups:
            continue
        if score > best_score:
            best_score, best_id = score, label_id
    return best_id, best_score

def resolve_head_constraint_violations(scores, label_log_probs, tree, vocab, max_iterations=5):
    """
    Repairs head-constraint violations (more than one nsubj/csubj, or more
    than one obj, attached to the same head) with a single Chu-Liu-Edmonds
    rerun per candidate, rather than separately trying "relabel in place" and
    "reroute structurally" and comparing the two afterward.

    For a violation with offending dependents `deps` at `head_idx`, and for
    each choice of which one of them keeps its original (in-group) label
    unchanged ("keep_idx"), every OTHER offending dependent gets its
    unlabeled score row replaced with a COMBINED row: arc log-prob + best
    available label log-prob, for every candidate head -- except at
    head_idx itself, where the violating group's label is excluded from
    that "best available" search (since this dependent is not the one
    keeping the group label this round).

    Rerunning CLE on this combined-row matrix lets CLE itself decide,
    per dependent, whether the best option is to stay at head_idx with a
    worse (but still legal) label, or to move to a genuinely better head
    elsewhere -- both possibilities are visible in the same score, so one
    CLE call captures what previously took two separate mechanisms (an
    in-place label swap, and a separately-scored structural reroute) and a
    max() over both. Eg for "I gave the cat a fish", if the parser scored
    obj(gave, cat) slightly above iobj(gave, cat), "cat"'s combined row will
    show that staying at "gave" with the iobj label beats any arc it has
    elsewhere, so CLE naturally keeps the attachment and only the label
    changes -- no separate label-swap-only code path is needed to find that.

    This takes k CLE calls for a violation with k offending dependents (one
    per choice of "keep_idx"), not 2k: there is only one repair mechanism,
    not two compared against each other.

    Loops up to max_iterations times to handle the rare case of more than
    one independent violation in the same sentence. After each round,
    excluded_groups records, for each dependent that was pushed away from
    head_idx this round, that it may not use group_name AT head_idx again --
    nothing broader than that. This is deliberately narrow: rather than
    freezing a dependent's entire row (which would also forbid it from ever
    moving again, or from reconsidering other labels elsewhere, even for
    unrelated reasons in some later round), it only rules out recreating
    the EXACT conflict just resolved. A dependent that was pushed away
    remains free to be reconsidered structurally in a later round if that
    round's own repair happens to affect it; it just can't silently drift
    back to (head_idx, group_name) again, since every label lookup for that
    pair consults excluded_groups first. Without this, a later round --
    fixing a different, unrelated violation -- reruns CLE and recomputes
    labels with no memory of what an earlier round decided, and can revert
    it (relabeling a dependent back to its original, violating label; or,
    for a dependent that was rerouted, if it were ever reconsidered again,
    landing back on the same excluded combination).

    Returns (tree, raw_label_ids), where raw_label_ids is a list of raw label
    indices (0..num_relations-1, no VOCAB_PREFIX_SIZE offset), one per real
    word (tree[1:]).
    """
    num_relations = label_log_probs.shape[-1]
    # precompute, once, which SINGLETON_DEPREL_GROUPS group (if any) each raw
    # label id belongs to -- static per vocab, cheap to build per sentence
    deprel_id_to_group = [DEPREL_TO_GROUP.get(vocab.unmap([r + VOCAB_PREFIX_SIZE])[0])
                          for r in range(num_relations)]

    # persistent memory: (dependent_word_idx, head_idx) -> set of groups that
    # dependent may no longer use AT THAT HEAD, because an earlier round
    # already decided it must yield the group there to a different dependent
    excluded_groups = defaultdict(set)

    def label_for(dep, head, extra_exclude=None):
        exclude = excluded_groups.get((dep, head), set())
        if extra_exclude:
            exclude = exclude | {extra_exclude}
        return label_for_arc(label_log_probs, deprel_id_to_group, dep, head, exclude)

    def labels_for_tree(t):
        return [label_for(j, t[j])[0] for j in range(1, len(t))]

    def combined_score(t, raw_ids):
        return sum(scores[j, t[j]] + label_log_probs[j, t[j], raw_ids[j-1]]
                  for j in range(1, len(t)))

    raw_ids = labels_for_tree(tree)

    for _ in range(max_iterations):
        deprel_strs = vocab.unmap([r + VOCAB_PREFIX_SIZE for r in raw_ids])
        violations = find_head_constraint_violations(tree, deprel_strs)
        if not violations:
            break
        head_idx, group_name, deps = violations[0]

        best_candidate = None
        for keep_idx in range(len(deps)):
            trial_scores = scores.copy()
            for k, dep in enumerate(deps):
                if k == keep_idx:
                    continue
                # combined row: best label available at each candidate head,
                # respecting any exclusions persisted from earlier rounds,
                # plus THIS round's exclusion of group_name at head_idx
                best_label_per_head = np.empty(len(tree))
                for h in range(len(tree)):
                    extra = group_name if h == head_idx else None
                    _, best_score = label_for(dep, h, extra_exclude=extra)
                    best_label_per_head[h] = best_score
                trial_scores[dep, :] = scores[dep, :] + best_label_per_head
            trial_tree = chuliu_edmonds_one_root(trial_scores)
            trial_ids = labels_for_tree(trial_tree)
            # if a pushed-away dependent still ended up back at head_idx,
            # its label must additionally exclude group_name THIS round
            # (not yet persisted into excluded_groups)
            for k, dep in enumerate(deps):
                if k != keep_idx and trial_tree[dep] == head_idx:
                    trial_ids[dep - 1], _ = label_for(dep, head_idx, extra_exclude=group_name)
            trial_score = combined_score(trial_tree, trial_ids)
            if best_candidate is None or trial_score > best_candidate[0]:
                best_candidate = (trial_score, trial_tree, trial_ids, keep_idx)

        tree = best_candidate[1]
        raw_ids = best_candidate[2]
        winning_keep_idx = best_candidate[3]

        # persist the exclusion only for the dependents that yielded the
        # group this round -- the one that kept it remains free to use it
        # here again, since it was never part of the conflict being resolved
        for k, dep in enumerate(deps):
            if k != winning_keep_idx:
                excluded_groups[(dep, head_idx)].add(group_name)

    return tree, raw_ids
