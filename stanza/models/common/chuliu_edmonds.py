# Adapted from Tim's code here: https://github.com/tdozat/Parser-v3/blob/master/scripts/chuliu_edmonds.py

import numpy as np

def tarjan(tree):
    """"""

    indices = -np.ones_like(tree)
    lowlinks = -np.ones_like(tree)
    onstack = np.zeros_like(tree, dtype=bool)
    stack = list()
    _index = [0]
    cycles = []
    #-------------------------------------------------------------
    def strong_connect(i):
        _index[0] += 1
        index = _index[-1]
        indices[i] = lowlinks[i] = index - 1
        stack.append(i)
        onstack[i] = True
        dependents = np.where(np.equal(tree, i))[0]
        for j in dependents:
            if indices[j] == -1:
                strong_connect(j)
                lowlinks[i] = min(lowlinks[i], lowlinks[j])
            elif onstack[j]:
                lowlinks[i] = min(lowlinks[i], indices[j])

        # There's a cycle!
        if lowlinks[i] == indices[i]:
            cycle = np.zeros_like(indices, dtype=bool)
            while stack[-1] != i:
                j = stack.pop()
                onstack[j] = False
                cycle[j] = True
            stack.pop()
            onstack[i] = False
            cycle[i] = True
            if cycle.sum() > 1:
                cycles.append(cycle)
        return
    #-------------------------------------------------------------
    for i in range(len(tree)):
        if indices[i] == -1:
            strong_connect(i)
    return cycles

def process_cycle(tree, cycle, scores):
    """
    Build a subproblem with one cycle broken
    """
    # indices of cycle in original tree; (c) in t
    cycle_locs = np.where(cycle)[0]
    # heads of cycle in original tree; (c) in t
    cycle_subtree = tree[cycle]
    # scores of cycle in original tree; (c) in R
    cycle_scores = scores[cycle, cycle_subtree]
    # total score of cycle; () in R
    cycle_score = cycle_scores.sum()

    # locations of noncycle; (t) in [0,1]
    noncycle = np.logical_not(cycle)
    # indices of noncycle in original tree; (n) in t
    noncycle_locs = np.where(noncycle)[0]
    #print(cycle_locs, noncycle_locs)

    # scores of cycle's potential heads; (c x n) - (c) + () -> (n x c) in R
    metanode_head_scores = scores[cycle][:,noncycle] - cycle_scores[:,None] + cycle_score
    # scores of cycle's potential dependents; (n x c) in R
    metanode_dep_scores = scores[noncycle][:,cycle]
    # best noncycle head for each cycle dependent; (n) in c
    metanode_heads = np.argmax(metanode_head_scores, axis=0)
    # best cycle head for each noncycle dependent; (n) in c
    metanode_deps = np.argmax(metanode_dep_scores, axis=1)

    # scores of noncycle graph; (n x n) in R
    subscores = scores[noncycle][:,noncycle]
    # pad to contracted graph; (n+1 x n+1) in R
    subscores = np.pad(subscores, ( (0,1) , (0,1) ), 'constant')
    # set the contracted graph scores of cycle's potential heads; (c x n)[:, (n) in n] in R -> (n) in R
    subscores[-1, :-1] = metanode_head_scores[metanode_heads, np.arange(len(noncycle_locs))]
    # set the contracted graph scores of cycle's potential dependents; (n x c)[(n) in n] in R-> (n) in R
    subscores[:-1,-1] = metanode_dep_scores[np.arange(len(noncycle_locs)), metanode_deps]
    return subscores, cycle_locs, noncycle_locs, metanode_heads, metanode_deps


def expand_contracted_tree(tree, contracted_tree, cycle_locs, noncycle_locs, metanode_heads, metanode_deps):
    """
    Given a partially solved tree with a cycle and a solved subproblem
    for the cycle, build a larger solution without the cycle
    """
    # head of the cycle; () in n
    #print(contracted_tree)
    cycle_head = contracted_tree[-1]
    # fixed tree: (n) in n+1
    contracted_tree = contracted_tree[:-1]
    # initialize new tree; (t) in 0
    new_tree = -np.ones_like(tree)
    #print(0, new_tree)
    # fixed tree with no heads coming from the cycle: (n) in [0,1]
    contracted_subtree = contracted_tree < len(contracted_tree)
    # add the nodes to the new tree (t)[(n)[(n) in [0,1]] in t] in t = (n)[(n)[(n) in [0,1]] in n] in t
    new_tree[noncycle_locs[contracted_subtree]] = noncycle_locs[contracted_tree[contracted_subtree]]
    #print(1, new_tree)
    # fixed tree with heads coming from the cycle: (n) in [0,1]
    contracted_subtree = np.logical_not(contracted_subtree)
    # add the nodes to the tree (t)[(n)[(n) in [0,1]] in t] in t = (c)[(n)[(n) in [0,1]] in c] in t
    new_tree[noncycle_locs[contracted_subtree]] = cycle_locs[metanode_deps[contracted_subtree]]
    #print(2, new_tree)
    # add the old cycle to the tree; (t)[(c) in t] in t = (t)[(c) in t] in t
    new_tree[cycle_locs] = tree[cycle_locs]
    #print(3, new_tree)
    # root of the cycle; (n)[() in n] in c = () in c
    cycle_root = metanode_heads[cycle_head]
    # add the root of the cycle to the new tree; (t)[(c)[() in c] in t] = (c)[() in c]
    new_tree[cycle_locs[cycle_root]] = noncycle_locs[cycle_head]
    #print(4, new_tree)
    return new_tree

def prepare_scores(scores):
    """
    Alter the scores matrix to avoid self loops and handle the root
    """
    # prevent self-loops, set up the root location
    np.fill_diagonal(scores, -float('inf')) # prevent self-loops
    scores[0] = -float('inf')
    scores[0,0] = 0

def chuliu_edmonds(scores):
    subtree_stack = []

    prepare_scores(scores)
    tree = np.argmax(scores, axis=1)
    cycles = tarjan(tree)

    #print(scores)
    #print(cycles)

    # recursive implementation:
    #if cycles:
    #    # t = len(tree); c = len(cycle); n = len(noncycle)
    #    # cycles.pop(): locations of cycle; (t) in [0,1]
    #    subscores, cycle_locs, noncycle_locs, metanode_heads, metanode_deps = process_cycle(tree, cycles.pop(), scores)
    #    # MST with contraction; (n+1) in n+1
    #    contracted_tree = chuliu_edmonds(subscores)
    #    tree = expand_contracted_tree(tree, contracted_tree, cycle_locs, noncycle_locs, metanode_heads, metanode_deps)
    # unfortunately, while the recursion is simpler to understand, it can get too deep for python's stack limit
    # so instead we make our own recursion, with blackjack and (you know how it goes)

    while cycles:
        # t = len(tree); c = len(cycle); n = len(noncycle)
        # cycles.pop(): locations of cycle; (t) in [0,1]
        subscores, cycle_locs, noncycle_locs, metanode_heads, metanode_deps = process_cycle(tree, cycles.pop(), scores)
        subtree_stack.append((tree, cycles, scores, subscores, cycle_locs, noncycle_locs, metanode_heads, metanode_deps))

        scores = subscores
        prepare_scores(scores)
        tree = np.argmax(scores, axis=1)
        cycles = tarjan(tree)

    while len(subtree_stack) > 0:
        contracted_tree = tree
        (tree, cycles, scores, subscores, cycle_locs, noncycle_locs, metanode_heads, metanode_deps) = subtree_stack.pop()
        tree = expand_contracted_tree(tree, contracted_tree, cycle_locs, noncycle_locs, metanode_heads, metanode_deps)

    return tree

#===============================================================
def chuliu_edmonds_one_root(scores):
    """"""

    scores = scores.astype(np.float64)
    tree = chuliu_edmonds(scores)
    roots_to_try = np.where(np.equal(tree[1:], 0))[0]+1
    if len(roots_to_try) == 1:
        return tree

    #-------------------------------------------------------------
    def set_root(scores, root):
        root_score = scores[root,0]
        scores = np.array(scores)
        scores[1:,0] = -float('inf')
        scores[root] = -float('inf')
        scores[root,0] = 0
        return scores, root_score
    #-------------------------------------------------------------

    best_score, best_tree = -np.inf, None # This is what's causing it to crash
    for root in roots_to_try:
        _scores, root_score = set_root(scores, root)
        _tree = chuliu_edmonds(_scores)
        tree_probs = _scores[np.arange(len(_scores)), _tree]
        tree_score = (tree_probs).sum()+(root_score) if (tree_probs > -np.inf).all() else -np.inf
        if tree_score > best_score:
            best_score = tree_score
            best_tree = _tree
    try:
        assert best_tree is not None
    except:
        with open('debug.log', 'w') as f:
            f.write('{}: {}, {}\n'.format(tree, scores, roots_to_try))
            f.write('{}: {}, {}, {}\n'.format(_tree, _scores, tree_probs, tree_score))
        raise
    return best_tree
