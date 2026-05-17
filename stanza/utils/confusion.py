from collections import defaultdict, namedtuple


"""
Named tuple holding per-class precision, recall, and F1 score.

Fields
------
precision : float
    TP / (TP + FP).  0.0 when the class was never predicted.
recall : float
    TP / (TP + FN).  0.0 when the class never appears in the gold data.
f1 : float
    Harmonic mean of precision and recall.  0.0 when both are 0.
"""
F1Result = namedtuple("F1Result", ['precision', 'recall', 'f1'])

def condense_ner_labels(confusion, gold_labels, pred_labels):
    """
    Strip IOB/BIOES prefixes from NER labels and merge the counts.

    Many NER taggers use labels like ``B-PER``, ``I-PER``, ``E-PER``,
    ``S-ORG``, etc.  For display purposes it is often cleaner to collapse
    these into their bare entity types (``PER``, ``ORG``, …).  This
    function rebuilds the confusion matrix under those condensed labels and
    returns updated label lists to match.

    Any label that contains a hyphen is split on the *first* hyphen and only
    the part after the hyphen is kept.  Labels without a hyphen (e.g. ``O``)
    are left as-is.

    Parameters
    ----------
    confusion : dict[str, dict[str, int]]
        The original confusion matrix, indexed as ``confusion[gold][pred]``.
    gold_labels : list[str]
        Ordered list of gold (true) labels currently present in *confusion*.
    pred_labels : list[str]
        Ordered list of predicted labels currently present in *confusion*.

    Returns
    -------
    new_confusion : defaultdict[str, defaultdict[str, int]]
        Rebuilt confusion matrix under the condensed label scheme.
    new_gold_labels : list[str]
        De-duplicated gold labels after prefix removal, in the order they
        were first encountered.
    new_pred_labels : list[str]
        De-duplicated predicted labels after prefix removal, in the order
        they were first encountered.

    Notes
    -----
    This function is called automatically by :func:`format_confusion` when
    the matrix is too wide to display comfortably (more than 150 characters).
    You rarely need to call it directly.
    """
    new_confusion = defaultdict(lambda: defaultdict(int))
    new_gold_labels = []
    new_pred_labels = []

    for l1 in gold_labels:
        if l1.find("-") >= 0:
            new_l1 = l1.split("-", 1)[1]
        else:
            new_l1 = l1
        if new_l1 not in new_gold_labels:
            new_gold_labels.append(new_l1)

        for l2 in pred_labels:
            if l2.find("-") >= 0:
                new_l2 = l2.split("-", 1)[1]
            else:
                new_l2 = l2
            if new_l2 not in new_pred_labels:
                new_pred_labels.append(new_l2)

            old_value = confusion.get(l1, {}).get(l2, 0)
            new_confusion[new_l1][new_l2] = new_confusion[new_l1][new_l2] + old_value

    return new_confusion, new_gold_labels, new_pred_labels


def format_confusion(confusion, labels=None, hide_zeroes=False, hide_blank=False, transpose=False):
    """
    Return a formatted string representation of a confusion matrix.

    Adapted from https://gist.github.com/zachguo/10296432

    The matrix is indexed as ``confusion[gold][pred]``: the outer key is the
    true (gold) label and the inner key is the predicted label.  Rows
    therefore represent gold classes and columns represent predicted classes,
    so a cell ``(i, j)`` contains the number of items whose gold label is
    ``i`` and whose predicted label is ``j``.  The diagonal holds correct
    predictions.

    The corner label ``t\\p`` stands for *true \\ predicted*, or ``p\\t``
    when transposed.

    Parameters
    ----------
    confusion : dict[str, dict[str, int | float]]
        The confusion matrix.  Values may be ints or floats; the formatting
        adapts accordingly.
    labels : list[str] | None, optional
        Explicit ordered list of labels to display.  Both rows and columns
        use this same list.  When *None* (the default), labels are inferred
        from the keys of *confusion* and sorted: alphabetically in general,
        or by entity-type body then BIES prefix for IOB/BIOES label sets,
        with ``O`` always placed first.
    hide_zeroes : bool, optional
        If ``True``, replace zero cells with blank space rather than ``0``.
        Useful for sparse matrices.  Default ``False``.
    hide_blank : bool, optional
        If ``True``, omit any label (row and column) whose entire row is
        zero.  Only has an effect when *labels* is ``None``.  Default
        ``False``.
    transpose : bool, optional
        If ``True``, swap rows and columns so that rows represent predicted
        labels and columns represent gold labels.  The corner label changes
        to ``p\\t``.  Default ``False``.

    Returns
    -------
    str
        A multi-line string ready to be printed.  Each row is right-stripped
        of trailing whitespace.

    Notes
    -----
    When the computed column width multiplied by the number of labels exceeds
    150 characters, :func:`condense_ner_labels` is called automatically to
    strip IOB/BIOES prefixes before rendering, keeping the output legible in
    a standard terminal.

    Examples
    --------
    >>> from collections import defaultdict
    >>> cm = defaultdict(lambda: defaultdict(int))
    >>> cm['NOUN']['NOUN'] = 50
    >>> cm['NOUN']['VERB'] = 3
    >>> cm['VERB']['NOUN'] = 1
    >>> cm['VERB']['VERB'] = 40
    >>> print(format_confusion(cm))
         t\\p  NOUN  VERB
         NOUN    50     3
         VERB     1    40
    """
    def sort_labels(labels):
        """
        Sort label list, respecting BIES ordering and placing O first.

        For plain (non-BIES) label sets, sorts alphabetically.
        For BIES label sets (where every label matches ``X-body`` with
        ``X`` in ``{B, I, E, S}``), sorts first by the body of the label
        and then by the BIES prefix, so all tags for one entity type are
        grouped together.
        ``O`` is always moved to the front of the list when present.
        """
        labels = set(labels)
        if 'O' in labels:
            had_O = True
            labels.remove('O')
        else:
            had_O = False

        if not all(isinstance(x, str) and len(x) > 2 and x[0] in ('B', 'I', 'E', 'S') and x[1] in ('-', '_') for x in labels):
            labels = sorted(labels)
        else:
            # sort first by the body of the label, then by BEIS
            labels = sorted(labels, key=lambda x: (x[2:], x[0]))

        if had_O:
            labels = ['O'] + labels
        return labels

    if transpose:
        new_confusion = defaultdict(lambda: defaultdict(int))
        for label1 in confusion.keys():
            for label2 in confusion[label1].keys():
                new_confusion[label2][label1] = confusion[label1][label2]
        confusion = new_confusion

    if labels is None:
        gold_labels = set(confusion.keys())
        if hide_blank:
            gold_labels = set(x for x in gold_labels if any(confusion[x][key] != 0 for key in confusion[x].keys()))
        pred_labels = set()
        for key in confusion.keys():
            if hide_blank:
                new_pred_labels = set(x for x in confusion[key].keys() if confusion[key][x] != 0)
            else:
                new_pred_labels = confusion[key].keys()
            pred_labels = pred_labels.union(new_pred_labels)
        if not hide_blank:
            gold_labels = gold_labels.union(pred_labels)
            pred_labels = gold_labels
        gold_labels = sort_labels(gold_labels)
        pred_labels = sort_labels(pred_labels)
    else:
        gold_labels = labels
        pred_labels = labels

    columnwidth = max([len(str(x)) for x in pred_labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # If the numbers are all ints, no need to include the .0 at the end of each entry
    all_ints = True
    for i, label1 in enumerate(gold_labels):
        for j, label2 in enumerate(pred_labels):
            if not isinstance(confusion.get(label1, {}).get(label2, 0), int):
                all_ints = False
                break
        if not all_ints:
            break

    if all_ints:
        format_cell = lambda confusion_cell: "%{0}d".format(columnwidth) % confusion_cell
    else:
        format_cell = lambda confusion_cell: "%{0}.1f".format(columnwidth) % confusion_cell

    # make sure the columnwidth can handle long numbers
    for i, label1 in enumerate(gold_labels):
        for j, label2 in enumerate(pred_labels):
            cell = confusion.get(label1, {}).get(label2, 0)
            columnwidth = max(columnwidth, len(format_cell(cell)))

    # if this is an NER confusion matrix (well, if it has - in the labels)
    # try to drop a bunch of labels to make the matrix easier to display
    if columnwidth * len(pred_labels) > 150:
        confusion, gold_labels, pred_labels = condense_ner_labels(confusion, gold_labels, pred_labels)

    # Print header
    if transpose:
        corner_label = "p\\t"
    else:
        corner_label = "t\\p"
    fst_empty_cell = (columnwidth-3)//2 * " " + corner_label + (columnwidth-3)//2 * " "
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    header = "    " + fst_empty_cell + " "
    for label in pred_labels:
        header = header + "%{0}s ".format(columnwidth) % str(label)
    text = [header.rstrip()]

    # Print rows
    for i, label1 in enumerate(gold_labels):
        row = "    %{0}s ".format(columnwidth) % str(label1)
        for j, label2 in enumerate(pred_labels):
            confusion_cell = confusion.get(label1, {}).get(label2, 0)
            cell = format_cell(confusion_cell)
            if hide_zeroes:
                cell = cell if confusion_cell else empty_cell
            row = row + cell + " "
        text.append(row.rstrip())

    return "\n".join(text)


def confusion_to_accuracy(confusion_matrix):
    """
    Compute overall accuracy from a confusion matrix.

    Counts all cells on the diagonal as correct predictions and all
    off-diagonal cells as errors.

    Parameters
    ----------
    confusion_matrix : dict[str, dict[str, int]]
        Confusion matrix indexed as ``confusion_matrix[gold][pred]``.

    Returns
    -------
    correct : int
        Number of items where gold label == predicted label (diagonal sum).
    total : int
        Total number of items (correct + incorrect).

    Examples
    --------
    >>> cm = {'NOUN': {'NOUN': 50, 'VERB': 3}, 'VERB': {'NOUN': 1, 'VERB': 40}}
    >>> correct, total = confusion_to_accuracy(cm)
    >>> correct, total
    (90, 94)
    >>> correct / total
    0.9574468085106383
    """
    correct = 0
    total = 0
    for l1 in confusion_matrix.keys():
        for l2 in confusion_matrix[l1].keys():
            if l1 == l2:
                correct = correct + confusion_matrix[l1][l2]
            else:
                total = total + confusion_matrix[l1][l2]
    return correct, (correct + total)


def confusion_to_f1(confusion_matrix):
    """
    Compute per-class precision, recall, and F1 from a confusion matrix.

    For each label ``k`` the standard one-vs-rest counts are derived from
    the full matrix:

    - **TP**: ``confusion_matrix[k][k]``
    - **FN**: sum of ``confusion_matrix[k][k2]`` for all ``k2 != k``
      (gold is ``k`` but predicted as something else)
    - **FP**: sum of ``confusion_matrix[k2][k]`` for all ``k2 != k``
      (predicted as ``k`` but gold is something else)

    Parameters
    ----------
    confusion_matrix : dict[str, dict[str, int]]
        Confusion matrix indexed as ``confusion_matrix[gold][pred]``.
        All labels that appear as either a gold or predicted label are
        scored; missing cells are treated as 0.

    Returns
    -------
    dict[str, F1Result]
        Mapping from label string to an :class:`F1Result` named tuple with
        fields ``precision``, ``recall``, and ``f1``, all floats.
        Precision is 0.0 for labels never predicted; recall is 0.0 for
        labels never present in the gold data; F1 is 0.0 when both are 0.

    Examples
    --------
    >>> cm = {'NOUN': {'NOUN': 50, 'VERB': 3}, 'VERB': {'NOUN': 1, 'VERB': 40}}
    >>> results = confusion_to_f1(cm)
    >>> results['NOUN'].f1
    0.9615384615384616
    >>> results['VERB'].f1
    0.9523809523809524
    """
    results = {}
    keys = set()
    for k in confusion_matrix.keys():
        keys.add(k)
        for k2 in confusion_matrix.get(k).keys():
            keys.add(k2)

    sum_f1 = 0
    for k in keys:
        tp = 0
        fn = 0
        fp = 0
        for k2 in keys:
            if k == k2:
                tp = confusion_matrix.get(k, {}).get(k, 0)
            else:
                fn = fn + confusion_matrix.get(k, {}).get(k2, 0)
                fp = fp + confusion_matrix.get(k2, {}).get(k, 0)

        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        results[k] = F1Result(precision, recall, f1)

    return results


def confusion_to_macro_f1(confusion_matrix):
    """
    Compute macro-averaged F1 from a confusion matrix.

    Macro F1 is the unweighted mean of per-class F1 scores: every class
    contributes equally regardless of how many instances it has.  This
    makes it sensitive to performance on rare classes.

    Parameters
    ----------
    confusion_matrix : dict[str, dict[str, int]]
        Confusion matrix indexed as ``confusion_matrix[gold][pred]``.

    Returns
    -------
    float
        Macro-averaged F1 score, in the range [0.0, 1.0].

    See Also
    --------
    confusion_to_f1 : per-class F1 scores used internally.
    confusion_to_weighted_f1 : instance-count-weighted alternative.

    Examples
    --------
    >>> cm = {'NOUN': {'NOUN': 50, 'VERB': 3}, 'VERB': {'NOUN': 1, 'VERB': 40}}
    >>> confusion_to_macro_f1(cm)
    0.956959706959707
    """
    sum_f1 = 0.0
    results = confusion_to_f1(confusion_matrix)
    for k in results.keys():
        sum_f1 = sum_f1 + results[k].f1
    return sum_f1 / len(results)


def confusion_to_weighted_f1(confusion_matrix, exclude=None):
    """
    Compute weighted (instance-count-weighted) F1 from a confusion matrix.

    Each class's F1 score is weighted by the number of gold instances of
    that class (i.e. the row sum), so frequent classes have more influence
    on the result than rare ones.

    Parameters
    ----------
    confusion_matrix : dict[str, dict[str, int]]
        Confusion matrix indexed as ``confusion_matrix[gold][pred]``.
    exclude : set[str] | list[str] | None, optional
        Labels to omit from the weighted average entirely (both from the
        numerator and the denominator).  Useful for excluding a dominant
        majority class such as ``O`` in NER tasks.  Default ``None``
        (include all labels).

    Returns
    -------
    float
        Weighted F1 score, in the range [0.0, 1.0].

    See Also
    --------
    confusion_to_f1 : per-class F1 scores used internally.
    confusion_to_macro_f1 : unweighted alternative.

    Examples
    --------
    >>> cm = {'NOUN': {'NOUN': 50, 'VERB': 3}, 'VERB': {'NOUN': 1, 'VERB': 40}}
    >>> confusion_to_weighted_f1(cm)
    0.9575442288208247
    >>> confusion_to_weighted_f1(cm, exclude={'NOUN'})
    0.9523809523809524
    """
    results = confusion_to_f1(confusion_matrix)
    sum_f1 = 0.0
    total_items = 0
    for k in results.keys():
        if exclude is not None and k in exclude:
            continue
        k_items = sum(confusion_matrix.get(k, {}).values())
        total_items += k_items
        sum_f1 += results[k].f1 * k_items
    return sum_f1 / total_items
