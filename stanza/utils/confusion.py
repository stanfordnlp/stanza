
from collections import defaultdict, namedtuple

F1Result = namedtuple("F1Result", ['precision', 'recall', 'f1'])

def condense_ner_labels(confusion, labels):
    new_confusion = defaultdict(lambda: defaultdict(int))
    new_labels = []
    for l1 in labels:
        if l1.find("-") >= 0:
            new_l1 = l1.split("-", 1)[1]
        else:
            new_l1 = l1
        if new_l1 not in new_labels:
            new_labels.append(new_l1)
        for l2 in labels:
            if l2.find("-") >= 0:
                new_l2 = l2.split("-", 1)[1]
            else:
                new_l2 = l2

            old_value = confusion.get(l1, {}).get(l2, 0)
            new_confusion[new_l1][new_l2] = new_confusion[new_l1][new_l2] + old_value
    return new_confusion, new_labels


def format_confusion(confusion, labels=None, hide_zeroes=False):
    """
    pretty print for confusion matrixes
    adapted from https://gist.github.com/zachguo/10296432

    The matrix should look like this:
      confusion[gold][pred]
    """
    def sort_labels(labels):
        if not all(len(x) > 2 and x[0] in ('B', 'I', 'E', 'S') and x[1] in ('-', '_') for x in labels):
            return sorted(labels)

        # sort first by the body of the lable, then by BEIS
        return sorted(labels, key=lambda x: (x[2:], x[0]))

    if labels is None:
        labels = set(confusion.keys())
        for key in confusion.keys():
            labels = labels.union(confusion[key].keys())
        if 'O' in labels:
            labels.remove('O')
            labels = ['O'] + sort_labels(labels)
        else:
            labels = sort_labels(labels)

    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # If the numbers are all ints, no need to include the .0 at the end of each entry
    all_ints = True
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
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
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            cell = confusion.get(label1, {}).get(label2, 0)
            columnwidth = max(columnwidth, len(format_cell(cell)))

    # if this is an NER confusion matrix (well, if it has - in the labels)
    # try to drop a bunch of labels to make the matrix easier to display
    if columnwidth * len(labels) > 150:
        confusion, labels = condense_ner_labels(confusion, labels)

    # Print header
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    header = "    " + fst_empty_cell + " "
    for label in labels:
        header = header + "%{0}s ".format(columnwidth) % label
    text = [header.rstrip()]

    # Print rows
    for i, label1 in enumerate(labels):
        row = "    %{0}s ".format(columnwidth) % label1
        for j, label2 in enumerate(labels):
            confusion_cell = confusion.get(label1, {}).get(label2, 0)
            cell = format_cell(confusion_cell)
            if hide_zeroes:
                cell = cell if confusion_cell else empty_cell
            row = row + cell + " "
        text.append(row.rstrip())
    return "\n".join(text)


def confusion_to_accuracy(confusion_matrix):
    """
    Given a confusion dictionary, return correct, total
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
    Return the macro f1 for a confusion matrix.
    """
    sum_f1 = 0.0
    results = confusion_to_f1(confusion_matrix)
    for k in results.keys():
        sum_f1 = sum_f1 + results[k].f1

    return sum_f1 / len(results)

def confusion_to_weighted_f1(confusion_matrix, exclude=None):
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
