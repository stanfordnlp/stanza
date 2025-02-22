"""
A tool with an initial set of error analysis for in-order parsing.

Analyzes the first error created in the parser

TODO: there are more errors to analyze, and see below for a case where attachment is misidentified as bracket
"""

from enum import Enum

from stanza.models.constituency.dynamic_oracle import advance_past_constituents
from stanza.models.constituency.parse_transitions import Shift, CompoundUnary, OpenConstituent, CloseConstituent, TransitionScheme, Finalize
from stanza.models.constituency.transition_sequence import build_sequence

class FirstError(Enum):
    NONE                        = 1
    UNKNOWN                     = 2
    WRONG_OPEN_LABEL_NO_CASCADE = 3
    WRONG_OPEN_LABEL_CASCADE    = 4
    WRONG_SUBTREE_NO_CASCADE    = 5
    WRONG_SUBTREE_CASCADE       = 6
    EXTRA_ATTACHMENT            = 7
    MISSING_ATTACHMENT          = 8
    EXTRA_BRACKET_NO_CASCADE    = 9
    EXTRA_BRACKET_CASCADE       = 10
    MISSING_BRACKET_NO_CASCADE  = 11
    MISSING_BRACKET_CASCADE     = 12

def advance_past_unaries(sequence, idx):
    while idx + 2 < len(sequence) and isinstance(sequence[idx+1], OpenConstituent) and isinstance(sequence[idx+2], CloseConstituent):
        idx += 2
    return idx

def check_attachment_error(gold_sequence, pred_sequence, idx, error_type):
    # this will find the Close that closes the constituent that
    # was just closed in the gold sequence
    # hopefully we will have built the same constituent(s)
    # that were built after the gold sequence closed
    pred_close_idx = advance_past_constituents(pred_sequence, idx)
    gold_close_idx = pred_close_idx + 1
    #gold_close_idx = find_in_order_constituent_end(gold_sequence, idx+1) # +1 represents, start counting from the Shift
    #pred_close_idx = find_in_order_constituent_end(pred_sequence, idx)
    if gold_sequence[idx+1:gold_close_idx] != pred_sequence[idx:pred_close_idx]:
        return FirstError.UNKNOWN, None, None
    if (isinstance(gold_sequence[gold_close_idx], CloseConstituent) and
        isinstance(pred_sequence[pred_close_idx], CloseConstituent) and
        isinstance(pred_sequence[pred_close_idx+1], CloseConstituent)):
        #print(error_type)
        #print(gold_sequence)
        #print(gold_close_idx)
        #print(gold_sequence[gold_close_idx:])
        #print(pred_sequence)
        #print(pred_close_idx)
        #print(pred_sequence[pred_close_idx+1:])
        #print("=================")
        return error_type, gold_close_idx, pred_close_idx+1

    return None

def analyze_tree(gold_tree, pred_tree):
    if gold_tree == pred_tree:
        return FirstError.NONE, None, None

    gold_sequence = build_sequence(gold_tree, TransitionScheme.IN_ORDER)
    pred_sequence = build_sequence(pred_tree, TransitionScheme.IN_ORDER)

    for idx, (gold_trans, pred_trans) in enumerate(zip(gold_sequence, pred_sequence)):
        if gold_trans != pred_trans:
            break
    else:
        # guess only the tags were different?
        return FirstError.NONE, None, None

    if isinstance(gold_trans, CloseConstituent) and isinstance(pred_trans, Shift) and isinstance(gold_sequence[idx + 1], Shift):
        # perhaps this is an attachment error
        # we can see if the exact same sequence of moved constituent was built
        error = check_attachment_error(gold_sequence, pred_sequence, idx, FirstError.EXTRA_ATTACHMENT)
        if error is not None:
            return error

    if isinstance(pred_trans, CloseConstituent) and isinstance(gold_trans, Shift) and isinstance(pred_sequence[idx + 1], Shift):
        # perhaps this is an attachment error
        # we can see if the exact same sequence of moved constituent was built
        error = check_attachment_error(pred_sequence, gold_sequence, idx, FirstError.MISSING_ATTACHMENT)
        if error is not None:
            # flip the gold & pred indices of the relevant locations
            return error[0], error[2], error[1]

    if isinstance(gold_trans, OpenConstituent) and isinstance(pred_trans, OpenConstituent):
        gold_close_idx = advance_past_constituents(gold_sequence, idx+1)
        gold_unary_idx = advance_past_unaries(gold_sequence, gold_close_idx)

        pred_close_idx = advance_past_constituents(pred_sequence, idx+1)
        pred_unary_idx = advance_past_unaries(pred_sequence, pred_close_idx)
        if gold_sequence[idx+1:gold_close_idx] != pred_sequence[idx+1:pred_close_idx]:
            # maybe the internal structure is the same?
            # actually, if the number of shifts inside is the same,
            # then the words shifted were the same,
            # so the internal structure is different but the parser
            # is getting back on track after closing
            if (sum(isinstance(gt, Shift) for gt in gold_sequence[idx+1:gold_close_idx]) ==
                sum(isinstance(pt, Shift) for pt in pred_sequence[idx+1:pred_close_idx])):
                if gold_sequence[gold_unary_idx:] == pred_sequence[pred_unary_idx:]:
                    return FirstError.WRONG_SUBTREE_NO_CASCADE, gold_unary_idx, pred_unary_idx
                else:
                    return FirstError.WRONG_SUBTREE_CASCADE, gold_unary_idx, pred_unary_idx
            return FirstError.UNKNOWN, None, None
        # at this point, everything is the same aside from the open being a different label
        if gold_sequence[gold_unary_idx:] == pred_sequence[pred_unary_idx:]:
            return FirstError.WRONG_OPEN_LABEL_NO_CASCADE, gold_unary_idx, pred_unary_idx
        else:
            return FirstError.WRONG_OPEN_LABEL_CASCADE, gold_unary_idx, pred_unary_idx

    if isinstance(gold_trans, Shift) and isinstance(pred_trans, OpenConstituent):
        # This could be a case of an extra bracket inserted into the tree
        # We will search for the end of the new bracket, then check if
        # all the children were properly constructed the way the gold sequence wanted to,
        # aside from the extra bracket

        # TODO: this is also capturing what are effectively attachment
        # errors in the case of nested nodes (S over S) where a node
        # at the start should have been connected to the below node
        #   gold:
        #  (ROOT
        #    (S
        #      (S
        #        (`` ``)
        #        (NP (PRP$ Our) (NN balance) (NNS sheets))
        #        (VP
        #          (VBP look)
        #          (SBAR
        #            (IN like)
        #            (S
        #              (NP (PRP they))
        #              (VP
        #                (VBD came)
        #                (PP
        #                  (IN from)
        #                  (NP
        #                    (NP (NNP Alice) (POS 's))
        #                    (NN wonderland)))))))
        #        (, ,)
        #        ('' ''))
        #      (NP (NNP Mr.) (NNP Fromstein))
        #      (VP (VBD said))
        #      (. .)))
        #
        #  pred:
        #  (ROOT
        #    (S
        #      (`` ``)
        #      (S
        #        (NP (PRP$ Our) (NN balance) (NNS sheets))
        #        (VP
        #          (VBP look)
        #          (SBAR
        #            (IN like)
        #            (S
        #              (NP (PRP they))
        #              (VP
        #                (VBD came)
        #                (PP
        #                  (IN from)
        #                  (NP
        #                    (NP (NNP Alice) (POS 's))
        #                    (NN wonderland))))))))
        #      (, ,)
        #      ('' '')
        #      (NP (NNP Mr.) (NNP Fromstein))
        #      (VP (VBD said))
        #      (. .)))

        pred_close_idx = advance_past_constituents(pred_sequence, idx+1)
        pred_unary_idx = advance_past_unaries(pred_sequence, pred_close_idx + 1)
        if gold_sequence[idx:pred_close_idx-1] == pred_sequence[idx+1:pred_close_idx]:
            #print(gold_sequence)
            #print(pred_sequence)
            #print(idx, pred_close_idx)
            #print("{:P}".format(gold_tree))
            #print("{:P}".format(pred_tree))
            #print("=================")
            gold_unary_idx = advance_past_unaries(gold_sequence, pred_close_idx - 1)
            if pred_sequence[pred_unary_idx:] == gold_sequence[gold_unary_idx:]:
                return FirstError.EXTRA_BRACKET_NO_CASCADE, pred_close_idx, pred_close_idx+2
            else:
                return FirstError.EXTRA_BRACKET_CASCADE, pred_close_idx, pred_close_idx+2

    if isinstance(pred_trans, Shift) and isinstance(gold_trans, OpenConstituent):
        # presumably this has attachment errors as well, similarly to EXTRA_BRACKET
        gold_close_idx = advance_past_constituents(gold_sequence, idx+1)
        gold_unary_idx = advance_past_unaries(gold_sequence, gold_close_idx + 1)
        if pred_sequence[idx:gold_close_idx-1] == gold_sequence[idx+1:gold_close_idx]:
            #print(gold_sequence)
            #print(pred_sequence)
            #print(idx, gold_close_idx)
            #print("{:P}".format(gold_tree))
            #print("{:P}".format(pred_tree))
            #print("=================")
            pred_unary_idx = advance_past_unaries(pred_sequence, gold_close_idx - 1)
            if pred_sequence[pred_unary_idx:] == gold_sequence[gold_unary_idx:]:
                return FirstError.MISSING_BRACKET_NO_CASCADE, gold_unary_idx, pred_unary_idx
            else:
                return FirstError.MISSING_BRACKET_CASCADE, gold_unary_idx, pred_unary_idx


    return FirstError.UNKNOWN, None, None
