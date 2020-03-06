"""
An NER scorer that calculates F1 score given gold and predicted tags.
"""
import sys
import os
import logging
from collections import Counter

from stanza.models.ner.utils import decode_from_bioes

logger = logging.getLogger('stanza')

def score_by_entity(pred_tag_sequences, gold_tag_sequences, verbose=True):
    """ Score predicted tags at the entity level.

    Args:
        pred_tags_sequences: a list of list of predicted tags for each word
        gold_tags_sequences: a list of list of gold tags for each word
        verbose: print log with results
    
    Returns:
        Precision, recall and F1 scores.
    """
    assert(len(gold_tag_sequences) == len(pred_tag_sequences)), \
        "Number of predicted tag sequences does not match gold sequences."
    
    def decode_all(tag_sequences):
        # decode from all sequences, each sequence with a unique id
        ents = []
        for sent_id, tags in enumerate(tag_sequences):
            for ent in decode_from_bioes(tags):
                ent['sent_id'] = sent_id
                ents += [ent]
        return ents

    gold_ents = decode_all(gold_tag_sequences)
    pred_ents = decode_all(pred_tag_sequences)

    # scoring
    correct_by_type = Counter()
    guessed_by_type = Counter()
    gold_by_type = Counter()

    for p in pred_ents:
        guessed_by_type[p['type']] += 1
        if p in gold_ents:
            correct_by_type[p['type']] += 1
    for g in gold_ents:
        gold_by_type[g['type']] += 1
    
    prec_micro = 0.0
    if sum(guessed_by_type.values()) > 0:
        prec_micro = sum(correct_by_type.values()) * 1.0 / sum(guessed_by_type.values())
    rec_micro = 0.0
    if sum(gold_by_type.values()) > 0:
        rec_micro = sum(correct_by_type.values()) * 1.0 / sum(gold_by_type.values())
    f_micro = 0.0
    if prec_micro + rec_micro > 0:
        f_micro = 2.0 * prec_micro * rec_micro / (prec_micro + rec_micro)
    
    if verbose:
        logger.info("Prec.\tRec.\tF1")
        logger.info("{:.2f}\t{:.2f}\t{:.2f}".format( \
            prec_micro*100, rec_micro*100, f_micro*100))
    return prec_micro, rec_micro, f_micro


def score_by_token(pred_tag_sequences, gold_tag_sequences, verbose=True):
    """ Score predicted tags at the token level.

    Args:
        pred_tags_sequences: a list of list of predicted tags for each word
        gold_tags_sequences: a list of list of gold tags for each word
        verbose: print log with results
    
    Returns:
        Precision, recall and F1 scores.
    """
    assert(len(gold_tag_sequences) == len(pred_tag_sequences)), \
        "Number of predicted tag sequences does not match gold sequences."
    
    correct_by_tag = Counter()
    guessed_by_tag = Counter()
    gold_by_tag = Counter()

    for gold_tags, pred_tags in zip(gold_tag_sequences, pred_tag_sequences):
        assert(len(gold_tags) == len(pred_tags)), \
            "Number of predicted tags does not match gold."
        for g, p in zip(gold_tags, pred_tags):
            if g == 'O' and p == 'O':
                continue
            elif g == 'O' and p != 'O':
                guessed_by_tag[p] += 1
            elif g != 'O' and p == 'O':
                gold_by_tag[g] += 1
            else:
                guessed_by_tag[p] += 1
                gold_by_tag[p] += 1
                if g == p:
                    correct_by_tag[p] += 1
    
    prec_micro = 0.0
    if sum(guessed_by_tag.values()) > 0:
        prec_micro = sum(correct_by_tag.values()) * 1.0 / sum(guessed_by_tag.values())
    rec_micro = 0.0
    if sum(gold_by_tag.values()) > 0:
        rec_micro = sum(correct_by_tag.values()) * 1.0 / sum(gold_by_tag.values())
    f_micro = 0.0
    if prec_micro + rec_micro > 0:
        f_micro = 2.0 * prec_micro * rec_micro / (prec_micro + rec_micro)
    
    if verbose:
        logger.info("Prec.\tRec.\tF1")
        logger.info("{:.2f}\t{:.2f}\t{:.2f}".format( \
            prec_micro*100, rec_micro*100, f_micro*100))
    return prec_micro, rec_micro, f_micro

def test():
    pred_sequences = [['O', 'S-LOC', 'O', 'O', 'B-PER', 'E-PER'],
                    ['O', 'S-MISC', 'O', 'E-ORG', 'O', 'B-PER', 'I-PER', 'E-PER']]
    gold_sequences = [['O', 'B-LOC', 'E-LOC', 'O', 'B-PER', 'E-PER'],
                    ['O', 'S-MISC', 'B-ORG', 'E-ORG', 'O', 'B-PER', 'E-PER', 'S-LOC']]
    print(score_by_token(pred_sequences, gold_sequences))
    print(score_by_entity(pred_sequences, gold_sequences))

if __name__ == '__main__':
    test()

