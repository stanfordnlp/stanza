""" Describes ClusterChecker, a class used to retrieve LEA scores.
See aclweb.org/anthology/P16-1060.pdf. """

from typing import Hashable, List, Tuple

from stanza.models.coref.const import EPSILON
import numpy as np

import math
import logging

logger = logging.getLogger('stanza')


class ClusterChecker:
    """ Collects information on gold and predicted clusters across documents.
    Can be used to retrieve weighted LEA-score for them.
    """
    def __init__(self):
        self._lea_precision = 0.0
        self._lea_recall = 0.0
        self._lea_precision_weighting = 0.0
        self._lea_recall_weighting = 0.0
        self._num_preds = 0.0

        # muc
        self._muc_precision = 0.0
        self._muc_recall = 0.0

        # b3
        self._b3_precision = 0.0
        self._b3_recall = 0.0

        # ceafe
        self._ceafe_precision = 0.0
        self._ceafe_recall = 0.0

    @staticmethod
    def _f1(p,r):
        return (p * r) / (p+r + EPSILON) * 2
    
    def add_predictions(self,
                        gold_clusters: List[List[Hashable]],
                        pred_clusters: List[List[Hashable]]):
        """
        Calculates LEA for the document's clusters and stores them to later
        output weighted LEA across documents.

        Returns:
            LEA score for the document as a tuple of (f1, precision, recall)
        """

        # if len(gold_clusters) == 0:
            # breakpoint()

        self._num_preds += 1
        
        recall, r_weight = ClusterChecker._lea(gold_clusters, pred_clusters)
        precision, p_weight = ClusterChecker._lea(pred_clusters, gold_clusters)

        self._muc_recall +=  ClusterChecker._muc(gold_clusters, pred_clusters)
        self._muc_precision += ClusterChecker._muc(pred_clusters, gold_clusters)

        self._b3_recall += ClusterChecker._b3(gold_clusters, pred_clusters)
        self._b3_precision += ClusterChecker._b3(pred_clusters, gold_clusters)

        ceafe_precision, ceafe_recall = ClusterChecker._ceafe(pred_clusters, gold_clusters)
        if math.isnan(ceafe_precision) and len(gold_clusters) > 0:
            # because our model predicted no clusters
            ceafe_precision = 0.0

        self._ceafe_precision += ceafe_precision
        self._ceafe_recall += ceafe_recall

        self._lea_recall += recall
        self._lea_recall_weighting += r_weight
        self._lea_precision += precision
        self._lea_precision_weighting += p_weight

        doc_precision = precision / (p_weight + EPSILON)
        doc_recall = recall / (r_weight + EPSILON)
        doc_f1 = (doc_precision * doc_recall) \
            / (doc_precision + doc_recall + EPSILON) * 2
        return doc_f1, doc_precision, doc_recall

    @property
    def bakeoff(self):
        """ Get the F1 macroaverage score used by the bakeoff """
        return sum(self.mbc)/3

    @property
    def mbc(self):
        """ Get the F1 average score of (muc, b3, ceafe) over docs """
        avg_precisions = [self._muc_precision, self._b3_precision, self._ceafe_precision]
        avg_precisions = [i/(self._num_preds + EPSILON) for i in avg_precisions]

        avg_recalls = [self._muc_recall, self._b3_recall, self._ceafe_recall]
        avg_recalls = [i/(self._num_preds + EPSILON) for i in avg_recalls]

        avg_f1s = [self._f1(p,r) for p,r in zip(avg_precisions, avg_recalls)]

        return avg_f1s

    @property
    def total_lea(self):
        """ Returns weighted LEA for all the documents as
        (f1, precision, recall) """
        precision = self._lea_precision / (self._lea_precision_weighting + EPSILON)
        recall = self._lea_recall / (self._lea_recall_weighting + EPSILON)
        f1 = self._f1(precision, recall)
        return f1, precision, recall

    @staticmethod
    def _lea(key: List[List[Hashable]],
             response: List[List[Hashable]]) -> Tuple[float, float]:
        """ See aclweb.org/anthology/P16-1060.pdf. """
        response_clusters = [set(cluster) for cluster in response]
        response_map = {mention: cluster
                        for cluster in response_clusters
                        for mention in cluster}
        importances = []
        resolutions = []
        for entity in key:
            size = len(entity)
            if size == 1:  # entities of size 1 are not annotated
                continue
            importances.append(size)
            correct_links = 0
            for i in range(size):
                for j in range(i + 1, size):
                    correct_links += int(entity[i]
                                         in response_map.get(entity[j], {}))
            resolutions.append(correct_links / (size * (size - 1) / 2))
        res = sum(imp * res for imp, res in zip(importances, resolutions))
        weight = sum(importances)
        return res, weight

    @staticmethod
    def _muc(key: List[List[Hashable]],
             response: List[List[Hashable]]) -> float:
        """ See aclweb.org/anthology/P16-1060.pdf. """

        response_clusters = [set(cluster) for cluster in response]
        response_map = {mention: cluster
                        for cluster in response_clusters
                        for mention in cluster}

        top = 0 # sum over k of |k_i| - response_partitions(|k_i|)
        bottom = 0 # sum over k of |k_i| - 1

        for entity in key:
            S = len(entity)
            # we need to figure the number of DIFFERENT clusters 
            # the response assigns to members of the entity; ideally
            # this number is 1 (i.e. they are all assigned the same
            # coref).
            response_clusters = [response_map.get(i, None) for i in entity]
            # and dedplicate
            deduped = []
            for i in response_clusters:
                if i == None:
                    deduped.append(i)
                elif i not in deduped:
                    deduped.append(i)
            # the "partitions" will then be size of the deduped list
            p_k = len(deduped)
            top += (S - p_k)
            bottom += (S - 1)
        
        try:
            return top/bottom
        except ZeroDivisionError:
            logger.warning("muc got a zero division error because the model predicted no spans!")
            return 0 # +inf technically

    @staticmethod
    def _b3(key: List[List[Hashable]],
            response: List[List[Hashable]]) -> float:
        """ See aclweb.org/anthology/P16-1060.pdf. """
        
        response_clusters = [set(cluster) for cluster in response]

        top = 0 # sum over key and response of (|k intersect response|^2/|k|)
        bottom = 0 # sum over k of |k_i|

        for entity in key:
            bottom += len(entity)
            entity = set(entity)

            for res_entity in response_clusters:
                top += (len(entity.intersection(res_entity))**2)/len(entity)

        try:
            return top/bottom
        except ZeroDivisionError:
            logger.warning("b3 got a zero division error because the model predicted no spans!")
            return 0 # +inf technically



    @staticmethod
    def _phi4(c1, c2):
        return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))

    @staticmethod
    def _ceafe(clusters: List[List[Hashable]], gold_clusters: List[List[Hashable]]):
        """ see https://github.com/ufal/corefud-scorer/blob/main/coval/eval/evaluator.py """

        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            raise ImportError("To perform CEAF scoring, please install scipy via `pip install scipy` for the Kuhn-Munkres linear assignment scheme.")

        clusters = [c for c in clusters]
        scores = np.zeros((len(gold_clusters), len(clusters)))
        for i in range(len(gold_clusters)):
            for j in range(len(clusters)):
                scores[i, j] = ClusterChecker._phi4(gold_clusters[i], clusters[j])
        row_ind, col_ind = linear_sum_assignment(-scores)
        similarity = scores[row_ind, col_ind].sum()

        # precision, recall
        try:
            prec = similarity/len(clusters)
        except ZeroDivisionError:
            logger.warning("ceafe got a zero division error because the model predicted no spans!")
            prec = 0
        recc = similarity/len(gold_clusters)
        return prec, recc

