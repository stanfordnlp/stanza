""" Describes ClusterChecker, a class used to retrieve LEA scores.
See aclweb.org/anthology/P16-1060.pdf. """

from typing import Hashable, List, Tuple

from coref.const import EPSILON


class ClusterChecker:
    """ Collects information on gold and predicted clusters across documents.
    Can be used to retrieve weighted LEA-score for them.
    """
    def __init__(self):
        self._p = 0.0
        self._r = 0.0
        self._p_weight = 0.0
        self._r_weight = 0.0

    def add_predictions(self,
                        gold_clusters: List[List[Hashable]],
                        pred_clusters: List[List[Hashable]]):
        """
        Calculates LEA for the document's clusters and stores them to later
        output weighted LEA across documents.

        Returns:
            LEA score for the document as a tuple of (f1, precision, recall)
        """
        recall, r_weight = ClusterChecker._lea(gold_clusters, pred_clusters)
        precision, p_weight = ClusterChecker._lea(pred_clusters, gold_clusters)

        self._r += recall
        self._r_weight += r_weight
        self._p += precision
        self._p_weight += p_weight

        doc_precision = precision / (p_weight + EPSILON)
        doc_recall = recall / (r_weight + EPSILON)
        doc_f1 = (doc_precision * doc_recall) \
            / (doc_precision + doc_recall + EPSILON) * 2
        return doc_f1, doc_precision, doc_recall

    @property
    def total_lea(self):
        """ Returns weighted LEA for all the documents as
        (f1, precision, recall) """
        precision = self._p / (self._p_weight + EPSILON)
        recall = self._r / (self._r_weight + EPSILON)
        f1 = (precision * recall) / (precision + recall + EPSILON) * 2
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
