from search_recs.metric import BaseMetric
import numpy as np
import math

class NdcgAtK(BaseMetric):
    @staticmethod
    def dcg_score(ranking: np.array, k: int):

        return np.sum([rel/(math.log2(i + 1)) for i, rel in enumerate(ranking[:k], 1)])

    def evaluate_metric(self, y_pred: list, y_true: list, topk: int = 3) -> float:

        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)

        if np.sum(y_true) == .0:
            return 0.0

        max_dcg = self.dcg_score(y_true[np.argsort(y_true)[::-1]], topk)
        actual_dcg = self.dcg_score(y_true[np.argsort(y_pred)[::-1]], topk)
        try:
            ndcg = (actual_dcg) / (max_dcg)
        except ZeroDivisionError:
            ndcg = 0

        return ndcg

