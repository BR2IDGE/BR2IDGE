import numpy as np
from search_recs.metric import BaseMetric

class PrecisionAtK(BaseMetric):
    def __init__(self, rel_threshold: float = 0.0):

        super().__init__()
        self.rel_threshold = rel_threshold

    def evaluate_metric(self, y_pred: list, y_true: list, topk: int) -> float:

        if topk <= 0:
            return 0.0

        if y_pred is None or y_true is None:
            return 0.0
        
        if len(y_pred) == 0 or len(y_pred) != len(y_true):
            return 0.0

        y_pred = np.asarray(y_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        k = min(topk, len(y_pred))

        topk_idx = np.argsort(-y_pred)[:k]

        rel_mask = y_true > self.rel_threshold

        tp = rel_mask[topk_idx].sum()

        # Calculate Precision@k 
        return float(tp) / float(k) if k > 0 else 0.0