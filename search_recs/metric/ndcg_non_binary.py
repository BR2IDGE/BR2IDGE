from search_recs.metric import BaseMetric
import numpy as np

class NdcgNonBinAtK(BaseMetric):    
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def dcg_score(ranking: np.array, k: int):

        ranks = np.arange(1, k + 1) 
        
        discount = 1.0 / np.log2(ranks + 1.0)
        
        gain = (2**ranking[:k] - 1)
        
        return np.sum(gain * discount)

    def evaluate_metric(self, y_pred: list, y_true: list, topk: int = 3) -> float:

        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)

        ideal_ranking = y_true[np.argsort(y_true)[::-1]]
        max_dcg = self.dcg_score(ideal_ranking, topk)

        if max_dcg == 0.0:
            return 0.0

        predicted_ranking_relevance = y_true[np.argsort(y_pred)[::-1]]
        actual_dcg = self.dcg_score(predicted_ranking_relevance, topk)

        # Calculate NDCG
        ndcg = actual_dcg / max_dcg

        return ndcg
