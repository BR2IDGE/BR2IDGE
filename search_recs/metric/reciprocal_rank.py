from search_recs.metric import BaseMetric
import pandas as pd

class ReciprocalRank(BaseMetric):
    def __init__(self):
        super().__init__()
        
    def evaluate_metric(self, y_pred: list, y_true: list, topk: int) -> float:

        df = pd.DataFrame({
            "relevance_true": y_true,
            "relevance_pred": y_pred
        })

        predicted_rank = df.sort_values(by="relevance_pred", ascending=False).reset_index(drop=True)

        first_relevant_item = predicted_rank[predicted_rank["relevance_true"] > 0].head(topk)

        if first_relevant_item.empty:
            return 0.0  
        
        first_relevant_rank = first_relevant_item.index[0] + 1
        
        # Reciprocal Rank calculation
        rr = 1.0 / first_relevant_rank

        return rr
