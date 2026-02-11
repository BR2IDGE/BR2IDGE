from search_recs.metric import BaseMetric
import pandas as pd

class AveragePrecision(BaseMetric):
    def __init__(self):
        super().__init__()
        
    def evaluate_metric(self, y_pred: list, y_true: list, topk: int = 15) -> float:

        df = pd.DataFrame({
            "relevance_true": y_true,
            "relevance_pred": y_pred
        })

        predicted_rank = df.sort_values(by="relevance_pred", ascending=False).reset_index(drop=True)
        
        predicted_rank = predicted_rank.head(topk)

        relevant_count = 0  
        precision_sum = 0.0 
        
        for k, row in predicted_rank.iterrows():
            is_relevant = row["relevance_true"] > 0
            
            if is_relevant:
                relevant_count += 1
    
                # Calculate Precision@k:
                P_k = relevant_count / (k + 1) 
                
                precision_sum += P_k
        
        if relevant_count == 0:
            return 0.0
            
        avg_precision = precision_sum / relevant_count
        
        return avg_precision
