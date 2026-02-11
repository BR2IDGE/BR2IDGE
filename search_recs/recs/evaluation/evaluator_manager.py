import pandas as pd
import numpy as np
from search_recs.recs.evaluation import BaseEvaluator
from search_recs.metric import BaseMetric

class EvaluatorRecsManager(BaseEvaluator):
    
    def __init__(self, model, metrics, test_data, n_neg_samples=100):
        """
        Initializes the evaluator.
        
        Args:
            model: Trained model wrapper.
            metrics: Dict of {name: metric_class}.
            test_data: Test DataFrame (positives only).
            n_neg_samples (int): Number of negative samples to generate per user.
        """
        super().__init__(model, metrics, test_data)
        
        self.n_neg_samples = n_neg_samples
        if self.n_neg_samples <= 0:
            print(f"[Evaluation] WARNING: Invalid n_neg_samples ({self.n_neg_samples}). Defaulting to 100.")
            self.n_neg_samples = 100
    
    def evaluate(self, topks: list) -> tuple:
        """
        Runs evaluation with negative sampling to ensure accurate metrics.
        """
        
        # --- 1. Prepare Negative Sampling ---
        n_neg_samples = self.n_neg_samples
        
        try:
            # Get model metadata
            user_col = self.model.user_col
            item_col = self.model.item_col
            label_col = self.model.label_col
            all_items_external = np.array(list(self.model.item2idx.keys()))
        except AttributeError as e:
            print("!! EVALUATION ERROR !!")
            print(f"Model wrapper ({type(self.model)}) missing attributes (.user_col, .item_col, etc): {e}")
            return {}

        pos_df = self.test_data.copy()
        test_users_external = pos_df[user_col].unique()
        negative_rows = []

        # Generate N negative samples for each user
        for user_ext in test_users_external:
            # Random sampling from item universe
            neg_item_samples = np.random.choice(all_items_external, n_neg_samples, replace=False)
            
            for item_ext in neg_item_samples:
                negative_rows.append({
                    user_col: user_ext,
                    item_col: item_ext,
                    label_col: 0.0  # Negative label
                })

        neg_df = pd.DataFrame(negative_rows, columns=[user_col, item_col, label_col])
        
        # Combine positive and negative data
        eval_df = pd.concat([pos_df, neg_df], ignore_index=True)
        
        # --- 2. Prediction ---
        y_true, y_pred = self.model.prediction(eval_df)
        
        if y_true is None:
             print("!! EVALUATION ERROR: 'label' column not found during prediction.")
             return {}
             
        y_true_list = list(y_true)
        y_pred_list = list(y_pred)

        # --- 3. Calculate Metrics ---
        results = {}
        for topk in topks:
            results[topk] = {}
            for metric_name, metric_cls in self.metrics.items():
                results[topk][metric_name] = self.evaluation_metric(
                    metric_cls,
                    y_true_list,
                    y_pred_list, 
                    topk,
                )
        return results
        
    def evaluation_metric(self, metric_cls: BaseMetric, y_true, y_pred, topk) -> float:
        """Instantiates and calculates a specific metric."""
        metric_instance = metric_cls()
        return metric_instance.evaluate_metric(
            y_pred=y_pred,
            y_true=y_true,
            topk=topk
        )