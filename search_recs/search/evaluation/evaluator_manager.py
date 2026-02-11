from search_recs.metric import BaseMetric
from search_recs.search.evaluation import BaseEvaluator


class EvaluatorRecsManager(BaseEvaluator):
    def __init__(self, model, metrics, test_data):
        super().__init__(model, metrics, test_data)
        
    def evaluate(self, topks: list) -> tuple:
        """
        Abstract method that outputs the evaluate for the metrics
        """
        y_true, y_pred = self.model.prediction(self.test_data)
        results = {}
        for topk in topks:
            results[topk] = {}
            for metric_name, metric_cls in self.metrics.items():
                results[topk][metric_name] = self.evaluation_metric(
                    metric_cls,
                    y_true,
                    y_pred, 
                    topk,
                )
        return results
            
    
    def evaluation_metric(self, metric_cls: BaseMetric, y_true, y_pred, topk) -> float:
        """
        Evaluates a single metric.

        Args:
            metric_cls (BaseMetric): _metric class to be evaluated.
            y_true (_type_): True labels.
            y_pred (_type_): Predicted labels.
            topk (_type_): Top K value for evaluation.

        Returns:
            Score of the evaluation metric.
        """
        metric_instance = metric_cls()
        return metric_instance.evaluate_metric(
            y_pred=y_pred,
            y_true=y_true,
            topk=topk
        )
    
    # def eval_by_user(self, user: str, metric: BaseMetric, topk: int) -> float:
    #     """
    #     Abstract method that computes the evaluation metric for a user.

    #     Args:
    #         user (str): user identifier
    #         metric (BaseMetric): evaluation metrics
            
    #     Returns:
    #         evaluation score.
    #     """
    #     y_pred = self.model.predict(
    #         user=user
    #     )
    #     y_true = self.test_data[self.test_data.user == user]["label"].to_list()
    #     return self.metric.eval(
    #         y_pred=y_pred,
    #         y_true=y_true,
    #         topk=topk
    #     )

