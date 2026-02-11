from abc import ABC, abstractmethod

class BaseMetric(ABC):
    
    @abstractmethod
    def evaluate_metric(self, y_pred: list, y_true: list, topk: int) -> float:
        """
        Abstract method to evaluate the output of a model.

        Args:
            y_pred (list): The predicted relevance scores
            y_true (list): The ground truth relevance labels
            topk (int): The number of results to consider
        Returns:
            evaluation score
        """
        pass
