from abc import ABC, abstractmethod
from search_recs.recs.model import BaseRecsModel
from search_recs.metric import BaseMetric
import pandas as pd

class BaseEvaluator(ABC):
    def __init__(self, model: BaseRecsModel, metrics: dict, test_data: pd.DataFrame):
        self.model = model
        self.metrics = metrics
        self.test_data = test_data
        
    @abstractmethod    
    def evaluate(self, topk: int):
        """
        Abstract method that outputs the evaluate for the metrics
        """
        pass
    
    @abstractmethod  
    def evaluation_metric(self, metric: BaseMetric, y_true, y_pred, topk):
        """
        Abstract method to evaluate a metric.

        Args:
            metric (BaseMetric): evaluation metric
        """
        pass
