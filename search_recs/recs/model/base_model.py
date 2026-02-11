import json
import pandas as pd
from abc import ABC, abstractmethod

# --- Abstract Base Class ---
class BaseRecsModel(ABC):
    """
    Abstract Base Class for Recommendation Models.
    Defines the common interface for preprocessing, training, and prediction.
    """

    def __init__(self, model_config: dict):
        """
        Initializes the base recommendation model.

        Args:
            model_name (str): The name of the specific model (e.g., 'DeepFM', 'BPR').
            model_params (dict): A dictionary of parameters specific to this model.
        """
        self.model_name = model_config.get("model_name")
        self.model_params = model_config.get("parameters")
        self.model = None
        print(f"Initializing {self.__class__.__name__} with model: {self.model_name}")
        print(f"Parameters: {json.dumps(self.model_params, indent=2)}")

    @abstractmethod
    def preprocess(self, train_data: pd.DataFrame, **kwargs):
        """
        Abstract method for data preprocessing specific to the model.
        This could involve feature engineering, creating sparse/dense features,
        or converting data into a library-specific format.

        Args:
            train_data (pd.DataFrame): The raw input train data.
            **kwargs: Additional arguments for preprocessing.
        """
        pass

    @abstractmethod
    def fit(self):
        """
        Abstract method for training the model.
        """
        pass

    @abstractmethod
    def prediction(self, preprocessed_data):
        """
        Abstract method for generating predictions using the trained model.

        Args:
            preprocessed_data: Data that has been preprocessed for prediction.

        Returns:
            Model predictions.
        """
        pass