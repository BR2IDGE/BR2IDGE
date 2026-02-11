from abc import ABC, abstractmethod
import json
import pandas as pd
from pathlib import Path
from libreco.data import split_by_ratio_chrono
from libreco.data import split_by_num_chrono

class RecsDataLoader(ABC):
    def __init__(self, dataloader_config: dict):
        """
        Initializes the base dataset for recommendation.

        Args:
            dataloader_config (dict): A dictionary of parameters specific to this dataset.
        """
        print(dataloader_config)
        self.dataset_name = dataloader_config.get("dataset_name")
        
        # Path resolution logic
        root_path = Path.cwd().parent.parent
        dataset_folder = dataloader_config.get("path")
        
        if dataset_folder is None:
            raise ValueError("The dataset folder must be defined in the configuration.")
        
        print(f"Dataset folder: {dataset_folder}")
        self.dataset_path = root_path / "data" / dataset_folder
        
        print(f"Full path: {self.dataset_path}")
        print(f"Initializing {self.__class__.__name__} with dataset: {self.dataset_name}")

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Abstract method to read the dataset.

        Returns:
            pd.DataFrame: A merged and preprocessed DataFrame with a 'label' column.
        """
        pass

    @staticmethod
    def temporal_split_ratio(
        dataset: pd.DataFrame, 
        test_ratio: float = 0.1,
    ) -> tuple:
        """
        Performs a temporal split using libreco's split_by_ratio_chrono.
        Sorts interactions by timestamp and assigns a proportion of the latest 
        interactions to the test set.

        Args:
            dataset (pd.DataFrame): The full interaction DataFrame.
            test_ratio (float): Proportion of data for the test set (e.g., 0.1 for 10%).

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: (train_data, test_data).
        """
        train_data, test_data = split_by_ratio_chrono(dataset, test_size=test_ratio)
        print(f"Temporal split complete. Train shape={train_data.shape}, Test shape={test_data.shape}")
        return train_data, test_data
    
    def temporal_split_num(
        self,
        dataset: pd.DataFrame, 
        test_num: int = 100
    ) -> tuple:
        """
        Performs a temporal split (Leave-N-Out) per user.

        Args:
            dataset (pd.DataFrame): The full interaction DataFrame.
            test_num (int): Number of latest items per user to put in the test set.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: (train_data, test_data).
        """
        train_data, test_data = split_by_num_chrono(dataset, test_size=test_num)
        
        print(f"Temporal split (leave-N-out) complete. N={test_num} per user.")
        print(f"Train size={train_data.shape}, Test size={test_data.shape}")
    
        return train_data, test_data