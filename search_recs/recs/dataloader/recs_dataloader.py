from abc import ABC, abstractmethod
import json
import pandas as pd
from pathlib import Path
from libreco.data import split_by_ratio_chrono
from libreco.data import split_by_num_chrono

class RecsDataLoader(ABC):
    def __init__(self, dataloader_config: dict):

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
        train_data, test_data = split_by_ratio_chrono(dataset, test_size=test_ratio)
        print(f"Temporal split complete. Train shape={train_data.shape}, Test shape={test_data.shape}")
        return train_data, test_data
    
    def temporal_split_num(
        self,
        dataset: pd.DataFrame, 
        test_num: int = 100
    ) -> tuple:

        train_data, test_data = split_by_num_chrono(dataset, test_size=test_num)
        
        print(f"Temporal split (leave-N-out) complete. N={test_num} per user.")
        print(f"Train size={train_data.shape}, Test size={test_data.shape}")
    
        return train_data, test_data