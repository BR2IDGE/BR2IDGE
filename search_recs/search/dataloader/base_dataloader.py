from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class BuildConfig:
    def __init__(
        self,
        test_size: float = 0.3,
        val_size: float = 0.1,
        random_state: int = 42,
        min_tag_count: Optional[int] = None,
        head_train: Optional[int] = None,
        head_val: Optional[int] = None,
        head_test: Optional[int] = None,
        stratify_on: Optional[str] = "category",
    ):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.min_tag_count = min_tag_count
        self.head_train = head_train
        self.head_val = head_val
        self.head_test = head_test
        self.stratify_on = stratify_on
        
        if not (0.0 < self.test_size < 1.0):
            raise ValueError(f"self.test_size out of range (0,1). Received: {self.test_size}")
        if not (0.0 < self.val_size < 1.0):
            raise ValueError(f"self.val_size out of range (0,1). Received: {self.val_size}")
        if self.test_size + self.val_size >= 1.0:
            raise ValueError(
                f"test_size + val_size out of range >= 1.0. Received: {self.test_size + self.val_size}"
            )


class BaseSearchDatasetBuilder(ABC):
    """
    Interface: Must return train_df and val_df with the following columns:
    - search_query (str)
    - document (str)
    - category (str)
    """

    def __init__(self, cfg: BuildConfig = BuildConfig()):
        self.cfg = cfg

    @abstractmethod
    def load_raw(self) -> None:
        ...

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df, val_df = train_test_split(
            df, test_size=self.cfg.test_size, random_state=self.cfg.random_state
        )
        if self.cfg.head_train:
            train_df = train_df.head(self.cfg.head_train)
        if self.cfg.head_val:
            val_df = val_df.head(self.cfg.head_val)
        return train_df, val_df

    def build(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.load_raw()
        pairs = self.build_pairs()
        return self.split(pairs)