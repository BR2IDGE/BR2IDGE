"""Recommendation adapter for the shared hybrid Amazon dataset."""

import json
from pathlib import Path
from typing import Tuple

import pandas as pd

from .recs_dataloader import RecsDataLoader


def _dataset_path(value: str) -> Path:
    path = Path(value).expanduser()
    candidates = [path, Path.cwd() / path, Path.cwd() / "data" / path]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Hybrid dataset directory not found: {value}. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


class HybridDatasetDataLoader(RecsDataLoader):
    """Load ``interactions.parquet`` and join product side information.

    The returned schema follows the recommendation models' convention:
    ``user``, ``item``, ``label`` and ``time``. Product metadata is retained so
    feature-aware models can consume ``brand`` and ``category``.
    """

    def __init__(self, dataloader_config: dict):
        dataloader_config = dataloader_config.get("dataloader", dataloader_config)
        self.dataset_name = dataloader_config.get("dataset_name", "HybridDataset")
        value = dataloader_config.get("path", "data/hybrid_dataset/hybrid_dataset")
        self.dataset_path = _dataset_path(value)
        self.fold = dataloader_config.get("fold", "fold_2")
        self.positive_threshold = dataloader_config.get("positive_threshold")

        required = ("interactions.parquet", "items.parquet", "splits.json")
        missing = [name for name in required if not (self.dataset_path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Invalid hybrid dataset at {self.dataset_path}; missing: {missing}"
            )

    def load_data(self) -> pd.DataFrame:
        interactions = pd.read_parquet(self.dataset_path / "interactions.parquet")
        items = pd.read_parquet(self.dataset_path / "items.parquet")
        item_features = items[["asin", "brand", "category"]].copy()
        for column in ("brand", "category"):
            item_features[column] = item_features[column].fillna("unknown").astype(str)

        data = interactions.merge(item_features, on="asin", how="left", validate="many_to_one")
        data = data.rename(
            columns={"user_id": "user", "asin": "item", "rating": "label", "timestamp": "time"}
        )
        if self.positive_threshold is not None:
            data["label"] = (data["label"] >= float(self.positive_threshold)).astype("float32")
        return data.sort_values("time", kind="stable").reset_index(drop=True)

    def temporal_split(self, dataset: pd.DataFrame = None, fold: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply one of the global temporal cuts recorded in ``splits.json``."""
        data = self.load_data() if dataset is None else dataset
        fold_name = fold or self.fold
        with (self.dataset_path / "splits.json").open(encoding="utf-8") as stream:
            folds = json.load(stream)
        if fold_name not in folds:
            raise ValueError(f"Unknown fold '{fold_name}'. Available folds: {list(folds)}")
        cutoff = float(folds[fold_name]["cutoff_timestamp"])
        train = data[data["time"] <= cutoff].copy().reset_index(drop=True)
        test = data[data["time"] > cutoff].copy().reset_index(drop=True)
        return train, test
