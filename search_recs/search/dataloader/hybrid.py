"""Search adapter for the shared hybrid Amazon dataset."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .base_dataloader import BuildConfig


def _resolve_dataset_path(value: str) -> Path:
    path = Path(value).expanduser()
    candidates = [path, Path.cwd() / path, Path.cwd() / "data" / path]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Hybrid dataset directory not found: {value}. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _document_text(items: pd.DataFrame) -> pd.Series:
    fields = ["title", "description", "category", "brand"]
    clean = items[fields].fillna("").astype(str)
    return clean.apply(lambda row: "\n".join(value.strip() for value in row if value.strip()), axis=1)


def load_hybrid_dataset(
    cfg: BuildConfig,
    dataset_path: str = "data/hybrid_dataset/hybrid_dataset",
    relevant_only: bool = True,
    path: str = None,
    **_unused,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return query-document pairs split by query (never by individual qrel).

    Besides the framework's required columns, the frames retain ``query_id``,
    ``document_id``, ``grade`` and ``relevant`` for graded evaluation.
    """
    root = _resolve_dataset_path(path or dataset_path)
    items = pd.read_parquet(root / "items.parquet")
    queries = pd.read_parquet(root / "queries.parquet")
    qrels = pd.read_parquet(root / "qrels.parquet")

    items = items.copy()
    items["document"] = _document_text(items)
    pairs = (
        qrels.merge(queries, on="query_id", how="inner", validate="many_to_one")
        .merge(items[["asin", "document", "category"]], on="asin", how="inner", validate="many_to_one")
        .rename(columns={"query_text": "search_query", "asin": "document_id"})
    )
    if relevant_only:
        pairs = pairs[pairs["relevant"] == 1]
    pairs["category"] = pairs["category"].fillna("unknown").astype(str)
    pairs = pairs[
        ["search_query", "document", "document_id", "category", "query_id", "grade", "relevant"]
    ].drop_duplicates()

    query_ids = pairs["query_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(cfg.random_state)
    rng.shuffle(query_ids)
    n_test = max(1, int(round(len(query_ids) * cfg.test_size)))
    n_val = max(1, int(round(len(query_ids) * cfg.val_size)))
    if n_test + n_val >= len(query_ids):
        raise ValueError("Not enough queries for the configured validation/test split.")
    test_ids = set(query_ids[:n_test])
    val_ids = set(query_ids[n_test:n_test + n_val])
    train_ids = set(query_ids[n_test + n_val:])

    def select(ids, head):
        frame = pairs[pairs["query_id"].isin(ids)].reset_index(drop=True)
        return frame.head(head).copy() if head else frame

    return (
        select(train_ids, cfg.head_train),
        select(val_ids, cfg.head_val),
        select(test_ids, cfg.head_test),
    )
