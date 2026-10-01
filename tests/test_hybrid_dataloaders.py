import json

import pandas as pd

from search_recs.recs.dataloader.hybrid import HybridDatasetDataLoader
from search_recs.search.dataloader import BuildConfig
from search_recs.search.dataloader.hybrid import load_hybrid_dataset


def _write_dataset(root):
    items = pd.DataFrame({
        "asin": ["a", "b", "c", "d", "e", "f"],
        "title": ["A", "B", "C", "D", "E", "F"],
        "description": ["desc"] * 6,
        "category": ["cat"] * 6,
        "brand": ["brand"] * 6,
    })
    interactions = pd.DataFrame({
        "user_id": ["u1", "u1", "u2"], "asin": ["a", "b", "c"],
        "rating": [5.0, 4.0, 3.0], "timestamp": [1.0, 3.0, 2.0],
    })
    queries = pd.DataFrame({"query_id": range(10), "query_text": [f"q{i}" for i in range(10)]})
    qrels = pd.DataFrame({
        "query_id": range(10), "asin": ["a", "b", "c", "d", "e", "f", "a", "b", "c", "d"],
        "esci_label": ["E"] * 10, "grade": [3] * 10, "relevant": [1] * 10,
    })
    for name, frame in (("items", items), ("interactions", interactions), ("queries", queries), ("qrels", qrels)):
        frame.to_parquet(root / f"{name}.parquet", index=False)
    (root / "splits.json").write_text(json.dumps({"fold_0": {"cutoff_timestamp": 2.0}}))


def test_recommendation_loader_schema_and_fold(tmp_path):
    _write_dataset(tmp_path)
    loader = HybridDatasetDataLoader({"path": str(tmp_path), "fold": "fold_0"})
    data = loader.load_data()
    assert {"user", "item", "label", "time", "brand", "category"} <= set(data.columns)
    train, test = loader.temporal_split(data)
    assert train["time"].max() <= 2.0 < test["time"].min()


def test_search_loader_splits_whole_queries(tmp_path):
    _write_dataset(tmp_path)
    cfg = BuildConfig(test_size=.2, val_size=.2, random_state=7, stratify_on=None)
    train, val, test = load_hybrid_dataset(cfg, str(tmp_path))
    assert set(train.query_id).isdisjoint(val.query_id)
    assert set(train.query_id).isdisjoint(test.query_id)
    assert set(val.query_id).isdisjoint(test.query_id)
    assert {"search_query", "document", "document_id", "category", "grade"} <= set(train.columns)
