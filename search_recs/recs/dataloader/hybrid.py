"""Recommendation adapter for the shared hybrid Amazon dataset."""

import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd

from .recs_dataloader import RecsDataLoader

QUERY_PREFIX = "query::"
CONTEXT_PREFIX = "user::"


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


def item_documents(items: pd.DataFrame) -> pd.DataFrame:
    """Product text used by the search side (same fields as search_recs.search.dataloader.hybrid)."""
    fields = ["title", "description", "category", "brand"]
    clean = items[fields].fillna("").astype(str)
    text = clean.apply(lambda row: "\n".join(value.strip() for value in row if value.strip()), axis=1)
    return pd.DataFrame({"document_id": items["asin"].astype(str), "document": text})


def load_relevant_qrels(root: Path, min_grade: Optional[float] = None) -> pd.DataFrame:
    """Relevant (query, product) judgements whose query and product exist in the dataset.

    ``min_grade=None`` uses the dataset's own ``relevant`` flag; a number keeps ``grade >= min_grade``.
    """
    for name in ("queries.parquet", "qrels.parquet", "items.parquet"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"Hybrid dataset at {root} has no {name}.")
    qrels = pd.read_parquet(root / "qrels.parquet")
    rel = qrels[qrels["grade"] >= float(min_grade)] if min_grade is not None else qrels[qrels["relevant"] == 1]
    queries = pd.read_parquet(root / "queries.parquet", columns=["query_id"])
    catalog = set(pd.read_parquet(root / "items.parquet", columns=["asin"])["asin"].astype(str))
    rel = rel.merge(queries, on="query_id", how="inner")
    rel = rel.assign(query_id=rel["query_id"].astype(str), asin=rel["asin"].astype(str))
    rel = rel[rel["asin"].isin(catalog)]
    return rel[["query_id", "asin", "grade"]].drop_duplicates(["query_id", "asin"]).reset_index(drop=True)


def hybrid_context_interactions(root: Path, item_set: Iterable[str], frac: float, seed: int,
                                min_rating: Optional[float] = None) -> pd.DataFrame:
    """Real users' interactions restricted to ``item_set``, as extra co-occurrence for training only.

    A fraction ``frac`` of the real users who touched the catalogue is sampled (seeded); all their
    interactions with catalogue items are returned with ``role='context'``, which framework.py always
    keeps in train and never evaluates. Restricting to ``item_set`` keeps the candidate pool unchanged.
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"context_users_frac must be in (0, 1], got {frac}.")
    items = set(map(str, item_set))
    inter = pd.read_parquet(root / "interactions.parquet", columns=["user_id", "asin", "rating", "timestamp"])
    inter = inter.assign(user_id=inter["user_id"].astype(str), asin=inter["asin"].astype(str))
    inter = inter[inter["asin"].isin(items)]
    if min_rating is not None:
        inter = inter[inter["rating"] >= float(min_rating)]
    users = inter["user_id"].drop_duplicates()
    keep = set(users.sample(n=int(round(frac * len(users))), random_state=seed)) if len(users) else set()
    inter = inter[inter["user_id"].isin(keep)].drop_duplicates(["user_id", "asin"])
    out = pd.DataFrame({
        "user": CONTEXT_PREFIX + inter["user_id"],
        "item": inter["asin"],
        "label": 1.0,
        "time": pd.to_numeric(inter["timestamp"], errors="coerce").fillna(0).to_numpy(),
        "score": 0.0,
        "role": "context",
    })
    print(f"[Hybrid][context] users {len(keep)}/{len(users)} (frac={frac}) | interactions={len(out)} | "
          f"catalogue items touched={out['item'].nunique()}/{len(items)}")
    return out.reset_index(drop=True)


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

        # Query-as-User (hybrid_load_data) settings
        min_grade = dataloader_config.get("min_grade")
        self.min_grade = None if min_grade is None else float(min_grade)
        self.min_user_interactions = int(dataloader_config.get("min_user_interactions", 2) or 0)
        self.query_limit = dataloader_config.get("query_limit")
        self.seed = int(dataloader_config.get("seed", 42))
        self.order_by = str(dataloader_config.get("order_by", "random")).strip().lower()
        if self.order_by not in {"score", "random"}:
            raise ValueError(f"[Hybrid-QueryAsUser] order_by must be 'score' or 'random', got '{self.order_by}'.")
        self.context_users_frac = float(dataloader_config.get("context_users_frac", 0.0) or 0.0)
        min_rating = dataloader_config.get("context_min_rating")
        self.context_min_rating = None if min_rating is None else float(min_rating)

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

    def hybrid_load_data(self) -> pd.DataFrame:
        """Query-as-User: each search query is a pseudo-user whose items are its relevant products.

        With ``context_users_frac > 0``, real users' interactions with the same products are appended
        as ``role='context'`` rows (training-only co-occurrence, never evaluated).
        """
        rel = load_relevant_qrels(self.dataset_path, self.min_grade)

        if self.min_user_interactions > 1:
            before = rel["query_id"].nunique()
            rel = rel[rel.groupby("query_id")["asin"].transform("size") >= self.min_user_interactions]
            print(f"[Hybrid-QueryAsUser] min_user_interactions>={self.min_user_interactions}: "
                  f"{before} -> {rel['query_id'].nunique()} query-users.")
        if rel.empty:
            raise ValueError("[Hybrid-QueryAsUser] No query has enough relevant judgements.")

        if self.query_limit:
            ids = rel["query_id"].drop_duplicates()
            keep = set(ids.sample(n=min(int(self.query_limit), len(ids)), random_state=self.seed))
            rel = rel[rel["query_id"].isin(keep)]
            print(f"[Hybrid-QueryAsUser] query_limit={self.query_limit}: kept {len(keep)} query-users.")

        out = pd.DataFrame({
            "user": QUERY_PREFIX + rel["query_id"],
            "item": rel["asin"],
            "score": pd.to_numeric(rel["grade"], errors="coerce").fillna(0.0).astype(float),
        })
        out = out.sort_values(["user", "score", "item"], kind="mergesort")
        if self.order_by == "random":
            out = out.sample(frac=1.0, random_state=self.seed).sort_values("user", kind="mergesort")
        out["time"] = out.groupby("user", sort=False).cumcount()
        out["label"] = 1.0
        out = out[["user", "item", "label", "time", "score"]].reset_index(drop=True)

        per_user = out.groupby("user").size()
        print(f"[Hybrid-QueryAsUser] interactions={len(out)} | query-users={out['user'].nunique()} | "
              f"items={out['item'].nunique()} | items/user: min={per_user.min()} mean={per_user.mean():.1f} "
              f"max={per_user.max()} | order_by={self.order_by}")

        if self.context_users_frac > 0:
            context = hybrid_context_interactions(self.dataset_path, out["item"].unique(), self.context_users_frac,
                                                  self.seed, self.context_min_rating)
            out = pd.concat([out.assign(role=None), context], ignore_index=True)
        return out

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
