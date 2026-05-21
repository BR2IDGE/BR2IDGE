import json
import random
import time
from typing import Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split

from search_recs.datasets import ensure_dataset
from search_recs.recs.dataloader.lastfm import (
    _load_360k_artist_lookup,
    _load_360k_user_lookup,
    _scan_360k_plays,
)
from .base_dataloader import BaseSearchDatasetBuilder, BuildConfig


class LastFmHybridBuilder(BaseSearchDatasetBuilder):
    def __init__(self, data_dir: str, cfg: BuildConfig = BuildConfig()):
        super().__init__(cfg)
        self.data_dir = ensure_dataset("lastfm-hybrid")
        self.lastfm360k_dir = ensure_dataset("lastfm-dataset-360K")

        self.tags_path = self.data_dir / "tags.dat"
        self.tagging_path = self.data_dir / "user_taggedartists.dat"
        self.artists_2k_path = self.data_dir / "artists.dat"

        self.tags_df = None
        self.tagging_df = None
        self.artists_2k_df = None

        random.seed(cfg.random_state)

    def load_raw(self) -> None:
        print("[DataLoader] Loading Last.fm local datasets (360K + HetRec tags)...")
        self.tags_df = pd.read_csv(self.tags_path, sep="\t", encoding="latin-1", on_bad_lines="skip")
        self.tagging_df = pd.read_csv(self.tagging_path, sep="\t", encoding="latin-1", on_bad_lines="skip")
        self.artists_2k_df = pd.read_csv(self.artists_2k_path, sep="\t", encoding="latin-1", on_bad_lines="skip")

    def build(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.load_raw()
        return pd.DataFrame(), pd.DataFrame()


def _generate_random_seed():
    now = time.time() * 1000
    random_factor = random.random()
    return int((random_factor * now) % 10000)


def _load_360k_interactions_pandas(dataset_path) -> pd.DataFrame:
    users = _load_360k_user_lookup(dataset_path)
    artists = _load_360k_artist_lookup(dataset_path)
    return (
        _scan_360k_plays(dataset_path)
        .join(users.lazy(), on="user_name", how="inner")
        .join(artists.lazy().select(["item", "artist_key", "artist_name"]), on="artist_key", how="inner")
        .select(["user", "item", "label", "artist_name"])
        .collect()
        .to_pandas()
    )


def load_lastfm_hybrid_dataset(
    cfg: BuildConfig,
    dataset_path: str = "./data/lastfm-hybrid",
    train_ratio: float = 0.6,
    n_neg: int = 200,
    max_test_items: int = 40,
    min_hist_len: int = 3,
    max_hist_len: int = 50,
    max_pairs_per_user: int = 50,
):
    builder = LastFmHybridBuilder(data_dir=dataset_path, cfg=cfg)
    builder.load_raw()

    artist_lookup = _load_360k_artist_lookup(builder.lastfm360k_dir)
    corpus_lookup = (
        artist_lookup.select([pl.col("item").alias("document_id"), pl.col("artist_name").alias("document")])
        .to_pandas()
    )

    interactions = _load_360k_interactions_pandas(builder.lastfm360k_dir)
    all_items = corpus_lookup["document_id"].to_numpy(dtype=int)
    rs = int(time.time() * 1000) % (2**31 - 1)
    rng = np.random.default_rng(rs)

    train_rows, test_rows = [], []
    for uid, group in interactions.groupby("user", sort=False):
        user_items = group["item"].to_numpy(dtype=int)
        user_vals = group["label"].to_numpy(dtype=float)

        if user_items.size < (min_hist_len + 2):
            continue
        if user_items.size > max_test_items:
            pick = rng.choice(np.arange(user_items.size), size=max_test_items, replace=False)
            user_items, user_vals = user_items[pick], user_vals[pick]

        perm = rng.permutation(user_items.size)
        cut = max(1, int(user_items.size * train_ratio))
        train_items, test_items = user_items[perm[:cut]].tolist(), user_items[perm[cut:]].tolist()

        if len(train_items) < (min_hist_len + 1) or len(test_items) < 2:
            continue

        val_map = {int(i): float(v) for i, v in zip(user_items.tolist(), user_vals.tolist())}
        seq = sorted(train_items, key=lambda it: (-val_map.get(int(it), 0.0), int(it)))
        target_indices = list(range(min_hist_len, len(seq)))
        if len(target_indices) > max_pairs_per_user:
            target_indices = sorted(rng.choice(target_indices, size=max_pairs_per_user, replace=False).tolist())

        for t in target_indices:
            hist = seq[max(0, t - max_hist_len):t]
            train_rows.append(
                {"search_query": ",".join(map(str, hist)), "document_id": int(seq[t]), "category": "UserHistoryTrain"}
            )

        half = len(test_items) // 2
        query_items, gt_items = test_items[:half], test_items[half:]
        seen = set(train_items) | set(query_items) | set(gt_items)
        possible = np.setdiff1d(all_items, np.fromiter(seen, dtype=int))
        negs = rng.choice(possible, size=min(n_neg, len(possible)), replace=False).tolist()

        test_rows.append(
            {
                "userId": int(uid),
                "search_query": ",".join(map(str, query_items)),
                "ground_truth_ids": json.dumps(list(map(int, gt_items))),
                "candidate_ids": json.dumps(list(map(int, gt_items)) + negs),
                "category": "UserHistoryTestUserwise",
                "document_id": -1,
                "document": "",
            }
        )

    train_pairs = pd.DataFrame(train_rows)
    train_df = train_pairs.merge(corpus_lookup, on="document_id", how="inner") if not train_pairs.empty else train_pairs
    test_df = pd.DataFrame(test_rows)

    print("[lastfm-hybrid] Built local user-history search dataset from LastFM 360K.")
    return train_df.reset_index(drop=True), pd.DataFrame(columns=train_df.columns), test_df.reset_index(drop=True)


def load_lastfm_search_dataset(
    cfg: BuildConfig,
    dataset_path: str = "./data/lastfm-hybrid",
    min_tag_freq: int = 50,
    top_k_tags: int = 20,
):
    random.seed(cfg.random_state)

    builder = LastFmHybridBuilder(data_dir=dataset_path, cfg=cfg)
    builder.load_raw()

    artist_lookup = _load_360k_artist_lookup(builder.lastfm360k_dir)
    plays_sum = (
        _scan_360k_plays(builder.lastfm360k_dir)
        .group_by("artist_key")
        .agg(pl.col("label").sum().alias("global_plays"))
        .collect()
    )
    df_360 = (
        artist_lookup.join(plays_sum, on="artist_key", how="left")
        .select(
            [
                pl.col("artist_name_clean"),
                pl.col("artist_name").alias("original_name"),
                pl.col("global_plays").fill_null(0),
            ]
        )
        .filter(pl.col("global_plays") > 0)
        .to_pandas()
    )

    builder.artists_2k_df["name_clean"] = builder.artists_2k_df["name"].astype(str).str.lower().str.strip()

    df_tags_full = builder.tagging_df.merge(builder.tags_df, on="tagID")
    df_tags_full = df_tags_full.merge(
        builder.artists_2k_df[["id", "name_clean"]],
        left_on="artistID",
        right_on="id",
    )

    df_tags_full["tagValue"] = df_tags_full["tagValue"].astype(str).str.lower().str.strip()
    tag_counts = df_tags_full["tagValue"].value_counts()
    valid_tags = tag_counts[tag_counts >= min_tag_freq].index

    df_tags_filtered = df_tags_full[df_tags_full["tagValue"].isin(valid_tags)].copy()
    df_tags_grouped = (
        df_tags_filtered.groupby(["name_clean", "tagValue"]).size().reset_index(name="tag_weight")
    )
    df_tags_grouped = df_tags_grouped.sort_values(["name_clean", "tag_weight"], ascending=[True, False])
    df_best_tags = df_tags_grouped.groupby("name_clean").head(top_k_tags)

    df_final = df_360.merge(df_best_tags, left_on="artist_name_clean", right_on="name_clean", how="inner")

    search_df = pd.DataFrame(
        {
            "search_query": df_final["tagValue"],
            "document": df_final["original_name"],
            "document_id": df_final["artist_name_clean"],
            "category": "music_artist",
        }
    )

    search_df["search_query"] = search_df["search_query"].astype(str)
    search_df["document"] = search_df["document"].astype(str)

    mask = (
        (search_df["search_query"].str.strip() != "")
        & (search_df["search_query"].str.lower() != "nan")
        & (search_df["document"].str.strip() != "")
    )
    search_df = search_df[mask].drop_duplicates()

    print(f"[LastFM-Search] Total pairs: {len(search_df)}")
    print(f"[LastFM-Search] Total UNIQUE QUERIES (Tags): {search_df['search_query'].nunique()}")

    test_size = float(cfg.test_size)
    val_size = float(cfg.val_size)
    if (test_size + val_size) >= 1.0:
        test_size, val_size = 0.1, 0.1

    total_eval = test_size + val_size
    train_df, eval_df = train_test_split(
        search_df,
        test_size=total_eval,
        random_state=_generate_random_seed(),
        shuffle=True,
    )

    relative_test = test_size / total_eval if total_eval > 0 else 0.5
    val_df, test_df = train_test_split(
        eval_df,
        test_size=relative_test,
        random_state=_generate_random_seed(),
        shuffle=True,
    )

    if getattr(cfg, "head_train", None):
        train_df = train_df.head(cfg.head_train)
    if getattr(cfg, "head_val", None):
        val_df = val_df.head(cfg.head_val)
    if getattr(cfg, "head_test", None):
        test_df = test_df.head(cfg.head_test)

    print("[DataLoader] Last.fm Search loaded from local 360K + HetRec tags.")
    print(f"   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    return train_df, val_df, test_df


def load_lastfm_dataset(cfg: BuildConfig, **kwargs):
    mode = kwargs.get("mode", "hybrid")
    dataset_path = kwargs.get("dataset_path", kwargs.get("path", "./data/lastfm-hybrid"))

    if mode == "hybrid":
        print("[DataLoader] Starting Last.fm HYBRID mode (User History)")
        return load_lastfm_hybrid_dataset(
            cfg=cfg,
            dataset_path=dataset_path,
            train_ratio=kwargs.get("train_ratio", 0.6),
            n_neg=kwargs.get("n_neg", 200),
            max_test_items=kwargs.get("max_test_items", 40),
            min_hist_len=kwargs.get("min_hist_len", 3),
            max_hist_len=kwargs.get("max_hist_len", 50),
            max_pairs_per_user=kwargs.get("max_pairs_per_user", 50),
        )

    print("[DataLoader] Starting Last.fm SEARCH mode (Tags -> Artists)")
    return load_lastfm_search_dataset(
        cfg=cfg,
        dataset_path=dataset_path,
        min_tag_freq=kwargs.get("min_tag_freq", 50),
        top_k_tags=kwargs.get("top_k_tags", 20),
    )


load_lastfm_user_query_dataset = load_lastfm_dataset
