"""Shared helpers for the BERT4Rec matched-protocol comparison and the tag
coverage audits (Exp1 follow-up). Reuses MovieLensDataLoader's existing
_split_test_fixed_gt / load_hybrid_data instead of duplicating that logic.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from search_recs.search.dataloader.base_dataloader import BuildConfig
from search_recs.search.dataloader.movielens import load_movielens_dataset
from search_recs.datasets import ensure_dataset


def load_exp1_frames(history_size: int, gt_size: int = 20, history_size_max: int = 20, seed: int = 42):
    """Rebuilds the exact Exp1 train/test frames for a given history_size.

    Returns (train_df, test_df); test_df gains query_items/gt_items/candidate_ids_list
    (parsed Python lists) alongside the original search_query/ground_truth_ids/candidate_ids
    string columns.
    """
    cfg = BuildConfig(random_state=seed)
    train_df, _val_df, test_df = load_movielens_dataset(
        cfg, mode="hybrid", seed=seed,
        history_size=history_size, gt_size=gt_size, history_size_max=history_size_max,
    )
    test_df = test_df.copy()
    test_df["query_items"] = test_df["search_query"].apply(
        lambda s: [int(x) for x in str(s).split(",") if x]
    )
    test_df["gt_items"] = test_df["ground_truth_ids"].apply(json.loads)
    test_df["candidate_ids_list"] = test_df["candidate_ids"].apply(json.loads)
    return train_df, test_df


def build_has_tag_map(min_relevance: float = 0.9) -> dict:
    """movieId -> bool, replicating load_hybrid_data's genome-tag enrichment
    (movielens.py, min_relevance=0.9 filter before the groupby)."""
    base_path = ensure_dataset("ml-25m")
    movies = pd.read_csv(base_path / "movies.csv")
    genome_tags = pd.read_csv(base_path / "genome-tags.csv")
    genome_scores = pd.read_csv(base_path / "genome-scores.csv")

    genome_scores = genome_scores[genome_scores["relevance"] >= min_relevance].copy()
    genome_data = genome_scores.merge(genome_tags, on="tagId", how="left")
    tag_agg = genome_data.groupby("movieId")["tag"].agg(list).reset_index()
    movies = movies.merge(tag_agg, on="movieId", how="left")

    return {
        int(mid): isinstance(tag, list)
        for mid, tag in zip(movies["movieId"], movies["tag"])
    }


def topk_candidate_ids(cand_ids: list, y_pred, k: int) -> list:
    """Argsort-based top-k extractor. cand_ids and y_pred must be aligned
    (same order), which holds for BiEncoderModel/BM25Model candidate-mode output."""
    y_pred = np.asarray(y_pred, dtype=float)
    k = min(k, len(cand_ids))
    order = np.argsort(-y_pred, kind="stable")[:k]
    return [cand_ids[i] for i in order]
