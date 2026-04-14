import random as r
import json
import pathlib
import os
from typing import Optional, Tuple, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
from .base_dataloader import BaseSearchDatasetBuilder, BuildConfig
import numpy as np
import time
from search_recs.datasets import ensure_dataset

# -------------------------------------------------------------------------
# BUILD AND LOADING CLASSES
# -------------------------------------------------------------------------

class MovieLensBuilder(BaseSearchDatasetBuilder):
    def __init__(self, movies_path: str, tags_path: str, cfg: BuildConfig = BuildConfig()):
        super().__init__(cfg)
        self.movies_path = movies_path
        self.tags_path = tags_path
        r.seed(cfg.random_state)

    @staticmethod
    def create_movie_document(row: dict, include_tags: bool = False) -> str:
        title = str(row.get("title", "")).strip()
        genres = str(row.get("genres", "")).replace("|", " ")
        doc = f"{title} \n {genres}"
        
        if include_tags and "tag" in row and isinstance(row["tag"], list):
            valid_tags = [str(t) for t in row["tag"] if t and str(t).lower() != "nan"]
            if valid_tags:
                doc += f" \n {' '.join(valid_tags)}"
        return doc

class MovieLensDataLoader:
    def __init__(self, cfg: BuildConfig, base_path: pathlib.Path):
        self.cfg = cfg
        self.base_path = base_path
        if cfg.random_state is not None:
            r.seed(cfg.random_state)

    # -----------------------------
    # Helper methods for lightweight hybrid mode
    # -----------------------------
    def _runtime_seed(self) -> int:
        """
        If cfg.random_state is set, use it (reproducible).
        Otherwise, generate a time-based seed.
        """
        # if self.cfg.random_state is not None:
        #    return int(self.cfg.random_state)
        return int(time.time() * 1000) % (2**31 - 1)

    def _split_userwise_chrono_ratio(self, ratings: pd.DataFrame, train_ratio: float = 0.6):
        """
        Chronological split per user.
        Returns train_ratings and test_ratings.
        """
        ratings = ratings.sort_values(["userId", "timestamp"], kind="mergesort").reset_index(drop=True)
        g = ratings.groupby("userId", sort=False)
        idx = g.cumcount()
        size = g["userId"].transform("size")
        cut = (size * train_ratio).astype(int)

        train_mask = idx < cut
        test_mask = ~train_mask

        train_df = ratings.loc[train_mask].copy()
        test_df = ratings.loc[test_mask].copy()
        return train_df, test_df

    def _cap_test_interactions(self, test_ratings: pd.DataFrame, max_test_items: int = 40):
        """
        Limits test interactions to a maximum of `max_test_items` per user 
        (selecting the most recent ones within the test block).
        """
        test_ratings = test_ratings.sort_values(["userId", "timestamp"], kind="mergesort")
        # Selects the last max_test_items within the test block
        return test_ratings.groupby("userId", sort=False).tail(max_test_items).copy()

    def _split_test_into_query_and_gt(self, test_ratings_user: pd.DataFrame):
        """
        Given the user's test block (already capped), splits it into query (history) and ground truth.
        """
        items = test_ratings_user["movieId"].tolist()
        if len(items) < 2:
            return [], []  # Cannot split

        half = len(items) // 2
        # First half -> query; second half -> GT
        query_items = items[:half]
        gt_items = items[half:]
        if len(gt_items) == 0:
            # Edge case: if len=2 and half=1, ok; if len=1 it returned early
            return [], []
        return query_items, gt_items

    def _sample_negatives(self, rng: np.random.Generator, all_items: np.ndarray, excluded_set: set, n_neg: int = 200):
        """
        Samples negatives without replacement, respecting excluded items.
        """
        # Filtered catalog (np.setdiff1d handles uniqueness)
        possible = np.setdiff1d(all_items, np.fromiter(excluded_set, dtype=all_items.dtype), assume_unique=False)
        if len(possible) == 0:
            return []
        k = min(n_neg, len(possible))
        return rng.choice(possible, size=k, replace=False).tolist()

    def _build_train_pairs_from_train_hist(
        self,
        train_ratings: pd.DataFrame,
        max_pairs_per_user: int = 50,
        min_hist_len: int = 3,
        max_hist_len: int = 50,
        seed: int = 48,
    ) -> pd.DataFrame:
        """
        Constructs training examples (history -> next item) ONLY from the training split (60%), 
        with sampling to manage memory usage.

        For each user:
          - Use chronological sequence from training.
          - Generate pairs (hist_prefix -> item_t).
          - Sample up to max_pairs_per_user pairs per user.
        """
        rng = np.random.default_rng(seed)
        train_ratings = train_ratings.sort_values(["userId", "timestamp"], kind="mergesort")

        rows = []
        for uid, g in train_ratings.groupby("userId", sort=False):
            seq = g["movieId"].tolist()
            if len(seq) < (min_hist_len + 1):
                continue

            # Target candidates: from index min_hist_len to end
            t_indices = list(range(min_hist_len, len(seq)))

            # Sample targets to limit examples
            if len(t_indices) > max_pairs_per_user:
                t_indices = rng.choice(t_indices, size=max_pairs_per_user, replace=False).tolist()
                t_indices.sort()

            for t in t_indices:
                hist = seq[max(0, t - max_hist_len):t]
                if len(hist) < min_hist_len:
                    continue
                target = seq[t]
                rows.append({
                    "search_query": ",".join(map(str, hist)),
                    "document_id": int(target),
                    "category": "UserHistoryTrain",
                })

        return pd.DataFrame(rows)

    def _build_hybrid_test_rows_userwise(
        self,
        train_ratings: pd.DataFrame,
        test_ratings: pd.DataFrame,
        all_items: np.ndarray,
        n_neg: int = 200,
        seed: int = 48,
    ) -> pd.DataFrame:
        """
        Creates a lightweight test dataset with 1 row per user:
          - search_query: ids (half of test) -> embedding average
          - ground_truth_ids: list of relevant items (half of test)
          - candidate_ids: ground_truth + 200 negatives
        """
        rng = np.random.default_rng(seed)

        # Items seen in training per user
        train_seen = train_ratings.groupby("userId")["movieId"].agg(set).to_dict()

        test_ratings = test_ratings.sort_values(["userId", "timestamp"], kind="mergesort")
        rows = []

        for uid, g in test_ratings.groupby("userId", sort=False):
            query_items, gt_items = self._split_test_into_query_and_gt(g)
            if not query_items or not gt_items:
                continue

            seen = set()
            seen |= train_seen.get(uid, set())
            seen |= set(query_items)
            seen |= set(gt_items)

            negs = self._sample_negatives(rng, all_items, excluded_set=seen, n_neg=n_neg)
            candidate_ids = list(map(int, gt_items)) + list(map(int, negs))

            rows.append({
                "userId": int(uid),
                "search_query": ",".join(map(str, query_items)),
                # Serialize as JSON string for DataFrame/CSV stability
                "ground_truth_ids": json.dumps(list(map(int, gt_items))),
                "candidate_ids": json.dumps(list(map(int, candidate_ids))),
                "category": "UserHistoryTestUserwise",
            })

        return pd.DataFrame(rows)

    def _split_and_sample(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        rs = 48  # Can be None

        train_val, test_df = train_test_split(
            df,
            test_size=self.cfg.test_size,
            random_state=rs,
            shuffle=True,  # Important
        )

        val_rel = self.cfg.val_size / (1.0 - self.cfg.test_size) if (1.0 - self.cfg.test_size) > 0 else 0.1
        train_df, val_df = train_test_split(
            train_val,
            test_size=min(0.9, val_rel),
            random_state=rs,
            shuffle=True,
        )

        # If head_train/head_test is configured, randomize before slicing
        if self.cfg.head_train:
            train_df = train_df.sample(n=min(self.cfg.head_train, len(train_df)), random_state=rs)
        if self.cfg.head_test:
            test_df = test_df.sample(n=min(self.cfg.head_test, len(test_df)), random_state=rs)

        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


    def load_search_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("Mode: Search")
        movies = pd.read_csv(self.base_path / "movies.csv")
        genome_tags = pd.read_csv(self.base_path / "genome-tags.csv")
        genome_scores = pd.read_csv(self.base_path / "genome-scores.csv")

        # Relevance filter for search
        genome_scores = genome_scores[genome_scores["relevance"] > 0.8].copy()
        genome_data = genome_scores.merge(genome_tags, on="tagId", how="left")

        # Corpus creation with Tags
        tag_agg = genome_data.groupby("movieId")["tag"].agg(list).reset_index()
        movies_enriched = movies.merge(tag_agg, on="movieId", how="left")
        movies_enriched["document"] = movies_enriched.apply(
            lambda r: MovieLensBuilder.create_movie_document(r, include_tags=False), axis=1
        )
        
        corpus_lookup = movies_enriched[["movieId", "document"]].rename(columns={"movieId": "document_id"})
        final_df = genome_data.merge(corpus_lookup, left_on="movieId", right_on="document_id")
        final_df = final_df.rename(columns={"tag": "search_query"}).dropna()
        final_df["category"] = "TagSearch"

        # --- STATISTICS ---
        unique_queries = final_df["search_query"].nunique()
        print(f"[MovieLens-Search] Total pairs: {len(final_df)}")
        print(f"[MovieLens-Search] Total UNIQUE QUERIES (Tags): {unique_queries}")
        # ------------------

        return self._split_and_sample(final_df[["search_query", "document", "document_id", "category"]])

    def load_hybrid_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("Mode: Hybrid (Lightweight 60/40 + Candidates)")

        rs = self._runtime_seed()
        print(f"[hybrid] Seed used (rs) = {rs}")

        ratings = pd.read_csv(self.base_path / "ratings.csv")
        movies = pd.read_csv(self.base_path / "movies.csv")

        # Document corpus
        movies["document"] = movies.apply(MovieLensBuilder.create_movie_document, axis=1)
        corpus_lookup = movies[["movieId", "document"]].rename(columns={"movieId": "document_id"})
        corpus_lookup["document_id"] = corpus_lookup["document_id"].astype(int)

        # ---- SPLIT 60/40 PER USER ----
        train_ratings, test_ratings = self._split_userwise_chrono_ratio(ratings, train_ratio=0.6)

        # Test cap: max 40 per user (from test block)
        test_ratings = self._cap_test_interactions(test_ratings, max_test_items=40)

        # Item catalog
        all_items = pd.unique(ratings["movieId"].astype(int).to_numpy())

        # ---- TRAINING: pairs (hist -> next) with user limit ----
        max_pairs_per_user = int(getattr(self.cfg, "max_pairs_per_user", 50) or 50)
        max_hist_len = int(getattr(self.cfg, "max_hist_len", 50) or 50)
        min_hist_len = int(getattr(self.cfg, "min_hist_len", 3) or 3)

        train_pairs = self._build_train_pairs_from_train_hist(
            train_ratings=train_ratings,
            max_pairs_per_user=max_pairs_per_user,
            min_hist_len=min_hist_len,
            max_hist_len=max_hist_len,
            seed=rs,
        )

        # Merge to add document text
        train_df = train_pairs.merge(corpus_lookup, on="document_id", how="inner")

        if getattr(self.cfg, "head_train", None):
            limit = int(self.cfg.head_train)
            if len(train_df) > limit:
                print(f"[hybrid] Applying head_train: reducing train from {len(train_df)} to {limit}")
                train_df = train_df.sample(n=limit, random_state=rs)

        # ---- TESTING: 1 row per user with candidates (GT + 200 negatives) ----
        test_userwise = self._build_hybrid_test_rows_userwise(
            train_ratings=train_ratings,
            test_ratings=test_ratings,
            all_items=all_items,
            n_neg=200,
            seed=rs,
        )

        # Add document column to maintain schema compatibility (unused in candidate mode)
        test_df = test_userwise.copy()
        test_df["document_id"] = -1
        test_df["document"] = ""

        if getattr(self.cfg, "head_test", None):
            limit = int(self.cfg.head_test)
            if len(test_df) > limit:
                print(f"[hybrid] Applying head_test: reducing test users from {len(test_df)} to {limit}")
                test_df = test_df.sample(n=limit, random_state=rs)

        # Validation set not used here
        val_df = pd.DataFrame(columns=train_df.columns)

        print(f"[hybrid] Final Train pairs: {len(train_df)} | Final Test users: {len(test_df)}")
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
    
# -------------------------------------------------------------------------
# INTERFACE FUNCTIONS (ENTRY POINTS)
# -------------------------------------------------------------------------

def load_movielens_dataset(cfg: BuildConfig, dataset_path: str = "./data/ml-25m", **kwargs):
    """
    Automatic selector: determines mode based on 'mode' parameter (default: search).
    """
    mode = kwargs.get("mode", "normal")
    base_path = ensure_dataset("ml-25m")
    
    # Path fallbacks
    if not base_path.exists():
        for p in [pathlib.Path("..") / dataset_path, pathlib.Path("data/ml-25m")]:
            if p.exists():
                base_path = p
                break
    
    if not base_path.exists():
        raise FileNotFoundError(f"MovieLens dataset path not found: {base_path}")

    loader = MovieLensDataLoader(cfg, base_path)
    
    if mode == "hybrid":
        print(f"[DataLoader] Starting HYBRID mode (User History)")
        return loader.load_hybrid_data()
    else:
        print(f"[DataLoader] Starting SEARCH mode (Semantic Tags)")
        return loader.load_search_data()

def load_genome_tag_map(dataset_path: str, min_relevance: float = 0.9) -> dict:
    base_path = pathlib.Path(dataset_path)
    scores_path = base_path / "genome-scores.csv"
    tags_path = base_path / "genome-tags.csv"
    
    if not scores_path.exists():
        scores_path = base_path / "ml-25m" / "genome-scores.csv"
        tags_path = base_path / "ml-25m" / "genome-tags.csv"

    if not scores_path.exists():
        return {}

    tag_id_to_name = dict(zip(pd.read_csv(tags_path)['tagId'], pd.read_csv(tags_path)['tag']))
    movie_tag_map = {}
    
    with pd.read_csv(scores_path, chunksize=1000000) as reader:
        for chunk in reader:
            filtered = chunk[chunk['relevance'] > min_relevance]
            for _, row in filtered.iterrows():
                mid = str(int(row['movieId']))
                tname = tag_id_to_name.get(int(row['tagId']))
                if tname:
                    if mid not in movie_tag_map: movie_tag_map[mid] = {}
                    movie_tag_map[mid][tname] = float(row['relevance'])
                    
    return movie_tag_map

def load_movielens_user_query_dataset(cfg: BuildConfig, **kwargs):
    """Alias for legacy compatibility."""
    return load_movielens_dataset(cfg, **kwargs)
