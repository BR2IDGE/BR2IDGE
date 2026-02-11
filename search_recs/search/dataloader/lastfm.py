import pandas as pd
import numpy as np
import random as r
import pathlib
import requests
import zipfile
import io
import os
from typing import Tuple, Optional
import time
import random
from sklearn.model_selection import train_test_split
import json

try:
    from implicit.datasets.lastfm import get_lastfm
except ImportError as e:
    raise ImportError("Install 'implicit' to download the 360K dataset: pip install implicit") from e

from .base_dataloader import BaseSearchDatasetBuilder, BuildConfig


class LastFmHybridBuilder(BaseSearchDatasetBuilder):
    TAGS_URL = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
    
    def __init__(self, data_dir: str, cfg: BuildConfig = BuildConfig()):
        super().__init__(cfg)
        self.data_dir = pathlib.Path(data_dir)
        
        self.tags_path = self.data_dir / "tags.dat"
        self.tagging_path = self.data_dir / "user_taggedartists.dat"
        self.artists_2k_path = self.data_dir / "artists.dat"
        
        self.lfm360_artists = None
        self.lfm360_plays = None
        self.tags_df = None
        self.tagging_df = None
        self.artists_2k_df = None
        
        r.seed(cfg.random_state)

    def _download_2k_tags(self):
        if self.tags_path.exists() and self.tagging_path.exists():
            return

        print(f"[DataLoader] Downloading Tags metadata (2K Base) to {self.data_dir}...")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(self.TAGS_URL, stream=True)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(self.data_dir)
        except Exception as e:
            raise RuntimeError(f"Error downloading auxiliary Tags: {e}")

    def load_raw(self) -> None:
        print("[DataLoader] Loading Last.fm 360K (via implicit)...")
        self.lfm360_artists, _, self.lfm360_plays = get_lastfm() 

        self._download_2k_tags()
        print("[DataLoader] Loading Tags files...")
        
        self.tags_df = pd.read_csv(self.tags_path, sep="\t", encoding="latin-1", on_bad_lines='skip')
        self.tagging_df = pd.read_csv(self.tagging_path, sep="\t", encoding="latin-1", on_bad_lines='skip')
        self.artists_2k_df = pd.read_csv(self.artists_2k_path, sep="\t", encoding="latin-1", on_bad_lines='skip')

    def build(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.load_raw()
        return pd.DataFrame(), pd.DataFrame()

def _generate_random_seed():
    now = time.time() * 1000
    random_factor = random.random()
    return int((random_factor * now) % 10000)

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

    def decode(x):
        return x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, bytearray)) else str(x)

    # --- 1. TAG EXTRACTION AND MAPPING ---
    # Intersects tags with artist names (bridge between 2k and 360k datasets)
    df_tags_full = builder.tagging_df.merge(builder.tags_df, on="tagID")
    tag_map_2k = df_tags_full.groupby("artistID")["tagValue"].apply(lambda x: list(x.unique())[:15]).to_dict()
    
    id2name = builder.artists_2k_df.set_index("id")["name"].str.lower().str.strip().to_dict()
    name2tags = {id2name[aid]: tgs for aid, tgs in tag_map_2k.items() if aid in id2name}

    # --- 2. CLEAN CORPUS CONSTRUCTION ---
    artist_docs = [decode(x) for x in builder.lfm360_artists]
    
    # Creates the tag map ONLY for use in QUERY generation (Fusion)
    item_tag_map = {} 
    for i, name_bytes in enumerate(builder.lfm360_artists):
        name_clean = decode(name_bytes).lower().strip()
        tags = name2tags.get(name_clean, [])
        if tags:
            item_tag_map[str(i)] = {t: 1.0 for t in tags}

    # corpus_lookup is purely the normal representation
    corpus_lookup = pd.DataFrame({
        "document_id": np.arange(len(artist_docs), dtype=int),
        "document": pd.Series(artist_docs, dtype=str), # Original name only
    })

    # --- 3. INTERACTION PROCESSING AND SPLIT ---
    plays = builder.lfm360_plays.tocsc()
    n_users = plays.shape[1]
    all_items = np.arange(len(artist_docs), dtype=int)
    rs = int(time.time() * 1000) % (2**31 - 1)
    rng = np.random.default_rng(rs)

    train_rows, test_rows = [], []
    indptr, indices, data = plays.indptr, plays.indices, plays.data

    for uid in range(n_users):
        s, e = indptr[uid], indptr[uid + 1]
        user_items = indices[s:e].astype(int)
        user_vals = data[s:e]

        if user_items.size < (min_hist_len + 2): continue
        if user_items.size > max_test_items:
            pick = rng.choice(np.arange(user_items.size), size=max_test_items, replace=False)
            user_items, user_vals = user_items[pick], user_vals[pick]

        perm = rng.permutation(user_items.size)
        cut = max(1, int(user_items.size * train_ratio))
        train_items, test_items = user_items[perm[:cut]].tolist(), user_items[perm[cut:]].tolist()

        if len(train_items) < (min_hist_len + 1) or len(test_items) < 2: continue

        val_map = {int(i): float(v) for i, v in zip(user_items.tolist(), user_vals.tolist())}
        seq = sorted(train_items, key=lambda it: (-val_map.get(int(it), 0.0), int(it)))

        for t in range(min_hist_len, len(seq)):
            hist = seq[max(0, t - max_hist_len):t]
            train_rows.append({"search_query": ",".join(map(str, hist)), "document_id": int(seq[t]), "category": "UserHistoryTrain"})

        half = len(test_items) // 2
        query_items, gt_items = test_items[:half], test_items[half:]
        
        seen = set(train_items) | set(query_items) | set(gt_items)
        possible = np.setdiff1d(all_items, np.fromiter(seen, dtype=int))
        negs = rng.choice(possible, size=min(n_neg, len(possible)), replace=False).tolist()

        test_rows.append({
            "userId": int(uid),
            "search_query": ",".join(map(str, query_items)),
            "ground_truth_ids": json.dumps(list(map(int, gt_items))),
            "candidate_ids": json.dumps(list(map(int, gt_items)) + negs),
            "category": "UserHistoryTestUserwise",
            "document_id": -1, "document": ""
        })

    train_pairs = pd.DataFrame(train_rows)
    train_df = train_pairs.merge(corpus_lookup, on="document_id", how="inner")
    test_df = pd.DataFrame(test_rows)
    
    print(f"[lastfm-hybrid] Tags processed for {len(item_tag_map)} artists.")
    return train_df.reset_index(drop=True), pd.DataFrame(columns=train_df.columns), test_df.reset_index(drop=True)

def load_lastfm_search_dataset(
    cfg: BuildConfig,
    dataset_path: str = "./data/lastfm-hybrid",
    min_tag_freq: int = 50,
    top_k_tags: int = 20
):
    r.seed(cfg.random_state)
    
    builder = LastFmHybridBuilder(data_dir=dataset_path, cfg=cfg)
    builder.load_raw()

    def decode(x):
        return x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else str(x)
    
    lfm360_names_clean = [decode(x).strip().lower() for x in builder.lfm360_artists]
    lfm360_names_original = [decode(x) for x in builder.lfm360_artists]
    
    plays_sum = np.array(builder.lfm360_plays.sum(axis=1)).flatten()
    
    df_360 = pd.DataFrame({
        "artist_name_clean": lfm360_names_clean,
        "original_name": lfm360_names_original,
        "global_plays": plays_sum
    })
    
    df_360 = df_360[df_360["global_plays"] > 0]

    builder.artists_2k_df["name_clean"] = builder.artists_2k_df["name"].astype(str).str.lower().str.strip()
    
    df_tags_full = builder.tagging_df.merge(builder.tags_df, on="tagID")
    df_tags_full = df_tags_full.merge(
        builder.artists_2k_df[["id", "name_clean"]], 
        left_on="artistID", right_on="id"
    )
    
    df_tags_full["tagValue"] = df_tags_full["tagValue"].astype(str).str.lower().str.strip()
    tag_counts = df_tags_full["tagValue"].value_counts()
    valid_tags = tag_counts[tag_counts >= min_tag_freq].index
    
    df_tags_filtered = df_tags_full[df_tags_full["tagValue"].isin(valid_tags)].copy()
    
    df_tags_grouped = (
        df_tags_filtered.groupby(["name_clean", "tagValue"])
        .size()
        .reset_index(name="tag_weight")
    )
    df_tags_grouped = df_tags_grouped.sort_values(["name_clean", "tag_weight"], ascending=[True, False])
    df_best_tags = df_tags_grouped.groupby("name_clean").head(top_k_tags)

    df_final = df_360.merge(df_best_tags, left_on="artist_name_clean", right_on="name_clean", how="inner")
    
    search_df = pd.DataFrame({
        "search_query": df_final["tagValue"],
        "document": df_final["original_name"],
        "document_id": df_final["artist_name_clean"],
        "category": "music_artist"
    })

    search_df["search_query"] = search_df["search_query"].astype(str)
    search_df["document"] = search_df["document"].astype(str)
    
    mask = (search_df["search_query"].str.strip() != "") & \
           (search_df["search_query"].str.lower() != "nan") & \
           (search_df["document"].str.strip() != "")
           
    search_df = search_df[mask].drop_duplicates()

    # --- STATISTICS ---
    unique_queries = search_df["search_query"].nunique()
    print(f"[LastFM-Search] Total pairs: {len(search_df)}")
    print(f"[LastFM-Search] Total UNIQUE QUERIES (Tags): {unique_queries}")
    # ------------------

    test_size = float(cfg.test_size)
    val_size = float(cfg.val_size)
    if (test_size + val_size) >= 1.0: test_size, val_size = 0.1, 0.1
    
    total_eval = test_size + val_size
    
    train_df, eval_df = train_test_split(
        search_df, 
        test_size=total_eval, 
        random_state=_generate_random_seed(),
        shuffle=True
    )
    
    relative_test = test_size / total_eval if total_eval > 0 else 0.5
    val_df, test_df = train_test_split(
        eval_df, 
        test_size=relative_test, 
        random_state=_generate_random_seed(),
        shuffle=True
    )

    if getattr(cfg, "head_train", None): train_df = train_df.head(cfg.head_train)
    if getattr(cfg, "head_val", None): val_df = val_df.head(cfg.head_val)
    if getattr(cfg, "head_test", None): test_df = test_df.head(cfg.head_test)

    print(f"[DataLoader] Last.fm Search loaded (Hybrid 360K/2K).")
    print(f"   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    return train_df, val_df, test_df

# -------------------------------------------------------------------------
# AUTOMATIC SELECTOR (MAIN ENTRY POINT)
# -------------------------------------------------------------------------

def load_lastfm_dataset(cfg: BuildConfig, **kwargs):
    """
    Automatic selector for the Last.fm dataset.
    Expected arguments in kwargs:
      - mode: "hybrid" (default) or "search"
      - dataset_path: path to the data directory
      - other params specific to each loader (train_ratio, top_k_tags, etc.)
    """
    mode = kwargs.get("mode", "hybrid")
    dataset_path = kwargs.get("dataset_path", "./data/lastfm-hybrid")

    if mode == "hybrid":
        print(f"[DataLoader] Starting Last.fm HYBRID mode (User History)")
        return load_lastfm_hybrid_dataset(
            cfg=cfg,
            dataset_path=dataset_path,
            train_ratio=kwargs.get("train_ratio", 0.6),
            n_neg=kwargs.get("n_neg", 200),
            max_test_items=kwargs.get("max_test_items", 40),
            min_hist_len=kwargs.get("min_hist_len", 3),
            max_hist_len=kwargs.get("max_hist_len", 50),
            max_pairs_per_user=kwargs.get("max_pairs_per_user", 50)
        )
    else:
        # 'search' mode (Search Query -> Document)
        print(f"[DataLoader] Starting Last.fm SEARCH mode (Tags -> Artists)")
        return load_lastfm_search_dataset(
            cfg=cfg,
            dataset_path=dataset_path,
            min_tag_freq=kwargs.get("min_tag_freq", 50),
            top_k_tags=kwargs.get("top_k_tags", 20)
        )

# Alias to maintain compatibility with frameworks calling the standard loader
load_lastfm_user_query_dataset = load_lastfm_dataset