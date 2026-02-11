import os
import zipfile
import urllib.request
import copy
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np

from search_recs.recs.dataloader import RecsDataLoader

try:
    from implicit.datasets.lastfm import get_lastfm
except ImportError:
    get_lastfm = None

def ensure_lastfm_hetrec_exists(target_path: Path):
    url = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
    zip_path = target_path.parent / "hetrec2011_lastfm_2k.zip"

    if (target_path / "user_taggedartists.dat").exists() and (target_path / "artists.dat").exists():
        return

    os.makedirs(target_path, exist_ok=True)
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_path)

    for fname in ["artists.dat", "user_taggedartists.dat"]:
        if not (target_path / fname).exists():
            found = list(target_path.rglob(fname))
            if found:
                (target_path / fname).write_bytes(found[0].read_bytes())

    if zip_path.exists():
        os.remove(zip_path)

def _find_repo_root(start: Path) -> Path:
    cur = start
    for _ in range(12):
        if (cur / "framework.py").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.parent

class LastFM360KDataLoader(RecsDataLoader):
    """
    LastFM Dataloader approach, merging 
    LastFM 360k (plays) with HetRec 2k (tags).
    """

    def __init__(self, dataloader_config: dict):
        if not isinstance(dataloader_config, dict):
            raise TypeError("dataloader_config must be a dict.")

        self.config_copy = copy.deepcopy(dataloader_config)
        cfg = dict(self.config_copy.get("dataloader", self.config_copy))
        cfg.setdefault("path", "lastfm360k")

        super().__init__(cfg)

        repo_root = _find_repo_root(Path(__file__).resolve())
        self.dataset_path = repo_root / "data" / cfg["path"]
        self.hetrec_path = repo_root / "data" / "lastfm_hetrec"

        self.generate_pseudo_timestamps = bool(cfg.get("generate_pseudo_timestamps", True))
        self.min_user_interactions = int(cfg.get("min_user_interactions", 2))

        if get_lastfm is None:
            raise ImportError("implicit.datasets.lastfm.get_lastfm not found. Install 'implicit'.")

    def load_data(self) -> pd.DataFrame:
        """Standard 360k User-Artist-Plays loading."""
        artists, users, artist_user_plays = get_lastfm()
        coo = artist_user_plays.tocoo(copy=False)
        
        df = pd.DataFrame({
            "item": coo.row.astype("int64"),
            "user": coo.col.astype("int64"),
            "label": coo.data.astype("int64"),
        })

        # Basic filtering to remove outliers
        user_counts = df.groupby("user").size()
        keep_users = user_counts[
            (user_counts > user_counts.quantile(0.10)) &
            (user_counts < user_counts.quantile(0.90))
        ].index
        df = df[df["user"].isin(keep_users)].copy()

        df["artist_name"] = pd.Series(artists).iloc[df["item"].values].values
        df["user_name"] = pd.Series(users).iloc[df["user"].values].values
        df["time"] = (df.groupby("user").cumcount() + 1).astype("int64") if self.generate_pseudo_timestamps else 0

        return df[["user", "item", "label", "time", "artist_name", "user_name"]]

    def hybrid_load_data(self) -> "pd.DataFrame":
        """Tag-as-User Mode: Merges 360k artists with HetRec tags."""
        ensure_lastfm_hetrec_exists(self.hetrec_path)

        artists_360k_names, _, _ = get_lastfm()
        artists_360k_df = pl.DataFrame({
            "item_360k": range(len(artists_360k_names)),
            "artist_name": artists_360k_names
        }).with_columns(pl.col("artist_name").str.to_lowercase().str.strip_chars())

        hetrec_artists = (
            pl.read_csv(self.hetrec_path / "artists.dat", separator="\t", has_header=True, ignore_errors=True, quote_char=None)
            .select([pl.col("id").alias("artistID_2k"), pl.col("name").alias("artist_name")])
            .with_columns(pl.col("artist_name").str.to_lowercase().str.strip_chars())
        )

        # Mapping tags to artists with constant relevance 0.8
        tags_df = (
            pl.scan_csv(self.hetrec_path / "user_taggedartists.dat", separator="\t", has_header=True, quote_char=None)
            .group_by(["tagID", "artistID"])
            .agg(pl.lit(0.8).alias("label"))
            .rename({"artistID": "artistID_2k", "tagID": "user"})
            .collect()
            .join(hetrec_artists, on="artistID_2k", how="inner")
        )

        final_df = tags_df.join(artists_360k_df, on="artist_name", how="inner")

        if self.min_user_interactions > 1:
            final_df = (
                final_df.join(final_df.group_by("user").len().rename({"len": "cnt"}), on="user")
                .filter(pl.col("cnt") >= self.min_user_interactions)
                .drop("cnt")
            )

        # Generate pseudo-timestamps for splitting logic
        rng = np.random.default_rng(42)
        final_df = final_df.with_columns([
            pl.Series("time", rng.integers(1, 1_000_000, size=final_df.height)),
            pl.col("item_360k").alias("item")
        ])

        return final_df.select(["user", "item", "label", "time", "artist_name"]).to_pandas()