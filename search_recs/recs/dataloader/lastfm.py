import copy
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from search_recs.recs.dataloader import RecsDataLoader
from search_recs.datasets import ensure_dataset


PLAYS_FILE = "usersha1-artmbid-artname-plays.tsv"
PROFILE_FILE = "usersha1-profile.tsv"


def _find_repo_root(start: Path) -> Path:
    cur = start
    for _ in range(12):
        if (cur / "framework.py").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.parent


def _clean_artist_expr(column: str = "artist_name") -> pl.Expr:
    return pl.col(column).cast(pl.Utf8).str.to_lowercase().str.strip_chars()


def _scan_360k_plays(dataset_path: Path) -> pl.LazyFrame:
    plays_path = dataset_path / PLAYS_FILE
    if not plays_path.exists():
        raise FileNotFoundError(f"LastFM 360K plays file not found: {plays_path}")

    return (
        pl.scan_csv(
            plays_path,
            separator="\t",
            has_header=False,
            quote_char=None,
            new_columns=["user_name", "artist_mbid", "artist_name", "label"],
            schema_overrides={
                "user_name": pl.Utf8,
                "artist_mbid": pl.Utf8,
                "artist_name": pl.Utf8,
                "label": pl.Int64,
            },
            ignore_errors=True,
        )
        .drop_nulls(["user_name", "artist_name", "label"])
        .with_columns(
            pl.when(pl.col("artist_mbid").is_not_null() & (pl.col("artist_mbid").str.len_chars() > 0))
            .then(pl.col("artist_mbid"))
            .otherwise(_clean_artist_expr("artist_name"))
            .alias("artist_key")
        )
    )


def _load_360k_user_lookup(dataset_path: Path) -> pl.DataFrame:
    return (
        _scan_360k_plays(dataset_path)
        .select("user_name")
        .unique(maintain_order=True)
        .collect()
        .with_row_index("user")
    )


def _load_360k_artist_lookup(dataset_path: Path) -> pl.DataFrame:
    return (
        _scan_360k_plays(dataset_path)
        .select(["artist_key", "artist_name"])
        .unique(subset=["artist_key"], maintain_order=True)
        .collect()
        .with_row_index("item")
        .with_columns(_clean_artist_expr("artist_name").alias("artist_name_clean"))
    )


def _load_profile(dataset_path: Path, user_lookup: pl.DataFrame) -> pl.DataFrame:
    profile_path = dataset_path / PROFILE_FILE
    if not profile_path.exists():
        return user_lookup

    profile = pl.read_csv(
        profile_path,
        separator="\t",
        has_header=False,
        quote_char=None,
        new_columns=["user_name", "gender", "age", "country", "signup"],
        schema_overrides={
            "user_name": pl.Utf8,
            "gender": pl.Utf8,
            "age": pl.Int64,
            "country": pl.Utf8,
            "signup": pl.Utf8,
        },
        ignore_errors=True,
    )
    return user_lookup.join(profile, on="user_name", how="left")


class LastFM360KDataLoader(RecsDataLoader):
    """
    Local LastFM 360K dataloader.

    It reads the canonical TSV files stored in data/lastfm-dataset-360K.
    HetRec tags are still read from data/lastfm-hybrid when hybrid/tag-as-user
    mode is requested.
    """

    def __init__(self, dataloader_config: dict):
        if not isinstance(dataloader_config, dict):
            raise TypeError("dataloader_config must be a dict.")

        self.config_copy = copy.deepcopy(dataloader_config)
        cfg = dict(self.config_copy.get("dataloader", self.config_copy))
        cfg.setdefault("path", "lastfm-dataset-360K")

        super().__init__(cfg)

        self.dataset_path = ensure_dataset("lastfm-dataset-360K")
        self.hetrec_path = ensure_dataset("lastfm-hybrid")

        self.generate_pseudo_timestamps = bool(cfg.get("generate_pseudo_timestamps", True))
        self.pseudo_time_order = str(cfg.get("pseudo_time_order", "by_plays_asc")).lower()
        self.min_user_interactions = int(cfg.get("min_user_interactions", 2))

    def load_data(self) -> pd.DataFrame:
        """Standard 360K user-artist-playcount loading from local TSV files."""
        plays_lf = _scan_360k_plays(self.dataset_path)
        users = _load_360k_user_lookup(self.dataset_path)
        artists = _load_360k_artist_lookup(self.dataset_path)

        data = (
            plays_lf.join(users.lazy(), on="user_name", how="inner")
            .join(artists.lazy().select(["item", "artist_key", "artist_name"]), on="artist_key", how="inner")
            .select(["user", "item", "label", "artist_name", "user_name"])
            .collect()
        )

        user_counts = data.group_by("user").len()
        lower = user_counts.select(pl.col("len").quantile(0.10)).item()
        upper = user_counts.select(pl.col("len").quantile(0.90)).item()
        keep_users = user_counts.filter((pl.col("len") > lower) & (pl.col("len") < upper)).select("user")
        data = data.join(keep_users, on="user", how="inner")

        if self.generate_pseudo_timestamps:
            descending = self.pseudo_time_order in {"by_plays_desc", "desc", "descending"}
            data = (
                data.sort(["user", "label", "item"], descending=[False, descending, False])
                .with_columns(pl.col("item").cum_count().over("user").cast(pl.Int64).alias("time"))
            )
        else:
            data = data.with_columns(pl.lit(0).cast(pl.Int64).alias("time"))

        profiles_all = _load_profile(self.dataset_path, users)
        profiles = profiles_all.select(
            [c for c in ["user", "gender", "age", "country", "signup"] if c in profiles_all.columns]
        )
        data = data.join(profiles, on="user", how="left")

        return data.to_pandas()

    def hybrid_load_data(self) -> pd.DataFrame:
        """Tag-as-user mode: maps HetRec tags to local LastFM 360K artists by name."""
        artists_360k = _load_360k_artist_lookup(self.dataset_path).select(
            [
                pl.col("item").alias("item_360k"),
                pl.col("artist_name"),
                pl.col("artist_name_clean"),
            ]
        )

        hetrec_artists = (
            pl.read_csv(
                self.hetrec_path / "artists.dat",
                separator="\t",
                has_header=True,
                ignore_errors=True,
                quote_char=None,
            )
            .select([pl.col("id").alias("artistID_2k"), pl.col("name").alias("artist_name")])
            .with_columns(_clean_artist_expr("artist_name").alias("artist_name_clean"))
        )

        tags_df = (
            pl.scan_csv(
                self.hetrec_path / "user_taggedartists.dat",
                separator="\t",
                has_header=True,
                quote_char=None,
            )
            .group_by(["tagID", "artistID"])
            .agg(pl.lit(0.8).alias("label"))
            .rename({"artistID": "artistID_2k", "tagID": "user"})
            .collect()
            .join(hetrec_artists, on="artistID_2k", how="inner")
        )

        final_df = tags_df.join(artists_360k, on="artist_name_clean", how="inner")

        if self.min_user_interactions > 1:
            final_df = (
                final_df.join(final_df.group_by("user").len().rename({"len": "cnt"}), on="user")
                .filter(pl.col("cnt") >= self.min_user_interactions)
                .drop("cnt")
            )

        rng = np.random.default_rng(42)
        final_df = final_df.with_columns(
            [
                pl.Series("time", rng.integers(1, 1_000_000, size=final_df.height)),
                pl.col("item_360k").alias("item"),
            ]
        )

        return final_df.select(["user", "item", "label", "time", "artist_name"]).to_pandas()
