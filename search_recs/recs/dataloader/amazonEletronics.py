import os
import copy
import gzip
import shutil
import urllib.request
from pathlib import Path
from typing import Optional, List

import polars as pl
import pandas as pd
import numpy as np

from search_recs.recs.dataloader import RecsDataLoader
from search_recs.datasets import ensure_dataset

CSV_FILENAME = "ratings.csv"
META_FILENAME = "meta_Electronics.json.gz"
DATASET_FOLDER_NAME = "amazonElectronics"

class AmazonElectronicsDataLoader(RecsDataLoader):

    def __init__(self, full_config: dict, sample_fraction: float = 1.0):
        self.config_copy = copy.deepcopy(full_config)
        cfg = self.config_copy.get("dataloader", self.config_copy)

        if "dataset_folder" not in cfg:
            cfg["dataset_folder"] = DATASET_FOLDER_NAME

        self.auto_download = bool(cfg.get("auto_download", True))
        self.max_items = int(cfg.get("max_items", 200000))
        self.sample_fraction = sample_fraction

        super().__init__(cfg)

        self.dataset_path = ensure_dataset("amazonElectronics")
        self._ensure_paths()

    def _ensure_paths(self):
        self.dataset_path.mkdir(parents=True, exist_ok=True)

    def _download_file(self, url: str, target_path: Path):
        print(f"Downloading {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(target_path, "wb") as f:
            shutil.copyfileobj(resp, f)

    def load_data(self) -> pd.DataFrame:
        """Standard User-Item-Rating loading (Enforcing User-Item order)."""
        csv_path = self.dataset_path / CSV_FILENAME

        if not csv_path.exists():
            self.dataset_path = ensure_dataset("amazonElectronics")
            csv_path = self.dataset_path / CSV_FILENAME

        try:
            df = pl.read_csv(
                csv_path,
                has_header=False,
                separator=",",
                truncate_ragged_lines=True,
                new_columns=["item", "user", "label", "time"], 
                schema_overrides={
                    "item": pl.String,      
                    "user": pl.String,      
                    "label": pl.Float32,  
                    "time": pl.Int64,     
                },
                infer_schema_length=10000,
            )
        except Exception:
            df = pl.read_csv(csv_path, has_header=False, separator="\0", quote_char=None, new_columns=["raw_line"])
            df = df.select(pl.col("raw_line").str.extract_all(r"[^,\s]+").alias("parts"))
            df = df.filter(pl.col("parts").list.len() >= 4).select([
                pl.col("parts").list.get(0).alias("item"),
                pl.col("parts").list.get(1).alias("user"),
                pl.col("parts").list.get(2).alias("label"),
                pl.col("parts").list.get(3).alias("time"),
            ])

        df = df.select(["user", "item", "label", "time"])

        df = df.drop_nulls()
        
        df = df.with_columns([
            pl.col("label").cast(pl.Float32, strict=False),
            pl.col("time").cast(pl.Int64, strict=False),
        ]).drop_nulls()

        if self.sample_fraction < 1.0:
            df = df.sample(fraction=self.sample_fraction, seed=42)


        k = 10
        for _ in range(5):
            prev_h = df.height
            item_counts = df.group_by("item").len()
            df = df.join(item_counts, on="item").filter(pl.col("len") >= k).drop("len")
            
            user_counts = df.group_by("user").len()
            df = df.join(user_counts, on="user").filter(pl.col("len") >= k).drop("len")
            
            if df.height == prev_h:
                break

        print(f"Loaded {df.height:,} interactions (Standard Mode: User-Item ordered).")
        return df.to_pandas()

    def hybrid_load_data(self) -> pd.DataFrame:
        """Tag-as-User Mode: Categories become Users, ASINs become Items."""
        json_gz_path = self.dataset_path / META_FILENAME

        if not json_gz_path.exists():
            self.dataset_path = ensure_dataset("amazonElectronics")
            json_gz_path = self.dataset_path / META_FILENAME

        # Lazy processing of metadata
        try:
            q = pl.scan_ndjson(json_gz_path, infer_schema_length=10000)
        except:
            q = pl.read_ndjson(json_gz_path).lazy()

        q = q.select([pl.col("asin").alias("item"), pl.col("category")])

        # Handle different schema detections for 'category'
        cat_dtype = q.schema.get("category")

        if cat_dtype == pl.Utf8:
            q = q.with_columns(
                pl.col("category").str.replace_all(r"[\[\]'\" ]", "").str.split(",")
            ).explode("category")
        elif isinstance(cat_dtype, pl.List):
            q = q.explode("category")
            if isinstance(cat_dtype.inner, pl.List):
                q = q.explode("category")

        # Cleanup and Relevance Assignment (0.8)
        q = q.filter(pl.col("category").is_not_null() & (pl.col("category") != ""))
        q = q.select([
            pl.col("category").alias("user"),
            pl.col("item"),
            pl.lit(0.8).alias("label"),
            pl.lit(0).alias("time")
        ]).unique(subset=["user", "item"])

        df_pl = q.collect()

        # Popularity-based item limiting
        if self.max_items > 0 and df_pl["item"].n_unique() > self.max_items:
            top_items = (
                df_pl.group_by("item").len()
                .sort("len", descending=True)
                .head(self.max_items)
                .select("item")
            )
            df_pl = df_pl.join(top_items, on="item", how="inner")

        # Essential Stats Log
        print(f"Tag-as-User Stats:")
        print(f"  - Interactions: {df_pl.height:,}")
        print(f"  - Categories (Users): {df_pl['user'].n_unique():,}")
        print(f"  - Products (Items): {df_pl['item'].n_unique():,}")

        return df_pl.to_pandas()
