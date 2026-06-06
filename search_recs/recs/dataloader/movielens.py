import os
import copy
from pathlib import Path
import polars as pl
import pandas as pd

from search_recs.recs.dataloader import RecsDataLoader
from search_recs.datasets import ensure_dataset

EXTRACTED_FOLDER = "ml-25m"

class MovieLens25MDataLoader(RecsDataLoader):

    def __init__(self, full_config: dict, sample_fraction: float = 1):
        self.config_copy = copy.deepcopy(full_config)
        
        if "dataloader" in self.config_copy:
            base_loader_config = self.config_copy["dataloader"]
        else:
            base_loader_config = self.config_copy

        base_loader_config["dataset_folder"] = EXTRACTED_FOLDER
        
        super().__init__(base_loader_config)

        cwd = Path(os.getcwd())
        self.data_root_path = cwd / "data"

        self.dataset_path = ensure_dataset("ml-25m")
        self.sample_fraction = sample_fraction

    def load_data(self) -> "pd.DataFrame":
        # Check for ratings.csv
        ratings_path = self.dataset_path / "ratings.csv"
        
        if not ratings_path.exists():
            nested_path = self.dataset_path / "ml-25m" / "ratings.csv"
            if nested_path.exists():
                ratings_path = nested_path
                self.dataset_path = self.dataset_path / "ml-25m"
            else:
                raise FileNotFoundError(f"ratings.csv not found in {self.dataset_path}")

        ratings = (
            pl.read_csv(ratings_path)
            .sample(fraction=self.sample_fraction, with_replacement=False, seed=42)
        )

        # Calculate user activity
        user_activity = ratings.group_by("userId").agg(
            pl.count().alias("rating_count")
        )

        # Define activity thresholds (remove outliers)
        lower_threshold = user_activity.select(
            pl.col("rating_count").quantile(0.10)
        ).item()
        
        upper_threshold = user_activity.select(
            pl.col("rating_count").quantile(0.90)
        ).item()
        
        # Filter users within thresholds
        users_to_keep = user_activity.filter(
            (pl.col("rating_count") > lower_threshold) &
            (pl.col("rating_count") < upper_threshold)
        ).select("userId")
        
        ratings_filtered = ratings.join(
            users_to_keep, 
            on="userId", 
            how="inner"
        )
        
        movies = pl.read_csv(self.dataset_path / "movies.csv")

        data = ratings_filtered.join(movies, on="movieId", how="left")
        
        # Standardize column names
        data = data.rename({
            "rating": "label",
            "userId": "user",
            "timestamp": "time",
            "movieId": "item"
        })

        data_pd = data.to_pandas()
        return data_pd
    
    def hybrid_load_data(self) -> "pd.DataFrame":
        dl_config = self.config_copy.get("dataloader", self.config_copy)
        
        min_relevance = float(dl_config.get("min_relevance", 0.8))
        genome_scores_path = self.dataset_path / "genome-scores.csv"

        # Subfolder fallback
        if not genome_scores_path.exists():
            nested_path = self.dataset_path / "ml-25m" / "genome-scores.csv"
            if nested_path.exists():
                genome_scores_path = nested_path
                self.dataset_path = self.dataset_path / "ml-25m"
            else:
                raise FileNotFoundError(f"genome-scores.csv not found in {self.dataset_path}")

        q = pl.scan_csv(genome_scores_path)
        q = q.filter(pl.col("relevance") > min_relevance)
        
        q = q.select([
            pl.col("tagId").alias("user"),      
            pl.col("movieId").alias("item"),    
            pl.col("relevance").alias("label"), 
            pl.col("relevance").alias("time")   
        ])
        
        df_pl = q.collect()
        
        return df_pl.to_pandas()
