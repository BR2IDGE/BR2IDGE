import os
import zipfile
import urllib.request
import copy
from pathlib import Path
import polars as pl
import pandas as pd

from search_recs.recs.dataloader import RecsDataLoader

MOVIELENS_25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ZIP_FILENAME = "ml-25m.zip"
EXTRACTED_FOLDER = "ml-25m"

class MovieLens25MDataLoader(RecsDataLoader):
    """
    MovieLens 25M DataLoader:
    """

    def __init__(self, full_config: dict, sample_fraction: float = 0.1):
        # 1. Copy config to avoid side effects
        self.config_copy = copy.deepcopy(full_config)
        
        # 2. Prepare config for the parent class (RecsDataLoader)
        if "dataloader" in self.config_copy:
            base_loader_config = self.config_copy["dataloader"]
        else:
            # Fallback for flat config (legacy compatibility)
            base_loader_config = self.config_copy

        # 3. Inject mandatory dataset path
        base_loader_config["dataset_folder"] = EXTRACTED_FOLDER
        
        # 4. Init parent class with corrected config
        super().__init__(base_loader_config)

        # 5. Local setup
        cwd = Path(os.getcwd())
        self.data_root_path = cwd / "data"
        
        # Define dataset path as ./data/ml-25m
        self.dataset_path = self.data_root_path / EXTRACTED_FOLDER 
        self.MOVIELENS_URL = MOVIELENS_25M_URL
        self.ZIP_FILENAME = ZIP_FILENAME
        self.sample_fraction = sample_fraction

        # Ensure download
        if not self.dataset_path.is_dir():
            self._download_and_extract()
        
    def _download_and_extract(self):
        zip_path = self.data_root_path / self.ZIP_FILENAME
        os.makedirs(self.data_root_path, exist_ok=True)

        # Silent download
        urllib.request.urlretrieve(self.MOVIELENS_URL, zip_path)

        # Silent extraction
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_root_path)

        os.remove(zip_path)

    # ---------------------------------------------------------
    # MODE 1: Ratings (Original)
    # ---------------------------------------------------------
    def load_data(self) -> "pd.DataFrame":
        # Check for ratings.csv
        ratings_path = self.dataset_path / "ratings.csv"
        
        if not ratings_path.exists():
            nested_path = self.dataset_path / "ml-25m" / "ratings.csv"
            if nested_path.exists():
                ratings_path = nested_path
                self.dataset_path = self.dataset_path / "ml-25m"
            else:
                # Critical error
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
    
    # ---------------------------------------------------------
    # MODE 2: Tag-as-User
    # ---------------------------------------------------------
    def hybrid_load_data(self) -> "pd.DataFrame":
        """
        Internal logic: Loads 'genome-scores' in Tag-as-User format.
        """
        # Get config from 'dataloader' key, fallback to root
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

        # 2. Processing with Polars
        q = pl.scan_csv(genome_scores_path)
        q = q.filter(pl.col("relevance") > min_relevance)
        
        # 3. Column Transformation
        q = q.select([
            pl.col("tagId").alias("user"),      # Tag -> User
            pl.col("movieId").alias("item"),    # Movie -> Item
            pl.col("relevance").alias("label"), # Relevance -> Label
            pl.col("relevance").alias("time")   # Relevance -> Time (for sorting)
        ])
        
        df_pl = q.collect()
        
        return df_pl.to_pandas()