from pathlib import Path
from typing import Optional, Dict, Union
import copy
import pandas as pd
import polars as pl
import os
from search_recs.recs.dataloader import RecsDataLoader

class GenericRecsDataLoader(RecsDataLoader):
    """
    Generic DataLoader for Recommendations.
    Loads CSV, Parquet, or JSONL mapping columns via config.
    """

    def __init__(self, full_config: dict):
        # 1. Copy and normalize config
        self.config_copy = copy.deepcopy(full_config)
        
        if "dataloader" in self.config_copy:
            base_loader_config = self.config_copy["dataloader"]
        else:
            base_loader_config = self.config_copy

        # 2. Initialize parent class
        super().__init__(base_loader_config)
        
        # 3. Local settings
        self.file_name = base_loader_config.get("file_name")
        if not self.file_name:
            raise ValueError("Dataloader config must contain 'file_name' (e.g., ratings.csv).")

        # Resolve file path
        if Path(self.file_name).is_absolute():
             self.file_path = Path(self.file_name)
        else:
             self.file_path = self.dataset_path / self.file_name
        
        # Read & Format settings
        self.file_format = base_loader_config.get("file_format", str(self.file_name).split('.')[-1].lower())
        self.separator = base_loader_config.get("separator", ",")
        self.has_header = base_loader_config.get("has_header", True)
        
        # Mappings & Defaults
        self.column_mapping = base_loader_config.get("column_mapping", {})
        self.defaults = base_loader_config.get("defaults", {})
        
        # Sampling
        self.sample_fraction = base_loader_config.get("sample_fraction", 1.0)
        self.seed = base_loader_config.get("seed", 42)

        if not self.file_path.exists():
             # Fallback to nested folder
             nested_path = self.dataset_path / self.file_path.name
             if nested_path.exists():
                 self.file_path = nested_path
             else:
                 raise FileNotFoundError(f"Data file not found: {self.file_path}")

        print(f"[GenericLoader] Configured to read: {self.file_path}")

    def load_data(self) -> pd.DataFrame:
        """Reads file, renames columns, and returns a Pandas DataFrame."""
        
        # 1. Lazy Read (Polars)
        try:
            if self.file_format == 'csv':
                lf = pl.scan_csv(self.file_path, separator=self.separator, has_header=self.has_header)
            elif self.file_format == 'parquet':
                lf = pl.scan_parquet(self.file_path)
            elif self.file_format in ['json', 'jsonl', 'ndjson']:
                lf = pl.scan_ndjson(self.file_path)
            else:
                raise ValueError(f"Format '{self.file_format}' not supported.")
        except Exception as e:
             raise RuntimeError(f"Error initializing read for file {self.file_path}: {e}")

        # 2. Rename columns
        schema_cols = lf.collect_schema().names()
        
        # Apply mapping only to columns that exist in the file
        valid_mapping = {k: v for k, v in self.column_mapping.items() if k in schema_cols}
        
        if valid_mapping:
            lf = lf.rename(valid_mapping)
        
        # 3. Selection & Defaults
        target_cols = set(self.column_mapping.values())
        current_cols = lf.collect_schema().names()
        
        expressions = []
        
        # Mandatory Columns (User/Item)
        if 'user' in current_cols: 
            expressions.append(pl.col('user').cast(pl.Utf8))
        elif 'user' in target_cols:
             raise ValueError(f"Column 'user' mapped but not found. Current schema: {current_cols}")
        
        if 'item' in current_cols: 
            expressions.append(pl.col('item').cast(pl.Utf8))
        elif 'item' in target_cols:
             raise ValueError(f"Column 'item' mapped but not found. Current schema: {current_cols}")
            
        # Label (or default)
        if 'label' in current_cols:
            expressions.append(pl.col('label').cast(pl.Float32))
        else:
            default_label = self.defaults.get('label', 1.0)
            expressions.append(pl.lit(default_label).alias('label'))
            
        # Time (or default)
        if 'time' in current_cols:
            expressions.append(pl.col('time').cast(pl.Int64, strict=False).fill_null(0))
        else:
            default_time = self.defaults.get('time', 0)
            expressions.append(pl.lit(default_time).alias('time'))

        # Extra mapped columns
        extra_cols = [c for c in current_cols if c in target_cols and c not in ['user', 'item', 'label', 'time']]
        for c in extra_cols:
            expressions.append(pl.col(c))

        lf = lf.select(expressions)

        # 4. Sampling
        if self.sample_fraction < 1.0:
            print(f"[GenericLoader] Applying sampling: {self.sample_fraction*100}%")
            lf = lf.collect().sample(fraction=self.sample_fraction, seed=self.seed).lazy()

        # 5. Convert to Pandas
        df = lf.collect().to_pandas()
        
        print(f"[GenericLoader] Load finished. Shape: {df.shape}")
        
        return df