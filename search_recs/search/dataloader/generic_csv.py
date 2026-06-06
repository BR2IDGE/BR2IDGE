import pandas as pd
from typing import List, Union, Tuple
from sklearn.model_selection import train_test_split
from .base_dataloader import BaseSearchDatasetBuilder, BuildConfig

class GenericCSVBuilder(BaseSearchDatasetBuilder):
    def __init__(
        self, 
        csv_path: str, 
        features: dict, 
        csv_sep: str = ",", 
        cfg: BuildConfig = BuildConfig()
    ):
        super().__init__(cfg)
        self.csv_path = csv_path
        self.features = features
        self.csv_sep = csv_sep
        self.raw_df = None

    def load_raw(self) -> None:
        """Reads the generic CSV file."""
        print(f"[GenericLoader] Reading file: {self.csv_path}")
        self.raw_df = pd.read_csv(self.csv_path, sep=self.csv_sep)

    def build_pairs(self) -> pd.DataFrame:
        """Constructs the mapped DataFrame."""
        df = self.raw_df.copy()
        
        query_col = self.features.get("query_col")
        doc_cols = self.features.get("doc_cols")
        cat_col = self.features.get("cat_col")
        id_col = self.features.get("id_col")

        if not query_col or not doc_cols:
            raise ValueError("Configuration 'features' must contain 'query_col' and 'doc_cols'.")

        if isinstance(doc_cols, list):
            df["document"] = df[doc_cols].astype(str).agg('\n'.join, axis=1)
        else:
            df["document"] = df[doc_cols].astype(str)

        rename_map = {
            query_col: "search_query",
            id_col: "document_id"
        }
        
        if cat_col and cat_col in df.columns:
            rename_map[cat_col] = "category"
        else:
            df["category"] = "unknown"

        df = df.rename(columns=rename_map)

        required_cols = ["search_query", "document", "category"]
        if "document_id" in df.columns:
            required_cols.append("document_id")
            
        final_df = df[required_cols].dropna().drop_duplicates(subset=["search_query", "document"])
        return final_df

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits into Train, Validation, and Test sets."""
        test_size = self.cfg.test_size
        val_size = self.cfg.val_size
        
        if test_size + val_size >= 1.0:
             raise ValueError("The sum of test_size and val_size must be less than 1.0")

        temp_size = val_size + test_size
        train_df, temp_df = train_test_split(
            df, 
            test_size=temp_size, 
            random_state=self.cfg.random_state
        )

        relative_test_size = test_size / temp_size
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=relative_test_size, 
            random_state=self.cfg.random_state
        )

        if self.cfg.head_train:
            train_df = train_df.head(self.cfg.head_train)
        if self.cfg.head_val:
            val_df = val_df.head(self.cfg.head_val)
        if self.cfg.head_test:
            test_df = test_df.head(self.cfg.head_test)

        print(f"[GenericLoader] Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df

def load_generic_csv_dataset(cfg: BuildConfig, **kwargs):
    csv_path = kwargs.get("path")
    features = kwargs.get("features", {})
    csv_sep = kwargs.get("csv_separator", ",")
    
    if not csv_path:
        raise ValueError("The dataset JSON must contain 'path'.")

    builder = GenericCSVBuilder(
        csv_path=csv_path,
        features=features,
        csv_sep=csv_sep,
        cfg=cfg
    )
    
    return builder.build()