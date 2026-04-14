import pandas as pd
import numpy as np
import json
import gzip
import urllib.request
import shutil
import os
import random as r
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
import polars as pl  # Used for efficient processing of large rating datasets

from .base_dataloader import BaseSearchDatasetBuilder, BuildConfig
from search_recs.datasets import ensure_dataset

class AmazonSearchDataLoader(BaseSearchDatasetBuilder):
    def __init__(self, cfg: BuildConfig, dataset_name="amazon_electronics", **kwargs):
        super().__init__(cfg)
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.task = kwargs.get("mode", "normal")  # Selects 'normal' or 'hybrid' mode
        
        if cfg.random_state is not None:
            r.seed(cfg.random_state)
            np.random.seed(cfg.random_state)
        
        self.dataset_dir = ensure_dataset("amazonElectronics")
        
        self.meta_path = self.dataset_dir / "meta_Electronics.json.gz"
        self.ratings_path = self.dataset_dir / "ratings.csv"

    def _download_file(self, url: str, dest_path: Path) -> None:
        if dest_path.exists() and dest_path.stat().st_size > 10**6: return
        print(f"[AmazonLoader] Downloading: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
            shutil.copyfileobj(resp, f)

    def load_search_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not self.meta_path.exists():
            self.dataset_dir = ensure_dataset("amazonElectronics")
            self.meta_path = self.dataset_dir / "meta_Electronics.json.gz"
            self.ratings_path = self.dataset_dir / "ratings.csv"
        df_meta = self.build_pairs()

        if self.task == "hybrid":
            print("Hybrid strategy selected.")
            return self._build_hybrid_dataset(df_meta)
        
        # Otherwise, proceed with the standard search workflow
        print("Standard strategy selected.")
        return self._split_and_sample(df_meta)

    def _build_hybrid_dataset(self, df_meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Centroid Strategy: 
        - 60% Short-term Training
        - 20% Query (History)
        - 20% Ground Truth
        """
        
        # Define cache file path within the dataset directory
        cache_path = self.dataset_dir / "hybrid_test_cache.joblib"

        # --- Cache Verification ---
        if cache_path.exists():
            print(f"[AmazonLoader] Cache found at {cache_path}. Loading hybrid dataset...")
            try:
                import joblib
                df_test = joblib.load(cache_path)
                
                # --- Random Variation: 10% Dropout ---
                # Applied when no fixed seed is enforced or unconditionally enabled here
                if True:
                    print("[AmazonLoader] Applying 10% 'History Dropout'...")
                    
                    def drop_percentage_items(history_str, drop_rate=0.18):
                        if not isinstance(history_str, str): return history_str
                        items = history_str.split(',')
                        n = len(items)
                        
                        # Skip removal if the history contains too few items
                        if n <= 1: return history_str
                        
                        # Calculate the number of items to drop (targeting ~10%)
                        # The max(1, ...) ensures at least one item is removed if possible
                        n_to_drop = max(1, int(n * drop_rate))
                        
                        # Safety check: Ensure at least one item remains in the history
                        if n_to_drop >= n:
                            n_to_drop = n - 1
                            
                        # Perform random removal
                        if n_to_drop > 0:
                            # Select indices to remove
                            indices_to_drop = set(r.sample(range(n), n_to_drop))
                            # Reconstruct the list preserving original order
                            items = [item for i, item in enumerate(items) if i not in indices_to_drop]
                            
                        return ",".join(items)

                    # Apply the dropout function
                    df_test['search_query'] = df_test['search_query'].apply(drop_percentage_items)
                    
                    # Optional: User subsampling can be applied alongside dropout
                    # df_test = df_test.sample(frac=0.9).reset_index(drop=True)
                
                print(f"[AmazonLoader] Cache loaded successfully ({len(df_test)} users).")
                return df_meta, pd.DataFrame(), df_test
                
            except Exception as e:
                print(f"[AmazonLoader] Error loading cache: {e}. Regenerating...")

        if not self.ratings_path.exists():
            self.dataset_dir = ensure_dataset("amazonElectronics")
            self.ratings_path = self.dataset_dir / "ratings.csv"
        print("[AmazonLoader] Building Hybrid Dataset (Centroid)... This may take time on the first run.")

        # 1. Load interactions using Polars for performance efficiency
        df_int = pl.read_csv(
            self.ratings_path, 
            has_header=False, 
            new_columns=["item", "user", "rating", "timestamp"],
            schema_overrides={
                "item": pl.String,
                "user": pl.String,
                "rating": pl.Float32,
                "timestamp": pl.Int64
            }
        )
        
        # 2. Density filter (Minimum 5 interactions)
        valid_users = df_int.group_by("user").count().filter(pl.col("count") >= 5).select("user")
        df_int = df_int.join(valid_users, on="user").sort(["user", "timestamp"])

        # 3. Processing per user
        hybrid_test_rows = []
        all_product_ids = df_meta["document_id"].unique().tolist()
        
        # Convert to pandas for the split loop
        df_pandas = df_int.to_pandas()
        for user, group in tqdm(df_pandas.groupby("user"), desc="Splitting Users"):
            n = len(group)
            if n < 5: continue
            
            idx60 = int(n * 0.6)
            idx80 = int(n * 0.8)
            
            # BiEncoder Training (60%)
            # Historical Query (20%) -> Converted to Average Vector
            query_items = group.iloc[idx60:idx80]["item"].tolist()
            # Ground Truth (20%) -> Future items
            gt_items = group.iloc[idx80:]["item"].tolist()
            
            if not query_items or not gt_items: continue

            # Negative Sampling (200 samples)
            pos_set = set(group["item"].tolist())
            neg_candidates = [i for i in all_product_ids if i not in pos_set]
            
            # Ensure sample size does not exceed available candidates
            n_negs = min(200, len(neg_candidates))
            neg_sample = r.sample(neg_candidates, n_negs)
            
            candidates = gt_items[:20] + neg_sample
            r.shuffle(candidates)

            hybrid_test_rows.append({
                "search_query": ",".join(query_items), # BiEncoder splits by comma
                "document_id": gt_items[0],
                "candidate_ids": json.dumps(candidates),
                "ground_truth_ids": json.dumps(gt_items[:20])
            })

        df_test = pd.DataFrame(hybrid_test_rows)

        # --- Automatic Cache Saving ---
        if not df_test.empty:
            print(f"[AmazonLoader] Saving processed dataset to {cache_path}...")
            import joblib
            joblib.dump(df_test, cache_path)

        # Training data remains based on search metadata to allow the BiEncoder to learn product semantics
        return df_meta, pd.DataFrame(), df_test

    def load_raw(self) -> None:
        """Ensures the file exists."""
        if not self.meta_path.exists():
            self.dataset_dir = ensure_dataset("amazonElectronics")
            self.meta_path = self.dataset_dir / "meta_Electronics.json.gz"

    def _clean_description(self, desc_raw) -> str:
        if isinstance(desc_raw, list):
            return " ".join([str(d).strip() for d in desc_raw if d]).strip()
        if isinstance(desc_raw, str):
            return desc_raw.strip()
        return ""

    def _extract_categories(self, cat_raw) -> List[str]:
        if not isinstance(cat_raw, list):
            return []
        
        unique_cats = set()
        for cat_chain in cat_raw:
            if isinstance(cat_chain, list):
                for c in cat_chain:
                    # Filter out generic or short categories
                    if c and len(c) > 2 and c.lower() not in ["electronics"]:
                        unique_cats.add(c)
            elif isinstance(cat_chain, str):
                unique_cats.add(cat_chain)
        return list(unique_cats)

    def build_pairs(self) -> pd.DataFrame:
        """
        Reads Metadata file line by line.
        Generates (Query=Category, Doc=Product) pairs.
        Finally, displays statistics confirming the 1-to-N Query-Document relationship.
        """
        print("[AmazonLoader] Processing metadata (Streaming)...")
        
        records = []
        # Define a limit for testing. Set to None to process all (millions of products).
        MAX_PRODUCTS_TO_PROCESS = None 
        
        count = 0
        printed_example = False 

        try:
            with gzip.open(self.meta_path, 'rt', encoding='utf-8') as f:
                for line in tqdm(f, desc="Generating Query-Doc Pairs"):
                    try:
                        row = json.loads(line)
                        
                        # DEBUG PRINT OF RAW DATA
                        if not printed_example:
                            print("\n" + "="*60)
                            print("[DEBUG] EXAMPLE OF RAW METADATA OBJECT:")
                            print(json.dumps(row, indent=4))
                            print("="*60 + "\n")
                            printed_example = True

                        asin = row.get("asin")
                        title = row.get("title", "").strip()
                        desc = self._clean_description(row.get("description", []))
                        categories = self._extract_categories(row.get("category", []))
                        
                        if not asin or not title or not categories:
                            continue
                            
                        if desc:
                            document_text = f"{title}. {desc}"
                        else:
                            document_text = title
                        
                        # --- Core Logic ---
                        # Ensures that if a product belongs to the "Camera" category,
                        # it is added to the document list for that query.
                        for cat in categories:
                            records.append({
                                "search_query": cat,
                                "document": document_text,
                                "document_id": asin,
                                "category": "Metadata-Category"
                            })
                        
                        count += 1
                        if MAX_PRODUCTS_TO_PROCESS and count >= MAX_PRODUCTS_TO_PROCESS:
                            print(f"[AmazonLoader] Limit of {MAX_PRODUCTS_TO_PROCESS} products reached.")
                            break
                            
                    except (ValueError, json.JSONDecodeError):
                        continue
                        
        except Exception as e:
            print(f"[AmazonLoader] Read error: {e}")
            raise

        df = pd.DataFrame(records)
        print(f"[AmazonLoader] Total pairs generated: {len(df)}")
        
        if not df.empty:
            # 1. Remove exact duplicates first
            df = df.drop_duplicates(subset=["search_query", "document_id"])
            
            # 2. Cut-off logic: Count documents per query
            query_counts = df['search_query'].value_counts()
            
            # 3. Filter: Keep only queries with count >= 10
            min_docs = 10
            valid_queries = query_counts[query_counts >= min_docs].index
            
            df_filtered = df[df['search_query'].isin(valid_queries)].copy()

            # --- INSERT HERE ---
            unique_queries = df_filtered['search_query'].nunique()
            print(f"[AmazonLoader] Queries before cut-off: {len(query_counts)}")
            print(f"[AmazonLoader] Queries after cut-off (min {min_docs} docs): {len(valid_queries)}")
            print(f"[AmazonLoader] Final total pairs: {len(df_filtered)}")
            print(f"[AmazonLoader] Total UNIQUE QUERIES (Categories): {unique_queries}")
            # --------------------
            
            return df_filtered

        return df

    def _split_and_sample(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if df.empty:
            return df, df, df

        rs = self.cfg.random_state if self.cfg.random_state else 42
        import json
        import random as r

        # 1. Initial Split between Train and the rest (Val/Test)
        train_val, test_df = train_test_split(
            df, test_size=self.cfg.test_size, random_state=rs, shuffle=True
        )

        # 2. Function to transform Normal Search into Search with Negative Sampling
        def transform_to_sampled_search(target_df, full_df):
            # CHANGE HERE: Use "document_id"
            all_item_ids = full_df["document_id"].unique().tolist()
            new_rows = []
            
            # CHANGE HERE: Use "search_query"
            grouped = target_df.groupby("search_query")

            for query_text, group in tqdm(grouped, desc="Sampling Negatives for Eval"):
                # CHANGE HERE: Use "document_id" and "document"
                gt_ids = group["document_id"].unique().tolist()
                
                pos_sample = gt_ids[:50]
                gt_set_full = set(gt_ids) 
                neg_candidates = [idx for idx in all_item_ids if idx not in gt_set_full]
                
                neg_sample = r.sample(neg_candidates, min(1000, len(neg_candidates)))
                candidates = pos_sample + neg_sample
                r.shuffle(candidates)
                
                new_rows.append({
                    "search_query": query_text,
                    "document": group["document"].iloc[0], 
                    "document_id": group["document_id"].iloc[0],
                    "candidate_ids": json.dumps(candidates),
                    "ground_truth_ids": json.dumps(pos_sample)
                })
            
            return pd.DataFrame(new_rows)

        print(f"[AmazonLoader] Applying search sampling to Test Set...")
        test_df = transform_to_sampled_search(test_df, df)

        # 3. Finalize Validation split
        val_rel = self.cfg.val_size / (1.0 - self.cfg.test_size) if (1.0 - self.cfg.test_size) > 0 else 0.1
        train_df, val_df = train_test_split(
            train_val, test_size=min(0.9, val_rel), random_state=rs, shuffle=True
        )

        # Apply head if configured
        if self.cfg.head_train:
            train_df = train_df.sample(n=min(self.cfg.head_train, len(train_df)), random_state=rs)
        
        return (
            train_df.reset_index(drop=True), 
            val_df.reset_index(drop=True), 
            test_df.reset_index(drop=True)
        )

# Entry Point
def get_loader(cfg: BuildConfig, **kwargs):
    dataset_name = kwargs.pop("path", kwargs.pop("dataset_name", "amazonElectronics"))
    loader = AmazonSearchDataLoader(cfg, dataset_name=dataset_name, **kwargs)
    return loader.load_search_data()
