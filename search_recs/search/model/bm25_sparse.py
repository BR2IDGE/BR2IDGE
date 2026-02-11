import json
import nltk
import pathlib
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Set, Any
from retriv import SparseRetriever
from search_recs.recs.model import BaseRecsModel
from search_recs.search.dataloader.movielens import load_genome_tag_map
from tqdm import tqdm

class BM25Model(BaseRecsModel):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        params = self.model_params or {}

        # 1. Identity and Algorithm Configuration
        self.index_name: str = params.get("index_name", "bm25_index")
        self.retriever_type: str = params.get("model", "bm25")  # bm25, tf-idf, etc.
        self.hyperparams: Dict = params.get("hyperparams", {"b": 0.75, "k1": 1.5})

        # 2. Column Mapping
        self.qcol: str = params.get("query_col", "search_query")
        self.dcol: str = params.get("doc_col", "document")
        self.did_col: str = params.get("doc_id_col", "document_id")

        # 3. Hybrid Strategy (Tag Fusion)
        self.strategy: str = params.get("task", "normal")  # 'normal' or 'hybrid'
        self.dataset_path: str = params.get("dataset_path", "./data/ml-25m")
        self.movie_tag_map: Optional[Dict] = None

        # 4. Internal State
        self._collection: Optional[List[Dict]] = None
        self._sr: Optional[SparseRetriever] = None
        self._doc_text_to_retriv_id: Optional[Dict[str, str]] = None
        self.corpus_size: int = 0

        print(f"[BM25Model] Init | Mode: {self.strategy.upper()} | Index: {self.index_name}")

    # -------------------------------------------------------------------------
    # HYBRID RECOMMENDATION LOGIC (TAG FUSION)
    # -------------------------------------------------------------------------

    def _ensure_external_tag_data(self):
        """
        Automatically loads external tag metadata based on the dataset path.
        Handles specific logic for MovieLens and Last.fm.
        """
        if self.movie_tag_map is not None:
            return

        path = pathlib.Path(self.dataset_path)
        
        # --- CASE 1: MOVIELENS (Genome Tags) ---
        if "ml-25m" in str(path).lower():
            print(f"[BM25Model] Loading internal Genome Tags (MovieLens)...")
            self.movie_tag_map = load_genome_tag_map(str(path), min_relevance=0.9)

        # --- CASE 2: LAST.FM (2K Tags associated with 360K base) ---
        elif "lastfm" in str(path).lower():
            print(f"[BM25Model] Loading internal Last.fm Metadata...")
            try:
                # 1. Load files downloaded by the loader
                tags_df = pd.read_csv(path / "tags.dat", sep="\t", encoding="latin-1", on_bad_lines='skip')
                tagging_df = pd.read_csv(path / "user_taggedartists.dat", sep="\t", encoding="latin-1", on_bad_lines='skip')
                artists_2k = pd.read_csv(path / "artists.dat", sep="\t", encoding="latin-1", on_bad_lines='skip')
                
                # 2. Merge to get Artist_Name -> Tags
                df_merged = tagging_df.merge(tags_df, on="tagID").merge(artists_2k[["id", "name"]], left_on="artistID", right_on="id")
                
                # Group by artist name
                name_to_tags = df_merged.groupby(df_merged["name"].str.lower().str.strip())["tagValue"].apply(
                    lambda x: {str(t): 1.0 for t in x.unique()[:15]}
                ).to_dict()

                # 3. Map to 360K numerical IDs (using the order of the indexed collection)
                # IMPORTANT: This maps the numerical ID (0, 1, 2...) to tags via Artist Name
                self.movie_tag_map = {}
                if self._collection:
                    for doc in self._collection:
                        name_clean = str(doc["text"]).lower().strip()
                        if name_clean in name_to_tags:
                            self.movie_tag_map[str(doc["id"])] = name_to_tags[name_clean]
                
                print(f"[BM25Model] Tags mapped for {len(self.movie_tag_map)} artists.")
            except Exception as e:
                print(f"[BM25Model] Error processing Last.fm tags: {e}")
                self.movie_tag_map = {}
        
        else:
            print("[BM25Model] No specific external tag loader found for this dataset path.")

    def _generate_tag_query(self, history_str: str) -> str:
        """
        Converts a history of IDs (e.g., ASINs) into a string of tags (Categories).
        Ex: "B001, B002" -> "Camera, Lens"
        """
        self._ensure_external_tag_data()

        if not self.movie_tag_map:
            # Fallback: If no map exists, use the text itself (replacing commas)
            return str(history_str).replace(",", " ")

        # Remove spaces and split by comma
        item_ids = [x.strip() for x in str(history_str).split(",") if x.strip()]
        
        tag_aggregator = {}

        for mid in item_ids:
            # Retrieve tags for this item from the map built in preprocess or loaded externally
            tags_dict = self.movie_tag_map.get(str(mid), {})
            for tag_name, relevance in tags_dict.items():
                tag_aggregator[tag_name] = tag_aggregator.get(tag_name, 0.0) + relevance

        # Retrieve top 15 most frequent tags in history
        sorted_tags = sorted(tag_aggregator.items(), key=lambda x: x[1], reverse=True)
        top_tags = [t[0] for t in sorted_tags[:15]]

        if not top_tags:
            return "" 
            
        return " ".join(top_tags)

    # -------------------------------------------------------------------------
    # PREPROCESS & FIT (INDEX CONSTRUCTION)
    # -------------------------------------------------------------------------

    def preprocess(self, train_data: pd.DataFrame, **kwargs):
        print(f"[BM25Model] Preprocess | Strategy: {self.strategy}")
        
        test_data = kwargs.get("test_data")
        has_candidate_mode = isinstance(test_data, pd.DataFrame) and ("candidate_ids" in test_data.columns)

        # --- TAG/CATEGORY MAPPING LOGIC (AMAZON / GENERIC) ---
        if self.strategy == "hybrid":
            # If the loader provided categories in the query column of the training set,
            # we build the ID -> Category map here.
            print("[BM25Model] Building Tag/Category map from Metadata...")
            self.movie_tag_map = {}
            
            # Optimization: zip is faster than iterrows
            ids = train_data[self.did_col].astype(str).tolist()
            cats = train_data[self.qcol].astype(str).tolist()
            
            for doc_id, cat in zip(ids, cats):
                if cat and str(cat).lower() != 'nan':
                    # Format {Tag: Weight}
                    self.movie_tag_map[doc_id] = {str(cat): 1.0}
            
            print(f"[BM25Model] {len(self.movie_tag_map)} items mapped with categories.")

        # --- INDEXING ---
        if has_candidate_mode:
            # Hybrid Mode: Index the product catalog
            print("[BM25Model] Indexing product catalog...")
            unique_docs = train_data[[self.did_col, self.dcol]].drop_duplicates(subset=[self.did_col])
            
            collection_df = unique_docs.rename(columns={self.dcol: "text", self.did_col: "id"})
            collection_df["id"] = collection_df["id"].astype(str)

            self._collection = collection_df[["id", "text"]].to_dict("records")
            self.corpus_size = len(self._collection)
        else:
            # Text Search Mode (Legacy)
            print("[BM25Model] Indexing Query-Doc pairs (Search Mode)...")
            parts = [train_data]
            if test_data is not None: parts.append(test_data)
            
            full_df = pd.concat(parts, ignore_index=True)
            unique_docs = full_df[[self.did_col, self.dcol]].drop_duplicates(subset=[self.did_col])
            collection_df = unique_docs.rename(columns={self.dcol: "text", self.did_col: "id"})
            collection_df["id"] = collection_df["id"].astype(str)

            self._collection = collection_df[["id", "text"]].to_dict("records")
            self.corpus_size = len(self._collection)
            self._doc_text_to_retriv_id = pd.Series(collection_df.id.values, index=collection_df.text).to_dict()

    def fit(self):
        """Trains the SparseRetriever (Inverted Index)."""
        if self._collection is None:
            raise RuntimeError("Execute preprocess() before fit().")

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        SparseRetriever.delete(self.index_name)
        self._sr = SparseRetriever(
            index_name=self.index_name,
            model=self.retriever_type,
            hyperparams=self.hyperparams,
            tokenizer="word",
            stemmer="english",
            stopwords="english",
            do_lowercasing=True,
            do_punctuation_removal=True,
        )
        self._sr.index(self._collection)
        print(f"[BM25Model] Fit complete. Index '{self.index_name}' ready.")

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _hits_to_map(self, hits: Any) -> Dict[str, float]:
        """
        Converts retriv output (which may vary) to a dict: {doc_id: score}.
        - If dict: OK.
        - If list: tries to extract pairs (id, score) or dicts with common keys.
        """
        if hits is None:
            return {}

        if isinstance(hits, dict):
            return {str(k): float(v) for k, v in hits.items()}

        if isinstance(hits, list):
            out = {}
            for it in hits:
                # Case 1: tuple/list (id, score)
                if isinstance(it, (tuple, list)) and len(it) >= 2:
                    out[str(it[0])] = float(it[1])
                    continue
                # Case 2: dict with common keys
                if isinstance(it, dict):
                    doc_id = it.get("id") or it.get("doc_id") or it.get("document_id")
                    score = it.get("score") or it.get("bm25") or it.get("value")
                    if doc_id is not None and score is not None:
                        out[str(doc_id)] = float(score)
            return out

        # Fallback: unknown format
        return {}

    def _parse_json_list(self, v: Any) -> List[str]:
        # Adapted for strings (ASINs/IDs)
        if isinstance(v, list): return [str(x) for x in v]
        if not v: return []
        try:
            l = json.loads(v)
            return [str(x) for x in l]
        except:
            return []

    # -------------------------------------------------------------------------
    # PREDICTION (SEARCH AND EVALUATION)
    # -------------------------------------------------------------------------

    def prediction(self, test_data: pd.DataFrame) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if self._sr is None:
            raise RuntimeError("Model not trained.")
        
        has_candidate_mode = ("candidate_ids" in test_data.columns)

        if has_candidate_mode:
            print("[BM25Model] Starting Optimized Hybrid Prediction (Global Index)...")
            
            # 1. Prepare Queries
            # Use the correct column (qcol) and convert to string
            raw_histories = test_data[self.qcol].astype(str).tolist()
            
            if self.strategy == "hybrid":
                # Convert IDs -> Categories (Tags) using the map generated in preprocess
                final_queries = [self._generate_tag_query(h) for h in raw_histories]
                # Debug to ensure transformation worked
                if final_queries:
                    print(f"[Debug] Example Transformation: '{raw_histories[0]}' -> '{final_queries[0]}'")
            else:
                final_queries = raw_histories

            # 2. Prepare Candidates and Ground Truth
            # Removed int() casting to support alphanumeric ASINs
            cand_lists = [self._parse_json_list(row["candidate_ids"]) for _, row in test_data.iterrows()]
            gt_sets = [set(self._parse_json_list(row["ground_truth_ids"])) for _, row in test_data.iterrows()]
            
            all_y_true, all_y_pred = [], []
            
            # Increased batch_size as search is lightweight (no re-indexing)
            batch_size = 1000 

            for bstart in tqdm(range(0, len(final_queries), batch_size), desc="BM25 Search"):
                bend = min(bstart + batch_size, len(final_queries))
                batch_idx = list(range(bstart, bend))

                # --- CRITICAL OPTIMIZATION ---
                # We do NOT recreate temporary indices. We use the Global self._sr trained in fit().
                
                # Prepare query batch in Retriv format
                batch_queries_tmp = [{"id": str(qi), "text": final_queries[qi]} for qi in batch_idx]
                
                # Global search: returns Top 1000 most relevant documents from entire catalog
                batch_res = self._sr.bsearch(batch_queries_tmp, cutoff=1000)

                for qi in batch_idx:
                    # Map of {doc_id: score} returned by global search
                    hits_map = self._hits_to_map(batch_res.get(str(qi), {}))
                    
                    cands = cand_lists[qi]
                    
                    # Map scores to the specific candidate list for this user.
                    # If a candidate did not appear in the global search (score 0), it is irrelevant.
                    y_pred = np.array([float(hits_map.get(str(cid), 0.0)) for cid in cands], dtype=np.float32)
                    y_true = np.array([1.0 if cid in gt_sets[qi] else 0.0 for cid in cands], dtype=np.float32)
                    
                    all_y_pred.append(y_pred)
                    all_y_true.append(y_true)

            return all_y_true, all_y_pred

        # -------------------------------------------------------------------------
        # NORMAL / LEGACY MODE (Maintained for backward compatibility)
        # -------------------------------------------------------------------------
        else:
            print("[BM25Model] Standard Search Mode (Full Corpus)...")
            raw_queries = test_data[self.qcol].tolist()

            if self.strategy == "hybrid":
                print(f"[BM25Model] Applying Tag Fusion on {len(raw_queries)} queries...")
                final_queries = [self._generate_tag_query(q) for q in raw_queries]
            else:
                final_queries = raw_queries

            max_k = 50
            batch_size = 2000
            all_results = {}

            print(f"[BM25Model] Searching against {self.corpus_size} docs in mini-batches...")
            for i in range(0, len(final_queries), batch_size):
                batch = [{"id": str(j), "text": txt} for j, txt in enumerate(final_queries[i:i + batch_size], start=i)]
                batch_res = self._sr.bsearch(batch, cutoff=max_k)
                all_results.update(batch_res)

            # Reverse mapping for dense evaluation
            corpus_ids_map = {str(doc["id"]): idx for idx, doc in enumerate(self._collection)}

            q_to_targets = {}
            for _, row in test_data.iterrows():
                q_orig = row[self.qcol]
                target_text = str(row[self.dcol])
                internal_id = self._doc_text_to_retriv_id.get(target_text)
                if internal_id:
                    if q_orig not in q_to_targets:
                        q_to_targets[q_orig] = set()
                    q_to_targets[q_orig].add(str(internal_id))

            all_y_pred = []
            all_y_true = []

            print("[BM25Model] Generating evaluation matrices...")
            for i, q_orig in enumerate(raw_queries):
                y_pred = np.zeros(self.corpus_size, dtype=np.float32)
                y_true = np.zeros(self.corpus_size, dtype=np.float32)

                hits = all_results.get(str(i), {})
                if isinstance(hits, dict):
                    for doc_id, score in hits.items():
                        idx = corpus_ids_map.get(str(doc_id))
                        if idx is not None:
                            y_pred[idx] = score

                targets = q_to_targets.get(q_orig, set())
                for t_id in targets:
                    idx = corpus_ids_map.get(str(t_id))
                    if idx is not None:
                        y_true[idx] = 1.0

                all_y_pred.append(y_pred)
                all_y_true.append(y_true)

            return all_y_true, all_y_pred