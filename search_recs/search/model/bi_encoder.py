import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import json
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from datasets import Dataset as HfDataset
import torch
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from search_recs.recs.model import BaseRecsModel
from search_recs.search.bi_encoders.trainer.sentencetransformers import (
    SentenceTransformerMnrlTrainer,
)

class BiEncoderModel(BaseRecsModel):
    """
    Bi-Encoder model
    """

    def __init__(self, model_config: dict):
        super().__init__(model_config)

        print(f"\n[BiEncoderModel] Config keys: {list(model_config.keys())}")
        
        params = model_config.get("parameters", {})
        
        # Backbone Configuration
        self.pretrained_name: str = params.get("pretrained_name", "all-MiniLM-L6-v2")
        self.token_max_length: int = int(params.get("token_max_length", 128))
        
        # Training Hyperparameters
        self.batch_size: int = int(params.get("batch_size", 16))
        self.learning_rate: float = float(params.get("learning_rate", 2e-5))
        self.epochs: int = int(params.get("epochs", 1))
        
        # Training Hyperparameters (Advanced) -- CORREÇÃO AQUI
        self.num_warmup_factor: float = float(params.get("num_warmup_factor", 0.1))
        self.early_stopping_patience: int = int(params.get("early_stopping_patience", 0))
        self.early_stopping_threshold: float = float(params.get("early_stopping_threshold", 0.01))
        
        self.model_dir: str = params.get("model_dir", "./output/bi_encoder")
        self.logging_steps: int = int(params.get("logging_steps", 100))
        self.save_steps: int = int(params.get("save_steps", 1000))
        self.evaluation_steps: int = int(params.get("evaluation_steps", 1000))
        
        # Indexing Configuration
        self.annoy_n_trees: int = int(params.get("annoy_n_trees", 10))

        # Data Column Mapping
        self.qcol: str = params.get("query_col", "search_query")
        self.dcol: str = params.get("doc_col", "document")
        self.did_col: str = params.get("doc_id_col", "document_id")

        # Strategy Selection
        self.task: str = params.get("task", "normal")
        print(f"[BiEncoderModel] Task initialized: '{self.task}'")

        # Model State
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[SentenceTransformer] = None

        self._train_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._val_df: Optional[pd.DataFrame] = None

        self.annoy_index: Optional[AnnoyIndex] = None
        self.corpus_docs: Optional[List[str]] = None
        self.doc_ref_to_annoy_idx: Dict[str, int] = {}

    # --------------------- HELPERS ---------------------

    def _ensure_model(self) -> None:
        if self.model is None:
            print(f"[BiEncoderModel] Loading HF model: {self.pretrained_name}")
            self.model = SentenceTransformer(
                model_name_or_path=self.pretrained_name,
                device=self.device,
            )
            self.model.max_seq_length = self.token_max_length

    def preprocess(self, train_data: pd.DataFrame, **kwargs):
        print(f"[BiEncoderModel] Preprocess started. Train shape: {train_data.shape}")
        self._train_df = train_data
        self._test_df = kwargs.get("test_data")
        self._val_df = kwargs.get("val_data")
        self._ensure_model()

    def _convert_to_hf_dataset(self, df: Optional[pd.DataFrame]) -> Optional[HfDataset]:
        if df is None or df.empty or self.qcol not in df.columns: 
            return None
            
        df_dict = {
            "anchor": "query: " + df[self.qcol].astype(str),
            "positive": "passage: " + df[self.dcol].astype(str),
            "label": [1.0] * len(df)
        }
        return HfDataset.from_dict(df_dict)

    def fit(self):
        if self._train_df is None or self.model is None:
            raise RuntimeError("preprocess() must be called before fit().")

        print("[BiEncoderModel] Starting training...")
        train_hf = self._convert_to_hf_dataset(self._train_df)
        val_hf = self._convert_to_hf_dataset(self._val_df)

        if self.task in ("hybrid", "centroid-vector", "centroid_vector"):
            print("[BiEncoderModel] Hybrid mode detected. Skipping fine-tuning (using pre-trained weights).")
            return

        trainer = SentenceTransformerMnrlTrainer(
            model_wrapper=self.model,
            train_dataset=train_hf,
            val_dataset=val_hf,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_warmup_factor=self.num_warmup_factor,
            model_dir=self.model_dir,
            epochs=self.epochs,
            logging_steps=self.logging_steps,
            evaluation_steps=self.evaluation_steps,
            save_steps=self.save_steps,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_threshold=self.early_stopping_threshold,
        )
        trainer.train()
        print("[BiEncoderModel] Training finished.")

    # --------------------- ENCODING & INDEXING ---------------------

    def _encode_texts(self, texts: List[str], text_type: str) -> np.ndarray:
        prefix = "query: " if text_type == "query" else "passage: "
        prefixed_texts = [f"{prefix}{str(t)}" for t in texts]
        
        print(f"[BiEncoderModel] Encoding {len(prefixed_texts)} texts ({text_type})...")
        embs = self.model.encode(
            prefixed_texts,
            batch_size=64,
            show_progress_bar=True,
            device=self.device,
            normalize_embeddings=False,
            convert_to_tensor=False, 
            convert_to_numpy=True
        )
        return embs.astype("float32")
    
    def _encode_history_mean(self, history_str: str) -> np.ndarray:
        """
        Computes the centroid of vectors for items in the user history.
        Supports both alphanumeric IDs and integers.
        """
        if not history_str or not isinstance(history_str, str):
            return np.zeros(self.annoy_index.f, dtype="float32")
        
        item_ids = [x.strip() for x in str(history_str).split(",") if x.strip()]
        vectors = []
        
        for item_id in item_ids:
            annoy_idx = self.doc_ref_to_annoy_idx.get(str(item_id))
            
            if annoy_idx is not None:
                vec = self.annoy_index.get_item_vector(annoy_idx)
                vec_np = np.array(vec, dtype="float32")
                
                norm = np.linalg.norm(vec_np)
                if norm > 1e-12:
                    vec_np = vec_np / norm
                    
                vectors.append(vec_np)
        
        if not vectors:
            return np.zeros(self.annoy_index.f, dtype="float32")
            
        return np.mean(vectors, axis=0).astype("float32")

    def build_index(self, force_rebuild: bool = False) -> None:
        if self.annoy_index is not None and not force_rebuild: return
        self._ensure_model()
        print("[BiEncoderModel] Building Annoy index...")

        unique_docs_map = {}
        def collect(df):
            if df is not None and self.dcol in df.columns:
                has_id = self.did_col in df.columns
                for _, row in df.iterrows():
                    dtext = str(row[self.dcol])
                    key = str(row[self.did_col]) if has_id else dtext
                    if key not in unique_docs_map: unique_docs_map[key] = dtext

        collect(self._train_df)
        collect(self._test_df)
        collect(self._val_df)
        
        if not unique_docs_map: raise ValueError("No documents found for indexing.")

        doc_keys = list(unique_docs_map.keys())
        doc_texts = list(unique_docs_map.values())
        self.corpus_docs = doc_texts
        
        print(f"[BiEncoderModel] Encoding {len(doc_texts)} unique documents...")
        doc_embs_np = self._encode_texts(doc_texts, text_type="document")

        dim = int(doc_embs_np.shape[1])
        index = AnnoyIndex(dim, "angular")
        self.doc_ref_to_annoy_idx = {}
        
        for i, (key, vec) in enumerate(zip(doc_keys, doc_embs_np)):
            index.add_item(i, vec)
            self.doc_ref_to_annoy_idx[str(key)] = i
            
        index.build(self.annoy_n_trees)
        self.annoy_index = index
        print(f"[BiEncoderModel] Index built. Dim={dim}, Docs={len(doc_keys)}.")

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype("float32")
        b = b.astype("float32")
        na = np.linalg.norm(a) + 1e-12
        nb = np.linalg.norm(b) + 1e-12
        return float(np.dot(a, b) / (na * nb))

    def _parse_json_list(self, x) -> List[str]:
        if x is None: return []
        if isinstance(x, list): return [str(v) for v in x]
        s = str(x).strip()
        if not s: return []
        try:
            obj = json.loads(s)
            if isinstance(obj, list): return [str(v) for v in obj]
        except Exception:
            pass
        return []

    # -------------------------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------------------------

    def prediction(self, test_data: pd.DataFrame) -> Tuple[List[List[int]], List[List[float]]]:
        print("\n[BiEncoderModel] Starting Prediction Phase...")

        if not isinstance(test_data, pd.DataFrame):
            raise ValueError("test_data must be a pandas DataFrame.")

        self._ensure_model()
        self.build_index(force_rebuild=False)

        if self.qcol not in test_data.columns:
            raise ValueError(f"Query column '{self.qcol}' missing in test_data.")

        queries = test_data[self.qcol].astype(str).tolist()
        if not queries:
            return [], []

        # Check for candidate columns
        has_candidate_col = "candidate_ids" in test_data.columns and "ground_truth_ids" in test_data.columns

        # ---------------------------------------------------------------------
        # 1. OPTIMIZED EVALUATION (Reranking Candidates)
        # ---------------------------------------------------------------------
        if has_candidate_col:
            print(f"[BiEncoderModel] Mode: Candidate Reranking. Task: {self.task.upper()}")
            
            query_embs_np = None

            if self.task in ("hybrid", "centroid-vector", "centroid_vector"):
                vecs = []
                for q_str in queries:
                    v = self._encode_history_mean(q_str)
                    vecs.append(v)
                query_embs_np = np.asarray(vecs, dtype="float32")

            else:
                query_embs_np = self._encode_texts(queries, text_type="query")

            all_y_true, all_y_pred = [], []

            for i, q_vec in enumerate(query_embs_np):
                cand_ids = self._parse_json_list(test_data.iloc[i]["candidate_ids"])
                gt_ids = set(self._parse_json_list(test_data.iloc[i]["ground_truth_ids"]))

                y_pred = []
                y_true = []

                for cid in cand_ids:
                    annoy_idx = self.doc_ref_to_annoy_idx.get(str(cid))
                    if annoy_idx is None:
                        score = -10.0
                    else:
                        d_vec = np.asarray(self.annoy_index.get_item_vector(annoy_idx), dtype="float32")
                        score = self._cosine_sim(q_vec, d_vec)
                    
                    y_pred.append(float(score))
                    y_true.append(1 if str(cid) in gt_ids else 0)

                all_y_true.append(y_true)
                all_y_pred.append(y_pred)

            return all_y_true, all_y_pred

        # ---------------------------------------------------------------------
        # 2. FULL CORPUS RETRIEVAL (Legacy Mode)
        # ---------------------------------------------------------------------
        print(f"[BiEncoderModel] Mode: Full Corpus Retrieval. Task: {self.task.upper()}")
        
        if self.task in ("hybrid", "centroid-vector", "centroid_vector"):
            if self.annoy_index is None: raise RuntimeError("Index not initialized.")
            vecs = [self._encode_history_mean(q) for q in queries]
            query_embs_np = np.array(vecs, dtype="float32")
        else:
            query_embs_np = self._encode_texts(queries, text_type="query")

        relevance_map = {}
        has_id_col = self.did_col in test_data.columns
        target_col = self.did_col if has_id_col else self.dcol
        
        for _, row in test_data.iterrows():
            q = row[self.qcol]
            target = str(row[target_col])
            if q not in relevance_map: relevance_map[q] = set()
            relevance_map[q].add(target)

        corpus_size = self.annoy_index.get_n_items()
        annoy_idx_to_key = {v: k for k, v in self.doc_ref_to_annoy_idx.items()}

        all_y_true, all_y_pred = [], []

        for q_text, q_vec in zip(queries, query_embs_np):
            ids, distances = self.annoy_index.get_nns_by_vector(q_vec, corpus_size, include_distances=True)

            y_pred = [0.0] * corpus_size
            for idx_annoy, dist in zip(ids, distances):
                y_pred[idx_annoy] = float(1.0 - (dist ** 2 / 2.0))
            all_y_pred.append(y_pred)

            y_true = [0] * corpus_size
            targets = relevance_map.get(q_text, set())
            for idx_annoy in range(corpus_size):
                key = annoy_idx_to_key.get(idx_annoy)
                if key in targets: y_true[idx_annoy] = 1
            all_y_true.append(y_true)

        return all_y_true, all_y_pred