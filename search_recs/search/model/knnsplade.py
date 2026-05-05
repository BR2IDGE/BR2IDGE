from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz
from tqdm import tqdm
import os
 
from search_recs.recs.model import BaseRecsModel
 
 
class SpladeKnnModel(BaseRecsModel):

    def __init__(self, model_config: dict):
        super().__init__(model_config)
 
        print(f"\n[SpladeKnnModel] Config keys: {list(model_config.keys())}")
 
        params = model_config.get("parameters", {})
 
        self.pretrained_name: str = params.get("pretrained_name", "naver/splade-v3")
        self.token_max_length: int = int(params.get("token_max_length", 512))
        self.auth_token: str = params.get("auth_token", "")
 
        self.n_neighbors: int = int(params.get("n_neighbors", 10))
 
        self.model_dir: str = params.get("model_dir", "./output/splade_knn")
        self.index_path: str = os.path.join(self.model_dir, "splade_matrix.npz")
 
        self.qcol: str = params.get("query_col", "search_query")
        self.dcol: str = params.get("doc_col", "document")
        self.did_col: str = params.get("doc_id_col", "document_id")
 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[AutoModelForMaskedLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.vocab_size: Optional[int] = None
 
        self._train_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._val_df: Optional[pd.DataFrame] = None
 
        self.knn: Optional[NearestNeighbors] = None
        self.corpus_matrix: Optional[csr_matrix] = None
        self.corpus_docs: Optional[List[str]] = None
        self.doc_ref_to_idx: Dict[str, int] = {}


    def _ensure_model(self) -> None:
        if self.model is None:
            print(f"[SpladeKnnModel] Loading SPLADE model: {self.pretrained_name}")
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.pretrained_name,
                use_auth_token=self.auth_token or None,
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_name,
                use_auth_token=self.auth_token or None,
            )
            self.vocab_size = len(self.tokenizer.get_vocab())
            print(f"[SpladeKnnModel] Model loaded. Vocab size: {self.vocab_size} | Device: {self.device}")
 
    def _encode_text(self, text: str) -> csr_matrix:
        with torch.no_grad():
            tokens = self.tokenizer(
                text,
                max_length=self.token_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
 
            output = self.model(**tokens)
 
            vec = torch.max(
                torch.log(1 + torch.relu(output.logits))
                * tokens.attention_mask.unsqueeze(-1),
                dim=1,
            )[0].squeeze()
 
            cols = vec.nonzero().squeeze().cpu().tolist()
            weights = vec[cols].cpu().tolist()
 
        if isinstance(cols, int):
            cols, weights = [cols], [weights]
 
        if not cols:
            return csr_matrix((1, self.vocab_size), dtype=np.float32)
 
        return csr_matrix(
            (weights, ([0] * len(cols), cols)),
            shape=(1, self.vocab_size),
            dtype=np.float32,
        )
 
    def _encode_texts_batch(self, texts: List[str], desc: str = "Encoding") -> csr_matrix:
        rows = []
        for text in tqdm(texts, desc=f"[SpladeKnnModel] {desc}"):
            rows.append(self._encode_text(text))
        return vstack(rows)
 
    def preprocess(self, train_data: pd.DataFrame, **kwargs):
        print(f"[SpladeKnnModel] Preprocess started. Train shape: {train_data.shape}")
        self._train_df = train_data
        self._test_df = kwargs.get("test_data")
        self._val_df = kwargs.get("val_data")
        self._ensure_model()
 
    def fit(self):
        if self._train_df is None or self.model is None:
            raise RuntimeError("preprocess() must be called before fit().")
 
        print("[SpladeKnnModel] SPLADE uses pre-trained weights. No fine-tuning step.")
 

    def build_index(self, force_rebuild: bool = False) -> None:
        if self.knn is not None and not force_rebuild:
            return
 
        self._ensure_model()
        print("[SpladeKnnModel] Building SPLADE+KNN index...")
 
        unique_docs_map: Dict[str, str] = {}
 
        def collect(df: Optional[pd.DataFrame]):
            if df is not None and self.dcol in df.columns:
                has_id = self.did_col in df.columns
                for _, row in df.iterrows():
                    dtext = str(row[self.dcol])
                    key = str(row[self.did_col]) if has_id else dtext
                    if key not in unique_docs_map:
                        unique_docs_map[key] = dtext
 
        collect(self._train_df)
        collect(self._test_df)
        collect(self._val_df)
 
        if not unique_docs_map:
            raise ValueError("[SpladeKnnModel] No documents found for indexing.")
 
        doc_keys = list(unique_docs_map.keys())
        doc_texts = list(unique_docs_map.values())
        self.corpus_docs = doc_texts
        self.doc_ref_to_idx = {str(k): i for i, k in enumerate(doc_keys)}
 
        if not force_rebuild and os.path.exists(self.index_path):
            print(f"[SpladeKnnModel] Loading cached matrix from: {self.index_path}")
            self.corpus_matrix = load_npz(self.index_path)
        else:
            print(f"[SpladeKnnModel] Encoding {len(doc_texts)} unique documents...")
            self.corpus_matrix = self._encode_texts_batch(doc_texts, desc="Indexing documents")
 
            os.makedirs(self.model_dir, exist_ok=True)
            save_npz(self.index_path, self.corpus_matrix)
            print(f"[SpladeKnnModel] Matrix saved to: {self.index_path}")
 
        print(f"[SpladeKnnModel] Fitting KNN (n_neighbors={self.n_neighbors}, metric=cosine)...")
        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        self.knn.fit(self.corpus_matrix)
 
        print(f"[SpladeKnnModel] Index built. Docs={len(doc_texts)} | Shape={self.corpus_matrix.shape}")
 
 
    def prediction(self, test_data: pd.DataFrame) -> Tuple[List[List[int]], List[List[float]]]:
        print("\n[SpladeKnnModel] Starting Prediction Phase...")
 
        if not isinstance(test_data, pd.DataFrame):
            raise ValueError("test_data must be a pandas DataFrame.")
 
        self._ensure_model()
        self.build_index(force_rebuild=False)
 
        if self.qcol not in test_data.columns:
            raise ValueError(f"[SpladeKnnModel] Query column '{self.qcol}' missing in test_data.")
 
        queries = test_data[self.qcol].astype(str).tolist()
        if not queries:
            return [], []
 
        has_candidate_col = (
            "candidate_ids" in test_data.columns
            and "ground_truth_ids" in test_data.columns
        )
 
        if has_candidate_col:
            print("[SpladeKnnModel] Mode: Candidate Reranking")
 
            all_y_true, all_y_pred = [], []
 
            for i, query in enumerate(tqdm(queries, desc="[SpladeKnnModel] Reranking")):
                q_vec = self._encode_text(query)
 
                cand_ids = self._parse_json_list(test_data.iloc[i]["candidate_ids"])
                gt_ids = set(self._parse_json_list(test_data.iloc[i]["ground_truth_ids"]))
 
                y_pred = []
                y_true = []
 
                for cid in cand_ids:
                    idx = self.doc_ref_to_idx.get(str(cid))
                    if idx is None:
                        score = -10.0
                    else:
                        d_vec = self.corpus_matrix[idx]
                        score = self._cosine_sim_sparse(q_vec, d_vec)
 
                    y_pred.append(float(score))
                    y_true.append(1 if str(cid) in gt_ids else 0)
 
                all_y_true.append(y_true)
                all_y_pred.append(y_pred)
 
            return all_y_true, all_y_pred
 
        print("[SpladeKnnModel] Mode: Full Corpus Retrieval")
 
        has_id_col = self.did_col in test_data.columns
        target_col = self.did_col if has_id_col else self.dcol
 
        relevance_map: Dict[str, set] = {}
        for _, row in test_data.iterrows():
            q = row[self.qcol]
            target = str(row[target_col])
            if q not in relevance_map:
                relevance_map[q] = set()
            relevance_map[q].add(target)
 
        corpus_size = self.corpus_matrix.shape[0]
        idx_to_key = {v: k for k, v in self.doc_ref_to_idx.items()}
 
        all_y_true, all_y_pred = [], []
 
        for q_text in tqdm(queries, desc="[SpladeKnnModel] Querying"):
            q_vec = self._encode_text(q_text)
 
            distances, indices = self.knn.kneighbors(q_vec, n_neighbors=corpus_size)
 
            y_pred = [0.0] * corpus_size
            for annoy_idx, dist in zip(indices[0], distances[0]):
                y_pred[annoy_idx] = float(1.0 - dist)
 
            y_true = [0] * corpus_size
            targets = relevance_map.get(q_text, set())
            for pos in range(corpus_size):
                key = idx_to_key.get(pos)
                if key in targets:
                    y_true[pos] = 1
 
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
 
        return all_y_true, all_y_pred
 
    def _cosine_sim_sparse(self, a: csr_matrix, b: csr_matrix) -> float:
        dot = float(a.dot(b.T).toarray()[0, 0])
        norm_a = float(np.sqrt(a.dot(a.T).toarray()[0, 0])) + 1e-12
        norm_b = float(np.sqrt(b.dot(b.T).toarray()[0, 0])) + 1e-12
        return dot / (norm_a * norm_b)
 
    def _parse_json_list(self, x) -> List[str]:
        import json
        if x is None:
            return []
        if isinstance(x, list):
            return [str(v) for v in x]
        s = str(x).strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(v) for v in obj]
        except Exception:
            pass
        return []