import hashlib
import json
import os
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz
from sklearn.preprocessing import normalize
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from search_recs.recs.model import BaseRecsModel


class SpladeKnnModel(BaseRecsModel):

    def __init__(self, model_config: dict):
        super().__init__(model_config)

        print(f"\n[SpladeKnnModel] Config keys: {list(model_config.keys())}")

        params = model_config.get("parameters", {})

        self.pretrained_name: str = params.get("pretrained_name", "naver/splade-v3")
        self.token_max_length: int = int(params.get("token_max_length", 256))
        self.query_max_length: int = int(params.get("query_max_length", 64))
        self.encode_batch_size: int = int(params.get("encode_batch_size", 16))
        self.query_chunk_size: int = int(params.get("query_chunk_size", 64))
        self.auth_token: str = params.get("auth_token", "")

        self.model_dir: str = params.get("model_dir", "./output/splade_knn")
        self.index_path: str = os.path.join(self.model_dir, "splade_matrix.npz")
        self.index_meta_path: str = os.path.join(self.model_dir, "splade_index_meta.json")

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

        self.corpus_matrix: Optional[csr_matrix] = None
        self.corpus_docs: Optional[List[str]] = None
        self.doc_ref_to_idx: Dict[str, int] = {}

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        print(f"[SpladeKnnModel] Loading SPLADE model: {self.pretrained_name}")
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.pretrained_name,
            use_auth_token=self.auth_token or None,
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_name,
            use_auth_token=self.auth_token or None,
        )
        self.vocab_size = len(self.tokenizer.get_vocab())
        print(f"[SpladeKnnModel] Loaded. Vocab={self.vocab_size} | Device={self.device}")

    @torch.no_grad()
    def _encode_texts(
        self,
        texts: List[str],
        max_length: int,
        batch_size: Optional[int] = None,
        desc: str = "Encoding",
    ) -> csr_matrix:
        """Encode texts in true batches and return an L2-normalized csr_matrix (N, vocab)."""
        if batch_size is None:
            batch_size = self.encode_batch_size

        rows: List[csr_matrix] = []
        n = len(texts)

        for start in tqdm(range(0, n, batch_size), desc=f"[SpladeKnnModel] {desc}"):
            batch_texts = [str(t) if t is not None else "" for t in texts[start : start + batch_size]]
            enc = self.tokenizer(
                batch_texts,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            output = self.model(**enc)
            activations = torch.log1p(torch.relu(output.logits)) * enc.attention_mask.unsqueeze(-1)
            vec = activations.max(dim=1).values 

            rows.append(csr_matrix(vec.cpu().numpy().astype(np.float32)))

        if not rows:
            return csr_matrix((0, self.vocab_size), dtype=np.float32)

        matrix = vstack(rows).astype(np.float32)
        matrix = normalize(matrix, norm="l2", axis=1, copy=False)
        return matrix

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

    def _corpus_signature(self, doc_keys: List[str]) -> str:
        h = hashlib.sha1()
        h.update(self.pretrained_name.encode("utf-8"))
        h.update(f"|max_len={self.token_max_length}".encode("utf-8"))
        h.update(f"|n={len(doc_keys)}".encode("utf-8"))
        for k in doc_keys:
            h.update(b"|")
            h.update(str(k).encode("utf-8"))
        return h.hexdigest()

    def build_index(self, force_rebuild: bool = False) -> None:
        self._ensure_model()
        print("[SpladeKnnModel] Building SPLADE index...")

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

        signature = self._corpus_signature(doc_keys)

        cache_valid = False
        if not force_rebuild and os.path.exists(self.index_path) and os.path.exists(self.index_meta_path):
            try:
                with open(self.index_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                cache_valid = meta.get("signature") == signature
            except Exception:
                cache_valid = False

        if cache_valid:
            print(f"[SpladeKnnModel] Loading cached matrix: {self.index_path}")
            self.corpus_matrix = load_npz(self.index_path)
        else:
            if os.path.exists(self.index_path):
                print("[SpladeKnnModel] Cache signature mismatch — re-encoding corpus.")
            print(f"[SpladeKnnModel] Encoding {len(doc_texts)} unique documents...")
            self.corpus_matrix = self._encode_texts(
                doc_texts,
                max_length=self.token_max_length,
                desc="Indexing documents",
            )
            os.makedirs(self.model_dir, exist_ok=True)
            save_npz(self.index_path, self.corpus_matrix)
            with open(self.index_meta_path, "w", encoding="utf-8") as f:
                json.dump({"signature": signature, "n_docs": len(doc_keys)}, f)
            print(f"[SpladeKnnModel] Matrix saved to: {self.index_path}")

        print(f"[SpladeKnnModel] Index ready. Docs={len(doc_texts)} | Shape={self.corpus_matrix.shape}")

    def prediction(self, test_data: pd.DataFrame) -> Tuple[List[List[int]], List[List[float]]]:
        print("\n[SpladeKnnModel] Starting Prediction Phase...")

        if not isinstance(test_data, pd.DataFrame):
            raise ValueError("test_data must be a pandas DataFrame.")

        self._ensure_model()
        if self.corpus_matrix is None:
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

        q_matrix = self._encode_texts(
            queries,
            max_length=self.query_max_length,
            desc="Encoding queries",
        )

        if has_candidate_col:
            return self._predict_candidates(test_data, q_matrix)
        return self._predict_full_corpus(test_data, queries, q_matrix)

    def _predict_candidates(
        self,
        test_data: pd.DataFrame,
        q_matrix: csr_matrix,
    ) -> Tuple[List[List[int]], List[List[float]]]:
        print("[SpladeKnnModel] Mode: Candidate Reranking")

        cand_lists = [self._parse_json_list(r) for r in test_data["candidate_ids"].tolist()]
        gt_sets = [set(self._parse_json_list(r)) for r in test_data["ground_truth_ids"].tolist()]

        all_y_true: List[List[int]] = []
        all_y_pred: List[List[float]] = []

        for i in tqdm(range(q_matrix.shape[0]), desc="[SpladeKnnModel] Reranking"):
            q_vec = q_matrix[i]
            cand_ids = cand_lists[i]
            gt_set = gt_sets[i]

            cand_indices = [self.doc_ref_to_idx.get(str(c), -1) for c in cand_ids]
            valid_positions = [pos for pos, idx in enumerate(cand_indices) if idx != -1]
            valid_idxs = [cand_indices[pos] for pos in valid_positions]

            y_pred: List[float] = [-10.0] * len(cand_ids)
            if valid_idxs:
                D_sub = self.corpus_matrix[valid_idxs]
                scores = np.asarray((q_vec @ D_sub.T).toarray()).flatten()
                for pos, score in zip(valid_positions, scores):
                    y_pred[pos] = float(score)

            y_true = [1 if str(c) in gt_set else 0 for c in cand_ids]
            all_y_pred.append(y_pred)
            all_y_true.append(y_true)

        return all_y_true, all_y_pred

    def _predict_full_corpus(
        self,
        test_data: pd.DataFrame,
        queries: List[str],
        q_matrix: csr_matrix,
    ) -> Tuple[List[List[int]], List[List[float]]]:
        print("[SpladeKnnModel] Mode: Full Corpus Retrieval")

        has_id_col = self.did_col in test_data.columns
        target_col = self.did_col if has_id_col else self.dcol

        relevance_map: Dict[str, set] = {}
        for q, target in zip(
            test_data[self.qcol].astype(str).tolist(),
            test_data[target_col].astype(str).tolist(),
        ):
            relevance_map.setdefault(q, set()).add(target)

        corpus_size = self.corpus_matrix.shape[0]
        idx_to_key = {v: k for k, v in self.doc_ref_to_idx.items()}

        D_T = self.corpus_matrix.T.tocsr()

        score_chunks: List[np.ndarray] = []
        chunk = max(1, int(self.query_chunk_size))
        for start in tqdm(range(0, q_matrix.shape[0], chunk), desc="[SpladeKnnModel] Querying"):
            end = min(start + chunk, q_matrix.shape[0])
            Q_chunk = q_matrix[start:end]
            S = (Q_chunk @ D_T).toarray()
            score_chunks.append(S)

        scores_full = (
            np.vstack(score_chunks)
            if score_chunks
            else np.zeros((0, corpus_size), dtype=np.float32)
        )

        all_y_true: List[List[int]] = []
        all_y_pred: List[List[float]] = []

        for i, q_text in enumerate(queries):
            targets = relevance_map.get(q_text, set())
            y_true = [1 if idx_to_key.get(pos) in targets else 0 for pos in range(corpus_size)]
            all_y_pred.append(scores_full[i].astype(float).tolist())
            all_y_true.append(y_true)

        return all_y_true, all_y_pred

    def _parse_json_list(self, x) -> List[str]:
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
