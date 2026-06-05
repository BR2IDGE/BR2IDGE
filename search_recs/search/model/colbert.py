import json
import string
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, List, Dict, Any
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from search_recs.recs.model import BaseRecsModel


class ResidualCompressor:
    def __init__(self, n_centroids: int = 1024, n_bits: int = 2, dim: int = 128):
        self.n_centroids = n_centroids
        self.n_bits      = n_bits
        self.dim         = dim
        self.n_buckets   = 2 ** n_bits
        self.centroids:      Optional[np.ndarray] = None
        self.bucket_cutoffs: Optional[np.ndarray] = None
        self.bucket_weights: Optional[np.ndarray] = None

    def fit(self, flat_embs: np.ndarray, sample_size: int = 300_000) -> None:
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for residual compression. "
                "Install it with: pip install scikit-learn"
            ) from exc

        n_total = len(flat_embs)
        if n_total > sample_size:
            idx    = np.random.choice(n_total, sample_size, replace=False)
            sample = flat_embs[idx]
        else:
            sample = flat_embs

        print(
            f"[ResidualCompressor] Fitting {self.n_centroids} centroids "
            f"on {len(sample):,} token embeddings..."
        )
        km = MiniBatchKMeans(
            n_clusters=self.n_centroids,
            batch_size=min(4096, len(sample)),
            n_init=4,
            max_iter=20,
            random_state=42,
        )
        km.fit(sample)
        self.centroids = km.cluster_centers_.astype(np.float32)

        centroid_ids = self._nearest_centroids(sample)
        residuals    = sample - self.centroids[centroid_ids]

        quantiles = np.linspace(0.0, 1.0, self.n_buckets + 1)
        self.bucket_cutoffs = np.zeros((self.dim, self.n_buckets - 1), dtype=np.float32)
        self.bucket_weights = np.zeros((self.dim, self.n_buckets),     dtype=np.float32)

        for d in range(self.dim):
            col  = residuals[:, d]
            cuts = np.quantile(col, quantiles[1:-1]).astype(np.float32)
            self.bucket_cutoffs[d] = cuts
            bucket_ids = np.searchsorted(cuts, col)
            for b in range(self.n_buckets):
                mask = bucket_ids == b
                self.bucket_weights[d, b] = float(col[mask].mean()) if mask.any() else 0.0

        print(
            f"[ResidualCompressor] Fit complete. "
            f"n_centroids={self.n_centroids}, n_bits={self.n_bits}, dim={self.dim}"
        )

    def compress(self, embs_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        centroid_ids = self._nearest_centroids(embs_2d)
        residuals    = embs_2d - self.centroids[centroid_ids]

        quantized = np.empty((len(embs_2d), self.dim), dtype=np.uint8)
        for d in range(self.dim):
            quantized[:, d] = np.clip(
                np.searchsorted(self.bucket_cutoffs[d], residuals[:, d]),
                0, self.n_buckets - 1,
            ).astype(np.uint8)

        return centroid_ids, quantized

    def decompress(
        self,
        centroid_ids: np.ndarray,
        quantized:    np.ndarray,
    ) -> np.ndarray:
        if centroid_ids.max() >= self.n_centroids or centroid_ids.min() < 0:
            raise ValueError(
                f"centroid_ids out of range [0, {self.n_centroids}): "
                f"got min={centroid_ids.min()}, max={centroid_ids.max()}."
            )
        centers   = self.centroids[centroid_ids]
        q_safe    = np.clip(quantized, 0, self.n_buckets - 1)
        residuals = self.bucket_weights[
            np.arange(self.dim)[:, None], q_safe.T
        ].T.astype(np.float32)
        return centers + residuals

    def _nearest_centroids(self, embs: np.ndarray, chunk: int = 8192) -> np.ndarray:
        ids = np.empty(len(embs), dtype=np.int32)
        C   = self.centroids
        for i in range(0, len(embs), chunk):
            sim                = embs[i : i + chunk] @ C.T
            ids[i : i + chunk] = sim.argmax(axis=1).astype(np.int32)
        return ids


class ColBERTEncoder(nn.Module):
    def __init__(self, backbone: str = "colbert-ir/colbertv2.0", dim: int = 128, device: str = "cpu"):
        super().__init__()
        self.bert   = AutoModel.from_pretrained(backbone)
        hidden      = self.bert.config.hidden_size
        self.linear = nn.Linear(hidden, dim, bias=False)
        self.dim    = dim
        self.to(device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out        = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embs = out.last_hidden_state
        projected  = self.linear(token_embs)
        normalised = F.normalize(projected, p=2, dim=-1)
        return normalised


class ColBERTModel(BaseRecsModel):
    QUERY_TOKEN = "[Q]"
    DOC_TOKEN   = "[D]"

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        params = self.model_params or {}

        self.backbone:     str = params.get("backbone",     "colbert-ir/colbertv2.0")
        self.dim:          int = params.get("dim",          128)
        self.query_maxlen: int = params.get("query_maxlen", 32)
        self.doc_maxlen:   int = params.get("doc_maxlen",   180)
        self.batch_size:   int = params.get("batch_size",   32)

        _dev = params.get("device", None)
        if _dev is None:
            if torch.cuda.is_available():
                _dev = "cuda"
            elif torch.backends.mps.is_available():
                _dev = "mps"
            else:
                _dev = "cpu"
        self.device: str = _dev

        self.qcol:    str  = params.get("query_col",        "search_query")
        self.dcol:    str  = params.get("doc_col",          "document")
        self.did_col: str  = params.get("doc_id_col",       "document_id")
        self.filter_punct: bool = params.get("filter_punctuation", True)

        self.use_compression: bool = params.get("use_compression", True)
        self.n_centroids:     int  = params.get("n_centroids",     1024)
        self.n_bits:          int  = params.get("n_bits",          2)
        self.compress_sample: int  = params.get("compress_sample", 300_000)
        self._compressor: Optional[ResidualCompressor] = None

        self._collection:  Optional[List[Dict]] = None
        self._doc_ids:     Optional[List[str]]  = None

        self._doc_embs:  Optional[torch.Tensor] = None
        self._doc_masks: Optional[torch.Tensor] = None

        self._doc_centroid_ids: Optional[np.ndarray] = None
        self._doc_quantized:    Optional[np.ndarray] = None

        self.encoder:     Optional[ColBERTEncoder] = None
        self.tokenizer    = None
        self.corpus_size: int = 0

        print(
            f"[ColBERTModel v2] Init | Backbone: {self.backbone} | dim={self.dim} | "
            f"device={self.device} | compression={'ON' if self.use_compression else 'OFF'} "
            f"(n_centroids={self.n_centroids}, n_bits={self.n_bits})"
        )

    # ------------------------------------------------------------------ encoder

    def _load_tokenizer_and_encoder(self):
        if self.tokenizer is None:
            print(f"[ColBERTModel v2] Loading tokenizer: {self.backbone}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.backbone)
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [self.QUERY_TOKEN, self.DOC_TOKEN]}
            )
        if self.encoder is None:
            print(f"[ColBERTModel v2] Building ColBERT encoder...")
            self.encoder = ColBERTEncoder(backbone=self.backbone, dim=self.dim, device=self.device)
            self.encoder.bert.resize_token_embeddings(len(self.tokenizer))
            self.encoder.eval()

    def _encode_queries(self, texts: List[str]) -> torch.Tensor:
        augmented = [f"{self.QUERY_TOKEN} {t}" for t in texts]
        enc = self.tokenizer(
            augmented, max_length=self.query_maxlen,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        mask_id   = self.tokenizer.mask_token_id
        pad_id    = self.tokenizer.pad_token_id
        input_ids = enc["input_ids"].clone()
        input_ids[input_ids == pad_id] = mask_id
        return self._run_encoder(input_ids, enc["attention_mask"])

    def _encode_docs(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        augmented = [f"{self.DOC_TOKEN} {t}" for t in texts]
        enc = self.tokenizer(
            augmented, max_length=self.doc_maxlen,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        embs = self._run_encoder(enc["input_ids"], enc["attention_mask"])
        mask = enc["attention_mask"].bool()
        if self.filter_punct:
            punct_ids = self._punct_token_ids()
            is_punct  = torch.isin(enc["input_ids"], punct_ids)
            mask      = mask & (~is_punct)
        return embs, mask

    def _run_encoder(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(input_ids), self.batch_size):
                ids_b  = input_ids[i : i + self.batch_size].to(self.device)
                mask_b = attention_mask[i : i + self.batch_size].to(self.device)
                all_embs.append(self.encoder(ids_b, mask_b).cpu())
        return torch.cat(all_embs, dim=0)

    @torch.no_grad()
    def _punct_token_ids(self) -> torch.Tensor:
        ids = set()
        for ch in list(string.punctuation):
            ids.update(self.tokenizer.encode(ch, add_special_tokens=False))
        return torch.tensor(list(ids), dtype=torch.long)

    # ------------------------------------------------------------------ MaxSim

    @staticmethod
    def _maxsim_scores(
        Q: torch.Tensor,
        D: torch.Tensor,
        D_mask: torch.Tensor,
        batch_docs: int = 512,
    ) -> np.ndarray:
        nq     = Q.shape[0]
        nd     = D.shape[0]
        scores = np.zeros((nq, nd), dtype=np.float32)

        for d_start in range(0, nd, batch_docs):
            d_end  = min(d_start + batch_docs, nd)
            D_b    = D[d_start:d_end]
            mask_b = D_mask[d_start:d_end]

            sim     = torch.einsum("qid,bjd->qbij", Q, D_b)
            invalid = ~mask_b.unsqueeze(0).unsqueeze(2)
            sim     = sim.masked_fill(invalid, float("-inf"))

            maxsim       = sim.max(dim=-1).values.clamp(min=0.0)
            chunk_scores = maxsim.sum(dim=-1)
            scores[:, d_start:d_end] = chunk_scores.numpy()

        return scores

    # ------------------------------------------------------------------ compression

    def _compress_index(self, embs: torch.Tensor) -> None:
        N, L, D = embs.shape
        flat    = embs.view(-1, D).numpy()

        self._compressor = ResidualCompressor(
            n_centroids=self.n_centroids, n_bits=self.n_bits, dim=D
        )
        self._compressor.fit(flat, sample_size=self.compress_sample)

        print(f"[ColBERTModel v2] Compressing {N * L:,} token embeddings...")
        centroid_ids, quantized = self._compressor.compress(flat)

        self._doc_centroid_ids = centroid_ids.reshape(N, L)
        self._doc_quantized    = quantized.reshape(N, L, D)

        original_mb   = embs.numel() * 4 / 1e6
        compressed_mb = (self._doc_centroid_ids.nbytes + self._doc_quantized.nbytes) / 1e6
        print(
            f"[ColBERTModel v2] Index compressed: "
            f"{original_mb:.1f} MB → {compressed_mb:.1f} MB "
            f"({original_mb / compressed_mb:.1f}x reduction)"
        )

    def _get_doc_embs(self, indices: List[int]) -> torch.Tensor:
        if self.use_compression and self._doc_centroid_ids is not None:
            n    = len(indices)
            L    = self._doc_centroid_ids.shape[1]
            cids = self._doc_centroid_ids[indices].reshape(-1)
            qres = self._doc_quantized[indices].reshape(-1, self.dim)
            rec  = self._compressor.decompress(cids, qres)
            return torch.from_numpy(rec).view(n, L, self.dim)
        else:
            return self._doc_embs[indices]

    # ------------------------------------------------------------------ preprocess

    def preprocess(self, train_data: pd.DataFrame, **kwargs):
        test_data          = kwargs.get("test_data")
        has_candidate_mode = isinstance(test_data, pd.DataFrame) and ("candidate_ids" in test_data.columns)

        if has_candidate_mode:
            unique_docs = train_data[[self.did_col, self.dcol]].drop_duplicates(subset=[self.did_col])
        else:
            parts   = [train_data] + ([test_data] if test_data is not None else [])
            full_df = pd.concat(parts, ignore_index=True)
            unique_docs = full_df[[self.did_col, self.dcol]].drop_duplicates(subset=[self.did_col])

        collection_df       = unique_docs.rename(columns={self.dcol: "text", self.did_col: "id"})
        collection_df["id"] = collection_df["id"].astype(str)
        self._collection    = collection_df[["id", "text"]].to_dict("records")
        self.corpus_size    = len(self._collection)
        print(f"[ColBERTModel v2] Corpus size: {self.corpus_size} documents.")

    # ------------------------------------------------------------------ fit

    def fit(self):
        if self._collection is None:
            raise RuntimeError("Execute preprocess() before fit().")

        self._load_tokenizer_and_encoder()
        doc_texts     = [d["text"] for d in self._collection]
        self._doc_ids = [d["id"]   for d in self._collection]

        print(f"[ColBERTModel v2] Encoding {self.corpus_size} documents (offline indexing)...")
        all_embs:  List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = []

        for i in tqdm(range(0, self.corpus_size, self.batch_size), desc="Indexing docs"):
            embs, masks = self._encode_docs(doc_texts[i : i + self.batch_size])
            all_embs.append(embs)
            all_masks.append(masks)

        raw_embs        = torch.cat(all_embs,  dim=0)
        self._doc_masks = torch.cat(all_masks, dim=0)
        print(
            f"[ColBERTModel v2] Encoding complete. "
            f"Raw embeddings: {tuple(raw_embs.shape)} | dtype: {raw_embs.dtype}"
        )

        if self.use_compression:
            self._compress_index(raw_embs)
            del raw_embs
            self._doc_embs = None
        else:
            print("[ColBERTModel v2] Compression disabled — storing raw float32 embeddings.")
            self._doc_embs = raw_embs

    # ------------------------------------------------------------------ helpers

    def _parse_json_list(self, v: Any) -> List[str]:
        if isinstance(v, list):
            return [str(x) for x in v]
        if not v:
            return []
        try:
            return [str(x) for x in json.loads(v)]
        except Exception:
            return []

    # ------------------------------------------------------------------ prediction

    def prediction(self, test_data: pd.DataFrame) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if self.use_compression and self._doc_centroid_ids is None:
            raise RuntimeError("Model not indexed. Execute fit() first.")
        if not self.use_compression and self._doc_embs is None:
            raise RuntimeError("Model not indexed. Execute fit() first.")

        self._load_tokenizer_and_encoder()
        has_candidate_mode = "candidate_ids" in test_data.columns

        # ---- CANDIDATE MODE ------------------------------------------------
        if has_candidate_mode:
            print("[ColBERTModel v2] Candidate-mode prediction (late-interaction)...")
            queries    = test_data[self.qcol].astype(str).tolist()
            cand_lists = [self._parse_json_list(r["candidate_ids"])          for _, r in test_data.iterrows()]
            gt_sets    = [set(self._parse_json_list(r["ground_truth_ids"])) for _, r in test_data.iterrows()]
            doc_id_to_idx: Dict[str, int] = {did: i for i, did in enumerate(self._doc_ids)}

            all_y_true: List[np.ndarray] = []
            all_y_pred: List[np.ndarray] = []
            n = len(queries)

            for bstart in tqdm(range(0, n, 64), desc="ColBERT v2 Search"):
                bend = min(bstart + 64, n)
                Q    = self._encode_queries(queries[bstart:bend])

                for bi, qi in enumerate(range(bstart, bend)):
                    cands = cand_lists[qi]
                    if not cands:
                        all_y_pred.append(np.array([], dtype=np.float32))
                        all_y_true.append(np.array([], dtype=np.float32))
                        continue

                    cand_indices = [doc_id_to_idx.get(str(c), -1) for c in cands]
                    valid = [(li, ci) for li, ci in enumerate(cand_indices) if ci != -1]

                    y_pred = np.zeros(len(cands), dtype=np.float32)
                    y_true = np.array(
                        [1.0 if c in gt_sets[qi] else 0.0 for c in cands], dtype=np.float32
                    )

                    if valid:
                        local_idxs  = [v[0] for v in valid]
                        corpus_idxs = [v[1] for v in valid]
                        D_cand      = self._get_doc_embs(corpus_idxs)
                        D_mask_cand = self._doc_masks[corpus_idxs]
                        Q_single    = Q[bi].unsqueeze(0)
                        cand_scores = self._maxsim_scores(Q_single, D_cand, D_mask_cand).squeeze(0)
                        for li, score in zip(local_idxs, cand_scores):
                            y_pred[li] = float(score)

                    all_y_pred.append(y_pred)
                    all_y_true.append(y_true)

            return all_y_true, all_y_pred

        # ---- STANDARD / FULL-CORPUS MODE -----------------------------------
        else:
            print("[ColBERTModel v2] Standard full-corpus search mode...")
            queries         = test_data[self.qcol].astype(str).tolist()
            doc_text_to_idx: Dict[str, int] = {d["text"]: i for i, d in enumerate(self._collection)}

            all_y_pred: List[np.ndarray] = []
            all_y_true: List[np.ndarray] = []
            doc_chunk_size = 256
            n = len(queries)

            print(f"[ColBERTModel v2] Running MaxSim over {self.corpus_size} docs...")
            for bstart in tqdm(range(0, n, 32), desc="ColBERT v2 Ranking"):
                bend         = min(bstart + 32, n)
                Q            = self._encode_queries(queries[bstart:bend])
                batch_scores = np.zeros((bend - bstart, self.corpus_size), dtype=np.float32)

                for d_start in range(0, self.corpus_size, doc_chunk_size):
                    d_end   = min(d_start + doc_chunk_size, self.corpus_size)
                    indices = list(range(d_start, d_end))
                    D_chunk      = self._get_doc_embs(indices)
                    D_mask_chunk = self._doc_masks[d_start:d_end]
                    batch_scores[:, d_start:d_end] = self._maxsim_scores(
                        Q, D_chunk, D_mask_chunk, batch_docs=doc_chunk_size
                    )

                for bi, qi in enumerate(range(bstart, bend)):
                    y_pred      = batch_scores[bi].astype(np.float32)
                    y_true      = np.zeros(self.corpus_size, dtype=np.float32)
                    target_text = str(test_data.iloc[qi][self.dcol])
                    target_idx  = doc_text_to_idx.get(target_text)
                    if target_idx is not None:
                        y_true[target_idx] = 1.0
                    all_y_pred.append(y_pred)
                    all_y_true.append(y_true)

            return all_y_true, all_y_pred
