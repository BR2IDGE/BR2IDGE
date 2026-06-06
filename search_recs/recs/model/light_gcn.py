from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import scipy.sparse as sp
except Exception: 
    sp = None

try:
    import torch
    from torch import nn
except Exception: 
    torch = None
    nn = None

from search_recs.recs.model import BaseRecsModel

_NNModuleBase = nn.Module if nn is not None else object


class _LightGCNCore(_NNModuleBase):

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        n_layers: int,
        norm_adj: "torch.Tensor",
        layer_weights: Optional["torch.Tensor"] = None,
    ) -> None:
        super().__init__()

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.n_layers = int(n_layers)
        self.register_buffer("norm_adj", norm_adj.coalesce())
        self.layer_weights = layer_weights

        self.user_embedding = nn.Embedding(self.n_users, embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def get_ego_embeddings(self) -> "torch.Tensor":
        return torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

    def forward(self) -> Tuple["torch.Tensor", "torch.Tensor"]:
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
            embeddings_list.append(all_embeddings)

        stacked = torch.stack(embeddings_list, dim=1)

        if self.layer_weights is not None:
            w = self.layer_weights.to(stacked.device).view(1, -1, 1)
            final_embeddings = torch.sum(stacked * w, dim=1)
        else:
            final_embeddings = torch.mean(stacked, dim=1)

        user_embeddings, item_embeddings = torch.split(
            final_embeddings, [self.n_users, self.n_items], dim=0
        )
        return user_embeddings, item_embeddings


class LightGCNModel(BaseRecsModel):

    def __init__(self, model_config: Dict[str, Any], features_config: Optional[dict] = None):
        super().__init__(model_config)

        if torch is None or nn is None:
            raise ImportError("LightGCNModel requires PyTorch (`torch`) to be installed.")
        if sp is None:
            raise ImportError("LightGCNModel requires SciPy (`scipy`) to be installed.")

        params = self.model_params or {}
        features_config = features_config or {}

        self.user_col = features_config.get("user_col") or params.get("user_col") or "user"
        self.item_col = features_config.get("item_col") or params.get("item_col") or "item"
        self.label_col = features_config.get("label_col") or params.get("label_col") or "label"

        self.embedding_dim = int(params.get("embedding_dim", 64))
        self.n_layers = int(params.get("n_layers", 3))
        self.epochs = int(params.get("epochs", 100))
        self.learning_rate = float(params.get("learning_rate", 1e-3))
        self.reg_weight = float(params.get("reg_weight", 1e-4))
        self.batch_size = int(params.get("batch_size", 2048))
        self.seed = int(params.get("seed", 42))
        self.positive_threshold = float(params.get("positive_threshold", 0.0))
        self.as_binary = bool(params.get("as_binary", True))
        self.verbose = bool(params.get("verbose", True))
        self.print_every = int(params.get("print_every", 10))
        self.cold_start_score = str(params.get("cold_start_score", "neg_inf")).strip().lower()
        self.device = str(params.get("device", "cuda" if torch.cuda.is_available() else "cpu")).lower()

        raw_layer_weights = params.get("layer_weights", None)
        self.layer_weights: Optional[np.ndarray] = None
        if raw_layer_weights is not None:
            if isinstance(raw_layer_weights, (str, bytes)) or (not isinstance(raw_layer_weights, Sequence)):
                raise ValueError("layer_weights must be a sequence of float values.")
            self.layer_weights = np.asarray(list(raw_layer_weights), dtype=np.float32)

        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0.")
        if self.n_layers < 0:
            raise ValueError("n_layers must be >= 0.")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if self.reg_weight < 0:
            raise ValueError("reg_weight must be >= 0.")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be 'cpu' or start with 'cuda'.")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print("[LightGCN] CUDA requested but unavailable. Falling back to CPU.")
            self.device = "cpu"

        if self.cold_start_score not in {"neg_inf", "zero", "popularity"}:
            raise ValueError("cold_start_score must be one of: neg_inf, zero, popularity.")

        self.model: Optional[_LightGCNCore] = None
        self.optimizer = None

        self.user2idx: Dict[str, int] = {}
        self.item2idx: Dict[str, int] = {}

        self.train_user_indices: Optional[np.ndarray] = None
        self.train_item_indices: Optional[np.ndarray] = None

        self.user_pos_items: List[set[int]] = []
        self.item_popularity: Optional[np.ndarray] = None

        self._cached_user_emb: Optional[np.ndarray] = None
        self._cached_item_emb: Optional[np.ndarray] = None
        self._fitted = False

    def _set_random_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _ensure_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("train_data/test_data must be a pandas DataFrame.")

        missing = [c for c in (self.user_col, self.item_col) if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. Available columns={list(df.columns)}"
            )

        out = df.copy()
        out[self.user_col] = out[self.user_col].astype(str)
        out[self.item_col] = out[self.item_col].astype(str)

        if self.label_col not in out.columns:
            out[self.label_col] = 1.0
        else:
            out[self.label_col] = pd.to_numeric(out[self.label_col], errors="coerce").fillna(0.0)
        return out

    def _build_norm_adj(self, n_users: int, n_items: int, u_idx: np.ndarray, i_idx: np.ndarray) -> "torch.Tensor":
        n_nodes = n_users + n_items

        rows = np.concatenate([u_idx, i_idx + n_users]).astype(np.int64, copy=False)
        cols = np.concatenate([i_idx + n_users, u_idx]).astype(np.int64, copy=False)
        data = np.ones(rows.shape[0], dtype=np.float32)
        a = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32).tocsr()

        degree = np.asarray(a.sum(axis=1)).reshape(-1)
        degree_inv_sqrt = np.power(np.maximum(degree, 1e-12), -0.5)
        d_inv = sp.diags(degree_inv_sqrt)
        norm_adj = (d_inv @ a @ d_inv).tocoo()

        indices = np.vstack([norm_adj.row, norm_adj.col]).astype(np.int64)
        values = norm_adj.data.astype(np.float32)
        tensor = torch.sparse_coo_tensor(
            indices=torch.from_numpy(indices),
            values=torch.from_numpy(values),
            size=norm_adj.shape,
            dtype=torch.float32,
        )
        return tensor.coalesce().to(self.device)

    def _sample_negative_items(self, users: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_items = len(self.item2idx)
        out = np.empty(len(users), dtype=np.int64)

        for idx, u in enumerate(users):
            u = int(u)
            pos_set = self.user_pos_items[u]
            if len(pos_set) >= n_items:
                out[idx] = int(rng.integers(0, n_items))
                continue

            neg = int(rng.integers(0, n_items))
            while neg in pos_set:
                neg = int(rng.integers(0, n_items))
            out[idx] = neg

        return out

    def _invalidate_cache(self) -> None:
        self._cached_user_emb = None
        self._cached_item_emb = None

    def _refresh_cached_embeddings(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        self.model.eval()
        with torch.no_grad():
            user_all, item_all = self.model()
        self._cached_user_emb = user_all.detach().cpu().numpy()
        self._cached_item_emb = item_all.detach().cpu().numpy()

    def preprocess(self, train_data: pd.DataFrame, **kwargs):
        self._set_random_seed()

        df = self._ensure_schema(train_data)

        if self.positive_threshold is not None:
            before = len(df)
            df = df[df[self.label_col] > float(self.positive_threshold)].copy()
            if df.empty:
                raise ValueError(
                    f"[LightGCN] No interactions left after label > {self.positive_threshold} filter."
                )
            if self.verbose:
                print(
                    f"[LightGCN] Applied positive_threshold>{self.positive_threshold}: "
                    f"{before} -> {len(df)}"
                )

        if self.as_binary:
            df[self.label_col] = 1.0
        df = df.drop_duplicates(subset=[self.user_col, self.item_col], keep="first")

        users = pd.Index(df[self.user_col].unique())
        items = pd.Index(df[self.item_col].unique())

        self.user2idx = {u: i for i, u in enumerate(users)}
        self.item2idx = {it: j for j, it in enumerate(items)}

        u_idx = df[self.user_col].map(self.user2idx).astype(np.int64).to_numpy()
        i_idx = df[self.item_col].map(self.item2idx).astype(np.int64).to_numpy()

        self.train_user_indices = u_idx
        self.train_item_indices = i_idx

        n_users = len(self.user2idx)
        n_items = len(self.item2idx)

        self.user_pos_items = [set() for _ in range(n_users)]
        for u, i in zip(u_idx, i_idx):
            self.user_pos_items[int(u)].add(int(i))

        self.item_popularity = np.bincount(i_idx, minlength=n_items).astype(np.float32)
        if self.item_popularity.max() > 0:
            self.item_popularity /= self.item_popularity.max()

        norm_adj = self._build_norm_adj(n_users, n_items, u_idx, i_idx)

        layer_weights_tensor = None
        if self.layer_weights is not None:
            expected = self.n_layers + 1
            if len(self.layer_weights) != expected:
                raise ValueError(
                    f"layer_weights length must be n_layers+1 ({expected}), got {len(self.layer_weights)}."
                )
            w = self.layer_weights.astype(np.float32)
            s = float(np.sum(w))
            if not math.isfinite(s) or abs(s) < 1e-12:
                raise ValueError("layer_weights sum must be finite and non-zero.")
            w = w / s
            layer_weights_tensor = torch.tensor(w, dtype=torch.float32, device=self.device)

        self.model = _LightGCNCore(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=self.embedding_dim,
            n_layers=self.n_layers,
            norm_adj=norm_adj,
            layer_weights=layer_weights_tensor,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self._fitted = False
        self._invalidate_cache()

        print(
            f"[LightGCN] Preprocess ready: users={n_users}, items={n_items}, "
            f"interactions={len(df)}, layers={self.n_layers}, emb_dim={self.embedding_dim}, device={self.device}"
        )

    def fit(self):
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Call preprocess(train_data) before fit().")
        if self.train_user_indices is None or self.train_item_indices is None:
            raise RuntimeError("Missing training interactions. Call preprocess(train_data) first.")

        n_interactions = len(self.train_user_indices)
        if n_interactions == 0:
            raise RuntimeError("No interactions available for training.")

        rng = np.random.default_rng(self.seed)
        self.model.train()

        for epoch in range(1, self.epochs + 1):
            perm = rng.permutation(n_interactions)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_interactions, self.batch_size):
                end = min(start + self.batch_size, n_interactions)
                idx = perm[start:end]

                users = self.train_user_indices[idx]
                pos_items = self.train_item_indices[idx]
                neg_items = self._sample_negative_items(users, rng)

                users_t = torch.from_numpy(users).long().to(self.device)
                pos_t = torch.from_numpy(pos_items).long().to(self.device)
                neg_t = torch.from_numpy(neg_items).long().to(self.device)

                user_all, item_all = self.model()
                u_emb = user_all[users_t]
                p_emb = item_all[pos_t]
                n_emb = item_all[neg_t]

                pos_scores = torch.sum(u_emb * p_emb, dim=1)
                neg_scores = torch.sum(u_emb * n_emb, dim=1)
                bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12).mean()

                u_ego = self.model.user_embedding(users_t)
                p_ego = self.model.item_embedding(pos_t)
                n_ego = self.model.item_embedding(neg_t)
                reg_loss = (
                    u_ego.pow(2).sum(dim=1)
                    + p_ego.pow(2).sum(dim=1)
                    + n_ego.pow(2).sum(dim=1)
                ).mean() * 0.5

                loss = bpr_loss + self.reg_weight * reg_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += float(loss.detach().cpu().item())
                n_batches += 1

            if self.verbose and (epoch == 1 or epoch == self.epochs or epoch % self.print_every == 0):
                mean_loss = epoch_loss / max(n_batches, 1)
                print(f"[LightGCN] Epoch {epoch:03d}/{self.epochs} | loss={mean_loss:.6f}")

        self._fitted = True
        self._refresh_cached_embeddings()
        print("[LightGCN] Training completed.")

    def prediction(self, data_input: pd.DataFrame, **kwargs) -> Tuple[List[float], List[float]]:
        if self.model is None:
            raise RuntimeError("Call preprocess() and fit() before prediction().")

        df = self._ensure_schema(data_input)

        if self._cached_user_emb is None or self._cached_item_emb is None:
            self._refresh_cached_embeddings()

        user_map = df[self.user_col].map(self.user2idx)
        item_map = df[self.item_col].map(self.item2idx)

        n_rows = len(df)
        if self.cold_start_score == "neg_inf":
            y_pred = np.full(n_rows, -1e9, dtype=np.float32)
        else:
            y_pred = np.zeros(n_rows, dtype=np.float32)

        valid_mask = user_map.notna() & item_map.notna()
        if valid_mask.any():
            u_idx = user_map[valid_mask].astype(np.int64).to_numpy()
            i_idx = item_map[valid_mask].astype(np.int64).to_numpy()
            u_emb = self._cached_user_emb[u_idx]
            i_emb = self._cached_item_emb[i_idx]
            scores = np.sum(u_emb * i_emb, axis=1).astype(np.float32, copy=False)
            y_pred[valid_mask.to_numpy()] = scores

        if self.cold_start_score == "popularity" and self.item_popularity is not None:
            item_known_mask = item_map.notna() & (~valid_mask)
            if item_known_mask.any():
                i_idx_cold = item_map[item_known_mask].astype(np.int64).to_numpy()
                y_pred[item_known_mask.to_numpy()] = self.item_popularity[i_idx_cold]

        y_true = (
            pd.to_numeric(df[self.label_col], errors="coerce").fillna(0.0).astype(float).tolist()
            if self.label_col in df.columns
            else [0.0] * n_rows
        )
        return y_true, y_pred.astype(float).tolist()
