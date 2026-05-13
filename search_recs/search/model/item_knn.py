from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from search_recs.recs.model import BaseRecsModel


class ItemKNNModel(BaseRecsModel):
    def __init__(self, model_config: Dict[str, Any], features_config: Optional[dict] = None):
        super().__init__(model_config)
        params = self.model_params or {}
        features_config = features_config or {}

        self.user_col = features_config.get("user_col") or params.get("user_col") or "user"
        self.item_col = features_config.get("item_col") or params.get("item_col") or "item"
        self.label_col = features_config.get("label_col") or params.get("label_col") or "label"

        self.similarity_metric = str(params.get("similarity_metric", "cosine")).lower().strip()
        self.cosine_variant_cfg = str(params.get("cosine_variant", "auto")).lower().strip()
        self.cosine_variant = "standard"
        self.as_binary_cfg = params.get("as_binary", "auto")
        self.as_binary = True
        self.as_similar_first = bool(params.get("as_similar_first", True))
        self.use_weighted_ratings_cfg = params.get("use_weighted_ratings", "auto")
        self.use_weighted_ratings = False
        self.normalize_scores_cfg = params.get("normalize_scores", "auto")
        self.normalize_scores = False
        self.exclude_seen_items = bool(params.get("exclude_seen_items", True))
        self.fallback_on_empty_neighbors = bool(params.get("fallback_on_empty_neighbors", False))
        self.center_ratings = bool(params.get("center_ratings", True))

        self.k_neighbors = params.get("k_neighbors", None)
        self.min_similarity = float(params.get("min_similarity", 0.0))
        self.positive_threshold_cfg = params.get("positive_threshold", "auto")
        self.positive_threshold: Optional[float] = None
        self.shrinkage = float(params.get("shrinkage", 0.0))
        self.cold_start_score = str(params.get("cold_start_score", "popularity")).lower().strip()
        self.eps = 1e-12

        if self.similarity_metric not in {"cosine", "jaccard"}:
            raise ValueError("similarity_metric must be 'cosine' or 'jaccard'.")
        if self.cosine_variant_cfg not in {"auto", "standard", "adjusted", "adjusted_cosine", "adjusted-cosine"}:
            raise ValueError(
                "cosine_variant must be one of: 'auto', 'standard', 'adjusted'."
            )

        self.user2idx: Dict[str, int] = {}
        self.item2idx: Dict[str, int] = {}
        self.idx2item: Dict[int, str] = {}

        self.interactions: Optional[csr_matrix] = None
        self.user_seen_items: List[np.ndarray] = []
        self.user_seen_vals: List[np.ndarray] = []

        self.item_neighbors_idx: List[np.ndarray] = []
        self.item_neighbors_sim: List[np.ndarray] = []
        self.item_neighbor_map: List[Dict[int, float]] = []

        self.item_popularity: Optional[np.ndarray] = None
        self.item_mean: Optional[np.ndarray] = None
        self.global_mean: float = 0.0
        self.feedback_mode: str = "unknown"

    @staticmethod
    def _is_auto(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str):
            return v.strip().lower() in {"auto", "infer", "adaptive", "default"}
        return False

    @staticmethod
    def _to_bool(v: Any, default: bool) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"1", "true", "yes", "on", "y"}:
                return True
            if s in {"0", "false", "no", "off", "n"}:
                return False
        return bool(default)

    def _infer_feedback_mode(self, labels: pd.Series) -> str:
        s = pd.to_numeric(labels, errors="coerce").fillna(0.0)
        nz = s[s > 0.0]
        if nz.empty:
            return "implicit_binary"
        max_label = float(nz.max())
        n_unique = int(nz.nunique())
        if max_label <= 1.0 and n_unique <= 2:
            return "implicit_binary"
        if max_label <= 10.0 and n_unique >= 5:
            return "explicit"
        return "implicit_count"

    def _resolve_positive_threshold(self, labels: pd.Series, feedback_mode: str) -> Optional[float]:
        if self._is_auto(self.positive_threshold_cfg):
            return 0.0

        try:
            thr = float(self.positive_threshold_cfg)
        except Exception:
            print(
                f"[ItemKNN] Warning: positive_threshold='{self.positive_threshold_cfg}' is invalid. "
                "Falling back to threshold>0."
            )
            return 0.0
        if labels is None or len(labels) == 0:
            return thr

        max_label = float(pd.to_numeric(labels, errors="coerce").fillna(0.0).max())

        if feedback_mode != "explicit" and max_label <= 1.0 and thr > 1.0:
            print(
                f"[ItemKNN] Warning: positive_threshold={thr} seems incompatible with "
                f"label range (max={max_label:.4f}). Falling back to threshold>0."
            )
            return 0.0

        return thr

    def _ensure_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("train_data/test_data must be a pandas DataFrame.")
        if self.user_col not in df.columns or self.item_col not in df.columns:
            raise ValueError(
                f"Required columns not found: user_col='{self.user_col}', item_col='{self.item_col}'. "
                f"Columns={list(df.columns)}"
            )
        out = df.copy()
        out[self.user_col] = out[self.user_col].astype(str)
        out[self.item_col] = out[self.item_col].astype(str)
        if self.label_col in out.columns:
            out[self.label_col] = pd.to_numeric(out[self.label_col], errors="coerce")
        else:
            out[self.label_col] = 1.0
        out[self.label_col] = out[self.label_col].fillna(0.0).astype(float)
        return out

    def _build_interaction_matrix(self, df: pd.DataFrame) -> csr_matrix:
        users = pd.Index(df[self.user_col].unique())
        items = pd.Index(df[self.item_col].unique())

        self.user2idx = {u: i for i, u in enumerate(users)}
        self.item2idx = {it: j for j, it in enumerate(items)}
        self.idx2item = {j: it for it, j in self.item2idx.items()}

        u_idx = df[self.user_col].map(self.user2idx).astype(int).to_numpy()
        i_idx = df[self.item_col].map(self.item2idx).astype(int).to_numpy()

        if self.as_binary:
            vals = np.ones(len(df), dtype=np.float32)
        else:
            vals = df[self.label_col].astype(np.float32).to_numpy()

        matrix = coo_matrix(
            (vals, (u_idx, i_idx)),
            shape=(len(self.user2idx), len(self.item2idx)),
            dtype=np.float32,
        ).tocsr()
        return matrix

    def _resolve_cosine_variant(self) -> str:
        cfg = self.cosine_variant_cfg
        if cfg in {"adjusted", "adjusted_cosine", "adjusted-cosine"}:
            return "adjusted"
        if cfg == "standard":
            return "standard"

        if (not self.as_binary) and self.feedback_mode == "explicit" and self.center_ratings:
            return "adjusted"
        return "standard"

    @staticmethod
    def _binary_support_matrix(x: csr_matrix) -> csr_matrix:
        out = x.copy()
        out.data = np.ones_like(out.data, dtype=np.float32)
        return out

    def _build_adjusted_user_centered_matrix(self, x: csr_matrix) -> csr_matrix:
        out = x.copy().astype(np.float32)
        data = out.data
        indptr = out.indptr

        for u in range(out.shape[0]):
            s = indptr[u]
            e = indptr[u + 1]
            if e <= s:
                continue
            mu = float(np.mean(data[s:e]))
            data[s:e] = data[s:e] - mu

        out.eliminate_zeros()
        return out

    @staticmethod
    def _gather_row_values(row_idx: np.ndarray, row_vals: np.ndarray, target_idx: np.ndarray) -> np.ndarray:
        if target_idx.size == 0 or row_idx.size == 0:
            return np.zeros(target_idx.size, dtype=np.float64)
        pos = np.searchsorted(row_idx, target_idx)
        ok = (pos < row_idx.size)
        out = np.zeros(target_idx.size, dtype=np.float64)
        if np.any(ok):
            ok2 = ok.copy()
            ok2[ok] = row_idx[pos[ok]] == target_idx[ok]
            if np.any(ok2):
                out[ok2] = row_vals[pos[ok2]].astype(np.float64, copy=False)
        return out

    def _compute_neighbors_cosine(
        self,
        x_sim: csr_matrix,
        x_support: csr_matrix,
        k_neighbors: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        xtx = (x_sim.T @ x_sim).tocsr()
        norms = np.sqrt(np.asarray(x_sim.power(2).sum(axis=0)).ravel()) + self.eps
        x_bin = self._binary_support_matrix(x_support)
        co_counts = (x_bin.T @ x_bin).tocsr()

        neighbors_idx: List[np.ndarray] = []
        neighbors_sim: List[np.ndarray] = []

        for i in range(x_sim.shape[1]):
            row = xtx.getrow(i)
            idx = row.indices
            dot_vals = row.data.astype(np.float64, copy=False)

            if idx.size == 0:
                neighbors_idx.append(np.empty(0, dtype=np.int32))
                neighbors_sim.append(np.empty(0, dtype=np.float32))
                continue

            sim = dot_vals / (norms[i] * norms[idx] + self.eps)
            if self.shrinkage > 0:
                row_counts = co_counts.getrow(i)
                co = self._gather_row_values(row_counts.indices, row_counts.data, idx)
                sim = sim * (co / (co + self.shrinkage + self.eps))

            mask = idx != i
            idx = idx[mask]
            sim = sim[mask]

            if self.min_similarity > 0:
                keep = sim >= self.min_similarity
                idx = idx[keep]
                sim = sim[keep]

            if idx.size > k_neighbors:
                top_pos = np.argpartition(-sim, k_neighbors - 1)[:k_neighbors]
                idx = idx[top_pos]
                sim = sim[top_pos]

            if idx.size > 1:
                order = np.argsort(-sim)
                idx = idx[order]
                sim = sim[order]

            neighbors_idx.append(idx.astype(np.int32, copy=False))
            neighbors_sim.append(sim.astype(np.float32, copy=False))

        return neighbors_idx, neighbors_sim

    def _compute_neighbors_jaccard(self, x: csr_matrix, k_neighbors: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        x_bin = x.copy()
        x_bin.data = np.ones_like(x_bin.data, dtype=np.float32)
        inter = (x_bin.T @ x_bin).tocsr()
        item_counts = np.asarray(x_bin.getnnz(axis=0), dtype=np.float64)

        neighbors_idx: List[np.ndarray] = []
        neighbors_sim: List[np.ndarray] = []

        for i in range(x.shape[1]):
            row = inter.getrow(i)
            idx = row.indices
            inter_vals = row.data.astype(np.float64, copy=False)

            if idx.size == 0:
                neighbors_idx.append(np.empty(0, dtype=np.int32))
                neighbors_sim.append(np.empty(0, dtype=np.float32))
                continue

            union = item_counts[i] + item_counts[idx] - inter_vals
            sim = inter_vals / np.maximum(union, self.eps)
            if self.shrinkage > 0:
                sim = sim * (inter_vals / (inter_vals + self.shrinkage + self.eps))

            mask = idx != i
            idx = idx[mask]
            sim = sim[mask]

            if self.min_similarity > 0:
                keep = sim >= self.min_similarity
                idx = idx[keep]
                sim = sim[keep]

            if idx.size > k_neighbors:
                top_pos = np.argpartition(-sim, k_neighbors - 1)[:k_neighbors]
                idx = idx[top_pos]
                sim = sim[top_pos]

            if idx.size > 1:
                order = np.argsort(-sim)
                idx = idx[order]
                sim = sim[order]

            neighbors_idx.append(idx.astype(np.int32, copy=False))
            neighbors_sim.append(sim.astype(np.float32, copy=False))

        return neighbors_idx, neighbors_sim

    def preprocess(self, train_data: pd.DataFrame, **kwargs):
        df = self._ensure_schema(train_data)

        self.feedback_mode = self._infer_feedback_mode(df[self.label_col])
        self.as_binary = (
            self._to_bool(self.as_binary_cfg, default=(self.feedback_mode != "explicit"))
            if not self._is_auto(self.as_binary_cfg)
            else (self.feedback_mode != "explicit")
        )
        self.use_weighted_ratings = (
            self._to_bool(self.use_weighted_ratings_cfg, default=(self.feedback_mode == "explicit"))
            if not self._is_auto(self.use_weighted_ratings_cfg)
            else (self.feedback_mode == "explicit")
        )
        self.normalize_scores = (
            self._to_bool(self.normalize_scores_cfg, default=self.use_weighted_ratings)
            if not self._is_auto(self.normalize_scores_cfg)
            else self.use_weighted_ratings
        )

        thr = self._resolve_positive_threshold(df[self.label_col], self.feedback_mode)
        self.positive_threshold = thr
        if thr is not None:
            filtered = df[df[self.label_col] > thr].copy()

            if filtered.empty and thr > 0.0:
                fallback = df[df[self.label_col] > 0.0].copy()
                if not fallback.empty:
                    print(
                        f"[ItemKNN] Warning: no interactions with {self.label_col} > {thr}. "
                        "Falling back to threshold>0."
                    )
                    filtered = fallback

            df = filtered

        if df.empty:
            raise ValueError(
                f"[ItemKNN] No interactions with {self.label_col} > {self.positive_threshold}."
            )

        df = (
            df.groupby([self.user_col, self.item_col], as_index=False)[self.label_col]
            .mean()
            .reset_index(drop=True)
        )

        self.global_mean = float(df[self.label_col].mean())
        self.interactions = self._build_interaction_matrix(df)
        self.interactions.sort_indices()

        n_users, n_items = self.interactions.shape
        if n_items < 2:
            raise ValueError("[ItemKNN] At least 2 items are required.")

        if self.k_neighbors is None:
            k_neighbors = int(np.sqrt(n_items))
        else:
            k_neighbors = int(self.k_neighbors)
        k_neighbors = max(1, min(k_neighbors, n_items - 1))
        self.k_neighbors = k_neighbors

        item_support = np.asarray(self.interactions.getnnz(axis=0), dtype=np.float32)
        max_support = float(item_support.max()) if item_support.size else 1.0
        if max_support <= 0:
            max_support = 1.0
        self.item_popularity = (item_support / max_support).astype(np.float32)

        item_means = df.groupby(self.item_col)[self.label_col].mean()
        self.item_mean = np.zeros(n_items, dtype=np.float32)
        for item_id, mean_val in item_means.items():
            idx = self.item2idx.get(str(item_id))
            if idx is not None:
                self.item_mean[idx] = float(mean_val)

        self.user_seen_items = []
        self.user_seen_vals = []
        for u in range(n_users):
            s = self.interactions.indptr[u]
            e = self.interactions.indptr[u + 1]
            self.user_seen_items.append(self.interactions.indices[s:e].astype(np.int32, copy=False))
            self.user_seen_vals.append(self.interactions.data[s:e].astype(np.float32, copy=False))

        if self.similarity_metric == "cosine":
            self.cosine_variant = self._resolve_cosine_variant()
            if self.cosine_variant == "adjusted":
                x_sim = self._build_adjusted_user_centered_matrix(self.interactions)
            else:
                x_sim = self.interactions
            neighbors_idx, neighbors_sim = self._compute_neighbors_cosine(
                x_sim=x_sim,
                x_support=self.interactions,
                k_neighbors=k_neighbors,
            )
        else:
            neighbors_idx, neighbors_sim = self._compute_neighbors_jaccard(self.interactions, k_neighbors)

        self.item_neighbors_idx = neighbors_idx
        self.item_neighbors_sim = neighbors_sim
        self.item_neighbor_map = [
            {int(j): float(s) for j, s in zip(idx, sim)}
            for idx, sim in zip(self.item_neighbors_idx, self.item_neighbors_sim)
        ]

        print(
            f"[ItemKNN] preprocess finished. users={n_users}, items={n_items}, "
            f"k_neighbors={self.k_neighbors}, similarity={self.similarity_metric}, "
            f"cosine_variant={self.cosine_variant}, "
            f"exclude_seen_items={self.exclude_seen_items}, feedback_mode={self.feedback_mode}, "
            f"as_binary={self.as_binary}, weighted={self.use_weighted_ratings}, "
            f"normalize={self.normalize_scores}, threshold={self.positive_threshold}"
        )

    def fit(self):
        if self.interactions is None or not self.item_neighbors_idx:
            raise RuntimeError("Call preprocess() before fit().")
        self.model = True
        print("[ItemKNN] fit finished (memory-based model).")

    def _cold_item_score(self, item_idx: int) -> float:
        if item_idx < 0:
            if self.cold_start_score == "global_mean":
                return float(self.global_mean)
            return 0.0

        if self.cold_start_score == "global_mean":
            return float(self.global_mean)
        if self.cold_start_score == "zero":
            return 0.0
        if self.item_popularity is None:
            return 0.0
        return float(self.item_popularity[item_idx])

    @staticmethod
    def _contains_sorted(arr: np.ndarray, value: int) -> bool:
        if arr.size == 0:
            return False
        pos = int(np.searchsorted(arr, value))
        return bool(pos < arr.size and int(arr[pos]) == int(value))

    def _empty_signal_score(self, item_idx: int) -> float:
        if self.fallback_on_empty_neighbors:
            return self._cold_item_score(item_idx)
        return 0.0

    def _finalize_score(self, score: float, item_idx: int) -> float:
        if score == float("-inf"):
            return score
        if item_idx >= 0 and self.item_popularity is not None:
            score += 1e-9 * float(self.item_popularity[item_idx])
        return float(score)

    def _score_similar_first(self, user_idx: int, item_idx: int) -> float:
        neigh_idx = self.item_neighbors_idx[item_idx]
        neigh_sim = self.item_neighbors_sim[item_idx]
        if neigh_idx.size == 0:
            return self._empty_signal_score(item_idx)

        seen_items = self.user_seen_items[user_idx]
        seen_vals = self.user_seen_vals[user_idx]
        if seen_items.size == 0:
            return self._cold_item_score(item_idx)

        pos = np.searchsorted(seen_items, neigh_idx)
        in_bounds = pos < seen_items.size
        if not np.any(in_bounds):
            return self._empty_signal_score(item_idx)

        valid = np.zeros_like(in_bounds, dtype=bool)
        valid[in_bounds] = seen_items[pos[in_bounds]] == neigh_idx[in_bounds]
        if not np.any(valid):
            return self._empty_signal_score(item_idx)

        sims = neigh_sim[valid].astype(np.float64, copy=False)
        den = float(np.sum(np.abs(sims)))

        if self.use_weighted_ratings:
            ratings = seen_vals[pos[valid]].astype(np.float64, copy=False)
            if self.center_ratings and (not self.as_binary) and self.item_mean is not None:
                neigh_item_means = self.item_mean[neigh_idx[valid]].astype(np.float64, copy=False)
                base = float(self.item_mean[item_idx]) if item_idx >= 0 else float(self.global_mean)
                if den <= self.eps:
                    return base
                num = float(np.dot(sims, ratings - neigh_item_means))
                return base + (num / den)
            num = float(np.dot(sims, ratings))
        else:
            num = float(np.sum(sims))

        if self.normalize_scores and den > self.eps:
            return num / den
        return num

    def _score_all_seen(self, user_idx: int, item_idx: int) -> float:
        seen_items = self.user_seen_items[user_idx]
        seen_vals = self.user_seen_vals[user_idx]
        if seen_items.size == 0:
            return self._cold_item_score(item_idx)

        sim_map = self.item_neighbor_map[item_idx]
        if not sim_map:
            return self._empty_signal_score(item_idx)

        sims: List[float] = []
        vals: List[float] = []
        neigh_items: List[int] = []
        for it, val in zip(seen_items, seen_vals):
            s = sim_map.get(int(it))
            if s is None:
                continue
            sims.append(float(s))
            vals.append(float(val))
            neigh_items.append(int(it))

        if not sims:
            return self._empty_signal_score(item_idx)

        sims_arr = np.asarray(sims, dtype=np.float64)
        vals_arr = np.asarray(vals, dtype=np.float64)
        den = float(np.sum(np.abs(sims_arr)))

        if self.use_weighted_ratings:
            if self.center_ratings and (not self.as_binary) and self.item_mean is not None:
                neigh_means = self.item_mean[np.asarray(neigh_items, dtype=np.int32)].astype(np.float64, copy=False)
                base = float(self.item_mean[item_idx]) if item_idx >= 0 else float(self.global_mean)
                if den <= self.eps:
                    return base
                num = float(np.dot(sims_arr, vals_arr - neigh_means))
                return base + (num / den)
            num = float(np.dot(sims_arr, vals_arr))
        else:
            num = float(np.sum(sims_arr))

        if self.normalize_scores and den > self.eps:
            return num / den
        return num

    def _score_pair(self, user_idx: int, item_idx: int) -> float:
        if item_idx < 0:
            return self._cold_item_score(item_idx)
        if user_idx < 0:
            return self._cold_item_score(item_idx)

        seen_items = self.user_seen_items[user_idx]
        if seen_items.size == 0:
            return self._cold_item_score(item_idx)

        if self.exclude_seen_items and self._contains_sorted(seen_items, item_idx):
            return float("-inf")

        if self.as_similar_first:
            score = self._score_similar_first(user_idx, item_idx)
        else:
            score = self._score_all_seen(user_idx, item_idx)

        return self._finalize_score(float(score), item_idx)

    def prediction(self, data_input: Any, **kwargs) -> Tuple[List[float], List[float]]:
        if self.interactions is None or not self.item_neighbors_idx:
            raise RuntimeError("Call preprocess() and fit() before prediction().")
        if not isinstance(data_input, pd.DataFrame):
            raise ValueError("prediction expects a pandas DataFrame.")

        df = self._ensure_schema(data_input)
        y_true = df[self.label_col].astype(float).tolist()

        u_map = df[self.user_col].map(self.user2idx)
        i_map = df[self.item_col].map(self.item2idx)

        scores = np.zeros(len(df), dtype=np.float64)
        for r in range(len(df)):
            u_idx = int(u_map.iat[r]) if pd.notna(u_map.iat[r]) else -1
            i_idx = int(i_map.iat[r]) if pd.notna(i_map.iat[r]) else -1
            scores[r] = self._score_pair(u_idx, i_idx)

        return y_true, scores.astype(float).tolist()

    def recommend_for_user(self, user_id: Any, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.interactions is None:
            raise RuntimeError("Call preprocess() and fit() before recommend_for_user().")
        top_k = int(top_k)
        if top_k <= 0:
            return []

        u_idx = self.user2idx.get(str(user_id), None)
        if u_idx is None:
            if self.item_popularity is None:
                return []
            top = np.argsort(-self.item_popularity)[: max(1, int(top_k))]
            return [(self.idx2item[int(i)], float(self.item_popularity[int(i)])) for i in top]

        seen = set(self.user_seen_items[u_idx].tolist())
        candidates = [i for i in range(self.interactions.shape[1]) if i not in seen]
        if not candidates:
            return []

        scores = np.asarray([self._score_pair(u_idx, i) for i in candidates], dtype=np.float64)
        if scores.size <= top_k:
            order = np.argsort(-scores)
        else:
            top_pos = np.argpartition(-scores, int(top_k) - 1)[: int(top_k)]
            order = top_pos[np.argsort(-scores[top_pos])]

        ranked = []
        for pos in order:
            item_idx = candidates[int(pos)]
            ranked.append((self.idx2item[item_idx], float(scores[int(pos)])))
        return ranked
