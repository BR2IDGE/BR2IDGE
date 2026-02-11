import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List
from sklearn.decomposition import NMF

from search_recs.recs.model import BaseRecsModel


class NMFRecommender(BaseRecsModel):

    def __init__(self, model_config: Dict[str, Any], features_config: Optional[dict] = None, **_: Any) -> None:
        super().__init__(model_config)
        self.feature_params = features_config or {}
        params = self.model_params or {}

        self.n_components: int = int(params.get("n_components", 64))
        self.solver: str = params.get("solver", "mu")
        self.beta_loss: str = params.get("beta_loss", "kullback-leibler")
        self.init: str = params.get("init", "nndsvd")
        self.max_iter: int = int(params.get("max_iter", 200))
        self.tol: float = float(params.get("tol", 1e-4))
        self.random_state: Optional[int] = params.get("random_state", None)
        self.alpha_W: float = float(params.get("alpha_W", 0.0))
        self.alpha_H: float = float(params.get("alpha_H", 0.0))  
        self.l1_ratio: float = float(params.get("l1_ratio", 0.0))
        self.n_runs: int = int(params.get("n_runs", 1))
        self.verbose: bool = bool(params.get("verbose", True))

        self.normalize_ratings: bool = bool(params.get("normalize_ratings", False))
        self.positive_threshold: float = float(params.get("positive_threshold", 0.0)) 

        if self.solver == "mu" and self.beta_loss == "frobenius":
            self.beta_loss = "kullback-leibler"

        self._W: Optional[np.ndarray] = None
        self._H: Optional[np.ndarray] = None

        self._user_map: Dict[str, int] = {}
        self._item_map: Dict[str, int] = {}
        self._user_inv_map: Dict[int, str] = {}
        self._item_inv_map: Dict[int, str] = {}

        self._user_seen: Dict[int, np.ndarray] = {}

        self.popular_items: List[str] = []

        self.global_mean: float = 0.0
        self.b_u: Optional[np.ndarray] = None
        self.b_i: Optional[np.ndarray] = None

        self._train_matrix_sparse = None

    # Helpers
    def _ensure_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("train_data/test_data must be a DataFrame.")
        if "user" not in df.columns or "item" not in df.columns:
            raise ValueError("DataFrame must have 'user' and 'item' columns.")
        out = df.copy()
        out["user"] = out["user"].astype(str)
        out["item"] = out["item"].astype(str)
        if "label" in out.columns:
            out["label"] = pd.to_numeric(out["label"], errors="coerce")
        return out

    def _normalize_labels_nonneg(self, s: pd.Series) -> np.ndarray:

        x = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float).to_numpy()

        if self.positive_threshold is not None:
            x = np.where(x > self.positive_threshold, x, 0.0)

        if self.normalize_ratings:
            mx = float(np.max(x)) if len(x) else 0.0
            if mx > 0:
                x = x / mx

        x = np.clip(x, 0.0, None)
        return x.astype(np.float32)

    # Preprocess
    def preprocess(self, train_data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        if self.verbose:
            print("[NMF] preprocess: starting...")

        df = self._ensure_schema(train_data)

        self.popular_items = df["item"].value_counts().head(50).index.astype(str).tolist()

        df["user"] = df["user"].astype("category")
        df["item"] = df["item"].astype("category")

        user_cats = df["user"].cat.categories
        item_cats = df["item"].cat.categories

        self._user_map = {u: i for i, u in enumerate(user_cats)}
        self._item_map = {it: j for j, it in enumerate(item_cats)}
        self._user_inv_map = {i: u for i, u in enumerate(user_cats)}
        self._item_inv_map = {j: it for j, it in enumerate(item_cats)}

        n_users = len(self._user_map)
        n_items = len(self._item_map)

        u_idx = df["user"].cat.codes.to_numpy(dtype=np.int32)
        i_idx = df["item"].cat.codes.to_numpy(dtype=np.int32)

        if "label" in df.columns:
            ratings = self._normalize_labels_nonneg(df["label"])
        else:
            ratings = np.ones(len(df), dtype=np.float32)

        self.global_mean = float(np.mean(ratings)) if len(ratings) else 0.0

        self.b_u = np.zeros(n_users, dtype=np.float32)
        self.b_i = np.zeros(n_items, dtype=np.float32)

        self._user_seen = {}
        tmp = pd.DataFrame({"u": u_idx, "i": i_idx, "r": ratings})
        for u, g in tmp.groupby("u", sort=False):
            self._user_seen[int(u)] = g["i"].to_numpy(dtype=np.int32)

        resid = ratings - self.global_mean
        for u, g in tmp.groupby("u", sort=False):
            self.b_u[int(u)] = float(np.mean(resid[g.index])) if len(g) else 0.0

        for i, g in tmp.groupby("i", sort=False):
            self.b_i[int(i)] = float(np.mean(resid[g.index])) if len(g) else 0.0

        try:
            from scipy.sparse import coo_matrix
            self._train_matrix_sparse = coo_matrix(
                (ratings, (u_idx, i_idx)),
                shape=(n_users, n_items),
                dtype=np.float32,
            ).tocsr()
        except ImportError:
            raise ImportError("scipy is required for NMFRecommender (sparse matrices).")

        if self.verbose:
            density = self._train_matrix_sparse.nnz / (n_users * n_items) if n_users and n_items else 0.0
            print(
                f"[NMF] matrix: {n_users} users x {n_items} items | "
                f"nnz={self._train_matrix_sparse.nnz} | dens={density:.6f}"
            )
            print(
                f"[NMF] global_mean={self.global_mean:.6f} | "
                f"normalize_ratings={self.normalize_ratings} | "
                f"positive_threshold={self.positive_threshold}"
            )

        return {"matrix_shape": (n_users, n_items)}

    # Fit
    def fit(self) -> None:
        if self.verbose:
            print("[NMF] fit: starting...")

        if self._train_matrix_sparse is None:
            raise RuntimeError("Call preprocess() before fit().")

        best_error = float("inf")
        best_W, best_H = None, None
        base_seed = self.random_state

        for run in range(max(1, self.n_runs)):
            seed = int(base_seed) + run if base_seed is not None else None

            nmf = NMF(
                n_components=self.n_components,
                init=self.init,
                solver=self.solver,
                beta_loss=self.beta_loss,
                max_iter=self.max_iter,
                tol=self.tol,
                alpha_W=self.alpha_W,
                alpha_H=self.alpha_H,
                l1_ratio=self.l1_ratio,
                random_state=seed,
                verbose=0,
            )

            W = nmf.fit_transform(self._train_matrix_sparse)
            H = nmf.components_
            err = float(nmf.reconstruction_err_)

            if self.verbose:
                print(f"[NMF] run {run + 1}/{self.n_runs}: reconstruction_err={err:.6f}")

            if err < best_error:
                best_error = err
                best_W, best_H = W, H

        self._W = best_W.astype(np.float32)
        self._H = best_H.astype(np.float32)

        if self.verbose:
            print("[NMF] fit: done.")

    # Prediction
    def prediction(self, data_input: Any, **kwargs) -> Tuple[List[float], List[float]]:

        if self._W is None or self._H is None:
            raise RuntimeError("Call fit() before prediction().")

        if not isinstance(data_input, pd.DataFrame):
            raise ValueError("[NMF] prediction expects a DataFrame with user/item columns.")

        df = self._ensure_schema(data_input)

        if "label" in df.columns:
            y_true = pd.to_numeric(df["label"], errors="coerce").fillna(0.0).astype(float).tolist()
        else:
            y_true = [0.0] * len(df)

        u_idxs = df["user"].map(self._user_map)
        i_idxs = df["item"].map(self._item_map)
        valid = u_idxs.notna() & i_idxs.notna()

        preds = np.full(len(df), self.global_mean, dtype=np.float32)

        if self.b_u is not None and valid.any():
            u_known = u_idxs[valid].astype(int).to_numpy()
            preds[valid.to_numpy()] += self.b_u[u_known]

        if self.b_i is not None and valid.any():
            i_known = i_idxs[valid].astype(int).to_numpy()
            preds[valid.to_numpy()] += self.b_i[i_known]

        if valid.any():
            u = u_idxs[valid].astype(int).to_numpy()
            i = i_idxs[valid].astype(int).to_numpy()

            w_vecs = self._W[u]     
            h_vecs = self._H.T[i]    
            preds[valid.to_numpy()] += (w_vecs * h_vecs).sum(axis=1).astype(np.float32)

        if self.popular_items:
            pop_set = set(self.popular_items[:50])
            cold_mask = ~valid
            if cold_mask.any():
                boost = np.array(
                    [0.01 if it in pop_set else 0.0 for it in df.loc[cold_mask, "item"].tolist()],
                    dtype=np.float32,
                )
                preds[cold_mask.to_numpy()] += boost

        return y_true, preds.astype(float).tolist()

    def recommend_for_user(self, user_id: Any, top_k: int = 10) -> List[Tuple[str, float]]:

        if self._W is None or self._H is None:
            raise RuntimeError("Call fit() before recommend_for_user().")

        user_id = str(user_id)
        u = self._user_map.get(user_id)

        if u is None:
            return [(it, 0.0) for it in self.popular_items[:top_k]]

        scores = (self._W[u] @ self._H).astype(np.float32) 
        scores = scores + self.global_mean
        if self.b_u is not None:
            scores = scores + self.b_u[u]
        if self.b_i is not None:
            scores = scores + self.b_i

        seen = self._user_seen.get(int(u))
        if seen is not None and len(seen) > 0:
            scores[seen] = -np.inf

        k = min(int(top_k), len(scores))
        if k <= 0:
            return []

        if k >= len(scores):
            top_idx = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, k)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [(self._item_inv_map[int(i)], float(scores[int(i)])) for i in top_idx]