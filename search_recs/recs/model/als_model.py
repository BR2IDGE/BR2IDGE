import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error

from search_recs.recs.model import BaseRecsModel


class ALSModel(BaseRecsModel):

    def __init__(self, model_config: Dict, features_config: Optional[Dict] = None):
        super().__init__(model_config)

        p = self.model_params or {}
        features_config = features_config or {}

        self.n_factors = int(p.get("n_factors", 64))
        self.n_iterations = int(p.get("n_iterations", 20))
        self.lambda_ = float(p.get("lambda_", 0.1))
        self.early_stopping_patience = int(p.get("early_stopping_patience", 3))
        self.seed = int(p.get("seed", 42))
        self.use_biases = bool(p.get("use_biases", True))
        self.weighted_regularization = bool(p.get("weighted_regularization", True))

        self.positive_threshold: float = float(p.get("positive_threshold", 0.0))

        self.prune_threshold: Optional[float] = (
            float(p["prune_threshold"]) if "prune_threshold" in p else None
        )

        self.subsample_fraction = float(p.get("subsample_fraction", 1.0))
        if not 0 < self.subsample_fraction <= 1.0:
            raise ValueError("subsample_fraction must be a positive number not exceeding 1.0")

        self.user_col = features_config.get("user_col") or p.get("user_col") or "user"
        self.item_col = features_config.get("item_col") or p.get("item_col") or "item"

        self.rating_col = (
            features_config.get("label_col")
            or p.get("rating_col")
            or p.get("label_col")
            or "label"
        )
        self.label_col = self.rating_col

        self.global_mean: float = 0.0

        self.user_map: Dict = {}
        self.item_map: Dict = {}

        self.user_inv_map: Dict = {}
        self.item_inv_map: Dict = {}

        self.user2idx: Dict = self.user_map
        self.item2idx: Dict = self.item_map

        self.popular_items: List[int] = []

        self.U: Optional[np.ndarray] = None
        self.V: Optional[np.ndarray] = None
        self.b_u: Optional[np.ndarray] = None
        self.b_i: Optional[np.ndarray] = None

        self._user_indices: List[np.ndarray] = []
        self._user_ratings: List[np.ndarray] = []
        self._item_indices: List[np.ndarray] = []
        self._item_ratings: List[np.ndarray] = []

        print(
            "ALSModel initialized with columns: "
            f"user_col='{self.user_col}', item_col='{self.item_col}', "
            f"label_col='{self.label_col}', positive_threshold={self.positive_threshold}"
        )

    # Helpers
    def _normalize_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols = set(df.columns)
        ren: Dict[str, str] = {}

        def pick(name: str, candidates: Tuple[str, ...]):
            if name in cols:
                return
            for c in candidates:
                if c in cols:
                    ren[c] = name
                    break

        pick(self.user_col, ("userId", "userid", "user_id", "uid", "user"))
        pick(self.item_col, ("movieId", "itemId", "item_id", "iid", "product_id", "item"))
        pick(self.rating_col, ("rating", "score", "label"))

        if ren:
            df = df.rename(columns=ren)
            cols = set(df.columns)

        for c in (self.user_col, self.item_col, self.rating_col):
            if c not in cols:
                print(f"ERROR: required column '{c}' is missing after normalization.")
                raise ValueError(f"Required column '{c}' missing after normalization.")

        return df

    # Preprocess
    def preprocess(self, train_data: pd.DataFrame, **kwargs) -> None:

        print("Starting preprocessing...")
        if train_data is None or not isinstance(train_data, pd.DataFrame):
            raise ValueError("Provide a DataFrame in 'train_data'.")

        train_data = self._normalize_cols(train_data)

        train_data[self.user_col] = train_data[self.user_col].astype(str)
        train_data[self.item_col] = train_data[self.item_col].astype(str)

        train_data[self.rating_col] = pd.to_numeric(train_data[self.rating_col], errors="coerce")

        if self.positive_threshold is not None:
            before = len(train_data)
            train_data = train_data[train_data[self.rating_col] > self.positive_threshold]
            after = len(train_data)
            if after == 0:
                raise ValueError(
                    f"No interactions with {self.rating_col} > {self.positive_threshold} "
                    "were found to train ALS."
                )
            print(
                f"[ALS preprocess] Kept {after} out of {before} interactions "
                f"(label > {self.positive_threshold})."
            )

        self.global_mean = float(train_data[self.rating_col].mean())
        self.popular_items = (
            train_data[self.item_col].value_counts().head(10).index.tolist()
        )

        train_data[self.user_col] = train_data[self.user_col].astype("category")
        train_data[self.item_col] = train_data[self.item_col].astype("category")

        user_cats = train_data[self.user_col].cat.categories
        item_cats = train_data[self.item_col].cat.categories

        self.user_map = {u: i for i, u in enumerate(user_cats)}
        self.item_map = {it: i for i, it in enumerate(item_cats)}
        self.user_inv_map = {i: u for i, u in enumerate(user_cats)}
        self.item_inv_map = {i: it for i, it in enumerate(item_cats)}

        self.user2idx = self.user_map
        self.item2idx = self.item_map

        n_users = len(self.user_map)
        n_items = len(self.item_map)

        train_data["u_idx_int"] = train_data[self.user_col].cat.codes
        train_data["i_idx_int"] = train_data[self.item_col].cat.codes

        self._user_indices = [np.empty(0, dtype=np.int32)] * n_users
        self._user_ratings = [np.empty(0, dtype=np.float32)] * n_users

        self._item_indices = [np.empty(0, dtype=np.int32)] * n_items
        self._item_ratings = [np.empty(0, dtype=np.float32)] * n_items

        for u, group in train_data.groupby("u_idx_int"):
            self._user_indices[u] = group["i_idx_int"].values.astype(np.int32)
            self._user_ratings[u] = group[self.rating_col].values.astype(np.float32)

        for i, group in train_data.groupby("i_idx_int"):
            self._item_indices[i] = group["u_idx_int"].values.astype(np.int32)
            self._item_ratings[i] = group[self.rating_col].values.astype(np.float32)

        print(
            f"[ALS preprocess] n_users={n_users}, n_items={n_items}, "
            f"global_mean={self.global_mean:.4f}"
        )
        print("Preprocessing completed.")

    # Training
    def fit(self, validation_data: Optional[pd.DataFrame] = None, **_) -> None:

        print("Starting model training...")
        n_users = len(self.user_map)
        n_items = len(self.item_map)
        if n_users == 0 or n_items == 0:
            print("ERROR: no training data provided. Call preprocess() first.")
            raise RuntimeError("No training data provided. Call preprocess() first.")

        rng = np.random.default_rng(self.seed)
        self.U = (rng.standard_normal((n_users, self.n_factors)) * 0.05).astype(np.float32)
        self.V = (rng.standard_normal((n_items, self.n_factors)) * 0.05).astype(np.float32)
        self.b_u = np.zeros(n_users, dtype=np.float32)
        self.b_i = np.zeros(n_items, dtype=np.float32)

        val_df = None
        if validation_data is not None and len(validation_data) > 0:
            val_df = self._normalize_cols(validation_data)

        I_f = np.eye(self.n_factors, dtype=np.float32)

        best_val = np.inf
        patience = 0
        best_snapshot: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = None

        if self.weighted_regularization:
            n_ratings_u = np.array([len(idx) for idx in self._user_indices], dtype=np.int32)
            n_ratings_i = np.array([len(idx) for idx in self._item_indices], dtype=np.int32)
        else:
            n_ratings_u = np.ones(n_users, dtype=np.int32)
            n_ratings_i = np.ones(n_items, dtype=np.int32)

        for it in range(1, self.n_iterations + 1):
            for u in range(n_users):
                idx = self._user_indices[u]
                if idx.size == 0:
                    continue

                if self.subsample_fraction < 1.0 and idx.size > 1:
                    sample_size = max(1, int(idx.size * self.subsample_fraction))
                    sample_idx = np.sort(rng.choice(idx, size=sample_size, replace=False))
                    mask = np.isin(idx, sample_idx)
                    V_u = self.V[sample_idx]
                    r_u = self._user_ratings[u][mask]
                    b_i = self.b_i[sample_idx] if self.use_biases else 0.0
                    reg = self.lambda_ * n_ratings_u[u]
                else:
                    V_u = self.V[idx]
                    r_u = self._user_ratings[u]
                    b_i = self.b_i[idx] if self.use_biases else 0.0
                    reg = self.lambda_ * n_ratings_u[u]

                A = V_u.T @ V_u + reg * I_f
                y = V_u.T @ (r_u - self.global_mean - b_i)
                self.U[u, :] = np.linalg.solve(A, y)

            for i in range(n_items):
                idx = self._item_indices[i]
                if idx.size == 0:
                    continue

                if self.subsample_fraction < 1.0 and idx.size > 1:
                    sample_size = max(1, int(idx.size * self.subsample_fraction))
                    sample_idx = np.sort(rng.choice(idx, size=sample_size, replace=False))
                    mask = np.isin(idx, sample_idx)
                    U_i = self.U[sample_idx]
                    r_i = self._item_ratings[i][mask]
                    b_u = self.b_u[sample_idx] if self.use_biases else 0.0
                    reg = self.lambda_ * n_ratings_i[i]
                else:
                    U_i = self.U[idx]
                    r_i = self._item_ratings[i]
                    b_u = self.b_u[idx] if self.use_biases else 0.0
                    reg = self.lambda_ * n_ratings_i[i]

                A = U_i.T @ U_i + reg * I_f
                y = U_i.T @ (r_i - self.global_mean - b_u)
                self.V[i, :] = np.linalg.solve(A, y)

            if self.use_biases:
                for u in range(n_users):
                    idx = self._user_indices[u]
                    if idx.size == 0:
                        continue
                    pred = self.U[u, :] @ self.V[idx, :].T
                    resid = self._user_ratings[u] - self.global_mean - self.b_i[idx] - pred
                    self.b_u[u] = float(resid.mean())

                for i in range(n_items):
                    idx = self._item_indices[i]
                    if idx.size == 0:
                        continue
                    pred = self.U[idx, :] @ self.V[i, :].T
                    resid = self._item_ratings[i] - self.global_mean - self.b_u[idx] - pred
                    self.b_i[i] = float(resid.mean())

            if self.prune_threshold is not None and self.prune_threshold > 0.0:
                thr = float(self.prune_threshold)
                self.U[np.abs(self.U) < thr] = 0.0
                self.V[np.abs(self.V) < thr] = 0.0

            train_rmse = self._rmse_sample()

            if val_df is not None:
                y_true, y_pred = self.prediction(val_df, verbose=False)
                val_rmse = float(np.sqrt(mean_squared_error(np.asarray(y_true), np.asarray(y_pred))))
            else:
                val_rmse = train_rmse

            print(
                f"[ALS] iter {it}/{self.n_iterations} | "
                f"train_sample={train_rmse:.4f} | val={val_rmse:.4f}"
            )

            if val_rmse + 1e-6 < best_val:
                best_val = val_rmse
                patience = 0
                best_snapshot = (
                    self.U.copy(),
                    self.V.copy(),
                    self.b_u.copy(),
                    self.b_i.copy(),
                    self.global_mean,
                )
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    print(f"[ALS] early stopping at iteration {it} (best val RMSE={best_val:.4f})")
                    break

        if best_snapshot is not None:
            self.U, self.V, self.b_u, self.b_i, self.global_mean = best_snapshot

        self.model = True  
        print("Training completed.")


    def _rmse_sample(self, n_samples: int = 10000) -> float:
        if len(self._user_indices) == 0:
            return 0.0

        valid_users = [u for u in range(len(self._user_indices)) if len(self._user_indices[u]) > 0]
        if not valid_users:
            return 0.0

        sample_u = np.random.choice(valid_users, size=min(n_samples, len(valid_users)))
        sq_err = 0.0
        count = 0

        for u in sample_u:
            idx = self._user_indices[u]
            rnd = np.random.randint(len(idx))
            i = idx[rnd]
            real = self._user_ratings[u][rnd]

            base = self.global_mean
            if self.use_biases:
                base += self.b_u[u] + self.b_i[i]
            pred = base + self.U[u, :] @ self.V[i, :].T

            sq_err += (real - pred) ** 2
            count += 1

        return float(np.sqrt(sq_err / max(count, 1)))

    def _rmse_full(self) -> float:
        sq_err = 0.0
        count = 0
        for u, idx in enumerate(self._user_indices):
            if idx.size == 0:
                continue
            preds = self.U[u, :] @ self.V[idx, :].T
            base = self.global_mean + self.b_u[u] + self.b_i[idx] if self.use_biases else self.global_mean
            diff = self._user_ratings[u] - base - preds
            sq_err += float(np.sum(diff ** 2))
            count += len(diff)
        return float(np.sqrt(sq_err / max(count, 1)))
    

    def prediction(self, test_data: pd.DataFrame, verbose: bool = True) -> Tuple[List[float], List[float]]:

        if self.U is None or self.V is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        test_data = self._normalize_cols(test_data)

        test_data[self.user_col] = test_data[self.user_col].astype(str)
        test_data[self.item_col] = test_data[self.item_col].astype(str)

        y_true: List[float] = []
        y_pred: List[float] = []

        u_arr = test_data[self.user_col].to_numpy()
        i_arr = test_data[self.item_col].to_numpy()
        r_arr = pd.to_numeric(test_data[self.rating_col], errors="coerce").to_numpy()

        for u_raw, i_raw, y in zip(u_arr, i_arr, r_arr):
            u = self.user_map.get(u_raw)
            i = self.item_map.get(i_raw)
            if u is None or i is None:
                pred = self.global_mean
            else:
                base = self.global_mean
                if self.use_biases:
                    base += float(self.b_u[u]) + float(self.b_i[i])
                pred = base + float(self.U[u, :] @ self.V[i, :].T)
            y_true.append(float(y))
            y_pred.append(float(pred))
        return y_true, y_pred


    def recommend_for_user(self, user_id, top_k: int = 10) -> List[Tuple[int, float]]:

        if self.U is None or self.V is None:
            raise RuntimeError("Model not trained.")

        user_id = str(user_id)

        u = self.user_map.get(user_id)
        if u is None:
            return [(it, 0.0) for it in self.popular_items[:top_k]]

        base = self.global_mean + (self.b_u[u] if self.use_biases else 0.0)
        if self.use_biases:
            scores = base + self.b_i + (self.V @ self.U[u, :].T)
        else:
            scores = base + (self.V @ self.U[u, :].T)

        scores = np.asarray(scores, dtype=np.float32)

        seen = self._user_indices[u]
        scores[seen] = -np.inf

        if top_k >= len(scores):
            top_idx = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, top_k)[:top_k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [(self.item_inv_map[i], float(scores[i])) for i in top_idx]