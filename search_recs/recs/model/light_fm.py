import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from lightfm import LightFM
from search_recs.recs.model import BaseRecsModel

class LightFMModel(BaseRecsModel):
    def __init__(self, model_config: dict, features_config: dict):
        super().__init__(model_config)

        features_config = features_config or {}

        # Enforce framework defaults
        self.user_col = "user"
        self.item_col = "item"
        self.label_col = "label"

        # Mappings
        self.user2idx = {}
        self.item2idx = {}
        
        self.interactions = None
        self.model = None

    def _build_indexers(self, train_df: pd.DataFrame):
        # Ensure string type to avoid issues
        users = train_df[self.user_col].astype(str).unique()
        items = train_df[self.item_col].astype(str).unique()

        self.user2idx = {u: i for i, u in enumerate(users)}
        self.item2idx = {m: i for i, m in enumerate(items)}

    def _to_coo(self, df: pd.DataFrame):
        # Vectorized and safe mapping
        u_ids = df[self.user_col].astype(str).map(self.user2idx)
        i_ids = df[self.item_col].astype(str).map(self.item2idx)

        # Filter known interactions for training matrix
        mask = u_ids.notna() & i_ids.notna()
        
        if not mask.any():
            return None, mask

        rows = u_ids[mask].astype(int).values
        cols = i_ids[mask].astype(int).values

        if self.label_col in df.columns:
            data = pd.to_numeric(df.loc[mask, self.label_col], errors='coerce').fillna(1.0).values
        else:
            data = np.ones(len(rows), dtype=float)

        shape = (len(self.user2idx), len(self.item2idx))
        return coo_matrix((data, (rows, cols)), shape=shape), mask

    def preprocess(self, train_data: pd.DataFrame, **kwargs):
        """Prepares mappings and interaction matrix."""
        if train_data is None or not isinstance(train_data, pd.DataFrame):
            raise ValueError("Must provide a DataFrame in 'train_data'.")

        self._build_indexers(train_data)
        self.interactions, _ = self._to_coo(train_data)
        
        if self.interactions is None:
            print("[LightFM] Warning: Empty interaction matrix.")
            
        print(f"[LightFM] Preprocessing finished. Vocabulary: {len(self.user2idx)} users, {len(self.item2idx)} items.")

    def fit(self):
        """Trains the model."""
        if self.interactions is None:
            raise RuntimeError("Call preprocess(train_data) before fit().")

        loss = self.model_params.get("loss", "warp") 
        learning_rate = self.model_params.get("learning_rate", 0.05)
        no_components = self.model_params.get("embedding_dim", 64)
        random_state = self.model_params.get("seed", 42)

        self.model = LightFM(
            loss=loss,
            learning_rate=learning_rate,
            no_components=no_components,
            item_alpha=self.model_params.get("item_alpha", 0.0),
            user_alpha=self.model_params.get("user_alpha", 0.0),
            random_state=random_state,
        )

        epochs = self.model_params.get("epochs", 10)
        num_threads = self.model_params.get("num_threads", 4)
        verbose = bool(self.model_params.get("verbose", 0))

        self.model.fit(
            self.interactions,
            epochs=epochs,
            num_threads=num_threads,
            verbose=verbose
        )
        print("[LightFM] Training finished.")

    def prediction(self, test_data: pd.DataFrame) -> tuple:
        if self.model is None:
            raise RuntimeError("Call fit() before prediction().")

        if test_data is None or not isinstance(test_data, pd.DataFrame):
            raise ValueError("Must provide a DataFrame in 'test_data'.")

        u_map = test_data[self.user_col].astype(str).map(self.user2idx)
        i_map = test_data[self.item_col].astype(str).map(self.item2idx)

        n_samples = len(test_data)
        y_pred = np.full(n_samples, -np.inf, dtype=np.float32)

        valid_mask = u_map.notna() & i_map.notna()
        
        if valid_mask.any():
            u_valid = u_map[valid_mask].astype(int).values
            i_valid = i_map[valid_mask].astype(int).values
            
            scores = self.model.predict(
                user_ids=u_valid, 
                item_ids=i_valid, 
                num_threads=self.model_params.get("num_threads", 4)
            )
            y_pred[valid_mask] = scores

        if self.label_col in test_data.columns:
            y_true = pd.to_numeric(test_data[self.label_col], errors='coerce').fillna(0.0).tolist()
        else:
            y_true = [0.0] * n_samples

        return (y_true, y_pred.tolist())