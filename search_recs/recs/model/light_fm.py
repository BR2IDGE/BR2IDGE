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

class LightFMTextModel(LightFMModel):
    """LightFM with item content features: an identity feature (weight 1) for every training item plus
    the TF-IDF terms of the item text, scaled to sum ``content_weight``. Rows are not renormalised, so
    an unseen item gets exactly the text contribution a training item gets, only without the identity.

    Same interactions, hyperparameters and BPR negative sampling as LightFMModel; the only change is
    the item representation, so items never seen in training are scored through their text instead of
    getting -inf. It is the control for whether the coverage requirement of the Search -> Rec
    adaptations is specific to pure collaborative filtering.

    Extra parameters: max_text_features (5000), min_df (2), identity_features (true), content_weight
    (1.0). The item text is resolved from the dataset parameters (search_recs.recs.item_text).
    """

    def __init__(self, model_config: dict, features_config: dict = None):
        super().__init__(model_config, features_config)
        params = self.model_params or {}
        self.max_text_features = int(params.get("max_text_features", 5000))
        self.min_df = int(params.get("min_df", 2))
        self.identity_features = bool(params.get("identity_features", True))
        self.content_weight = float(params.get("content_weight", 1.0))
        self._dataset_params = {k: v for k, v in model_config.items() if k != "parameters"}
        self._texts = None
        self._vectorizer = None
        self.item_features = None

    def _feature_rows(self, item_ids) -> "sp.csr_matrix":
        import scipy.sparse as sp

        item_ids = [str(i) for i in item_ids]
        content = self._vectorizer.transform([self._texts.get(i, "") for i in item_ids]).tocsr()
        sums = np.asarray(content.sum(axis=1)).ravel()
        sums[sums == 0] = 1.0
        content = sp.diags(self.content_weight / sums) @ content
        blocks = [content]
        if self.identity_features:
            rows = [r for r, i in enumerate(item_ids) if i in self.item2idx]
            cols = [self.item2idx[item_ids[r]] for r in rows]
            identity = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                     shape=(len(item_ids), len(self.item2idx)))
            blocks = [identity, content]
        return sp.hstack(blocks, format="csr", dtype=np.float32)

    def preprocess(self, train_data: pd.DataFrame, **kwargs):
        super().preprocess(train_data, **kwargs)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from search_recs.recs.item_text import load_item_texts

        self._texts = load_item_texts(self._dataset_params)
        self._vectorizer = TfidfVectorizer(max_features=self.max_text_features, min_df=self.min_df,
                                           stop_words="english", sublinear_tf=True, dtype=np.float32)
        self._vectorizer.fit(list(self._texts.values()))
        train_items = sorted(self.item2idx, key=self.item2idx.get)
        self.item_features = self._feature_rows(train_items)
        no_text = sum(1 for i in train_items if not self._texts.get(i))
        print(f"[LightFMText] item features: {self.item_features.shape[1]} "
              f"({len(train_items) if self.identity_features else 0} identity + "
              f"{len(self._vectorizer.vocabulary_)} terms) | texts for {len(self._texts)} items | "
              f"training items without text: {no_text}")

    def fit(self):
        if self.interactions is None or self.item_features is None:
            raise RuntimeError("Call preprocess(train_data) before fit().")
        params = self.model_params or {}
        self.model = LightFM(
            loss=params.get("loss", "warp"),
            learning_rate=params.get("learning_rate", 0.05),
            no_components=params.get("embedding_dim", 64),
            item_alpha=params.get("item_alpha", 0.0),
            user_alpha=params.get("user_alpha", 0.0),
            random_state=params.get("seed", 42),
        )
        self.model.fit(
            self.interactions,
            item_features=self.item_features,
            epochs=params.get("epochs", 10),
            num_threads=params.get("num_threads", 4),
            verbose=bool(params.get("verbose", 0)),
        )
        print("[LightFMText] Training finished.")

    def prediction(self, test_data: pd.DataFrame) -> tuple:
        if self.model is None:
            raise RuntimeError("Call fit() before prediction().")
        if test_data is None or not isinstance(test_data, pd.DataFrame):
            raise ValueError("Must provide a DataFrame in 'test_data'.")

        u_map = test_data[self.user_col].astype(str).map(self.user2idx)
        items = test_data[self.item_col].astype(str)
        # Feature rows for the candidates of this batch: training items keep their identity feature,
        # unseen items are represented by their text only.
        candidates = pd.unique(items)
        row_of = {item: k for k, item in enumerate(candidates)}
        features = self._feature_rows(candidates)

        y_pred = np.full(len(test_data), -np.inf, dtype=np.float32)
        valid = u_map.notna().to_numpy()
        if valid.any():
            y_pred[valid] = self.model.predict(
                user_ids=u_map[valid].astype(int).to_numpy(),
                item_ids=items[valid].map(row_of).astype(int).to_numpy(),
                item_features=features,
                num_threads=(self.model_params or {}).get("num_threads", 4),
            )

        if self.label_col in test_data.columns:
            y_true = pd.to_numeric(test_data[self.label_col], errors="coerce").fillna(0.0).tolist()
        else:
            y_true = [0.0] * len(test_data)
        return (y_true, y_pred.tolist())


class LightFMContentModel(LightFMTextModel):
    """Content-only LightFM: every item (seen in training or not) is represented only by its text terms,
    so all candidates compete on equal footing. Contrast with LightFMTextModel, whose training items
    also carry an identity feature that unseen items lack."""

    def __init__(self, model_config: dict, features_config: dict = None):
        super().__init__(model_config, features_config)
        self.identity_features = False
