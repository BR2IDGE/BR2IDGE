from search_recs.recs.model import BaseRecsModel
from libreco.data import DatasetFeat
from libreco.algorithms import DeepFM
import pandas as pd
import numpy as np
import tensorflow as tf

MODELS_MAP = {
    "DeepFM": DeepFM
}

class DeepFMModel(BaseRecsModel):

    def __init__(self, model_config: dict, features_config: dict):
        tf.compat.v1.reset_default_graph()
        super().__init__(model_config)
        
        self.model_type = MODELS_MAP.get(self.model_name)
        if self.model_type is None:
            raise ValueError(f"Model '{self.model_name}' not found in MODELS_MAP for LibReco.")
        
        self.sparse_col = features_config.get("sparse_col")
        self.dense_col = features_config.get("dense_col")
        self.user_col = features_config.get("user_col")
        self.item_col = features_config.get("item_col")
        
        print("DeepFM initialized.")
        
    def preprocess(self, train_data: pd.DataFrame, **kwargs):
        print("LibReCo: Starting preprocessing...")
        self.train_dataset, self.data_info = DatasetFeat.build_trainset(
            train_data, 
            self.user_col, 
            self.item_col, 
            self.sparse_col, 
            self.dense_col
        )
        print("LibReCo: Preprocessing finished.")

    def fit(self):
        print(f"LibReCo: Starting training (fit) for {self.model_name}...")
        
        # Initialize model with parameters
        self.model = self.model_type(
            task="ranking",
            data_info=self.data_info,
            embed_size=self.model_params.get("embed_size", 32),
            n_epochs=self.model_params.get("n_epochs", 10),
            lr=self.model_params.get("lr", 1e-4),
            batch_size=self.model_params.get("batch_size", 32),
            use_bn=self.model_params.get("use_bn", True),
            hidden_units=self.model_params.get("hidden_units", (128, 64, 32)),
        )

        # Train model
        self.model.fit(
            self.train_dataset,
            neg_sampling=self.model_params.get("neg_sampling", True),
            verbose=self.model_params.get("verbose", 2),
            shuffle=self.model_params.get("shuffle", True),
        )
        print(f"LibReCo: Training finished.")

    def prediction(self, test_data: pd.DataFrame):
        # Get true labels
        y_test = test_data['label'].values.astype(np.float32)   
        
        # Batch prediction
        y_pred = self.model.predict(
            user=test_data['user'].values,
            item=test_data['item'].values
        )

        # Convert to list if necessary
        if isinstance(y_pred, np.ndarray):
            y_pred = y_pred.tolist()

        return (list(y_test), y_pred)