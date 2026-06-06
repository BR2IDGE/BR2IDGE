import os
import shutil
import pandas as pd
import numpy as np
import torch
from typing import Optional
from pathlib import Path

from search_recs.recs.model import BaseRecsModel
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import BERT4Rec
from recbole.trainer import Trainer
from recbole.data.interaction import Interaction

class Bert4REC(BaseRecsModel):
    """
    Wrapper for RecBole's BERT4Rec implementation.
    Handles data conversion to .inter format and sequential inference.
    """
    
    def __init__(self, model_config: dict, features_config: Optional[dict] = None, **_):
        super().__init__(model_config)
        self.feature_params = features_config or {}
        
        # Hyperparameters
        self.epochs = int(self.model_params.get("epochs", 50))
        self.train_batch_size = int(self.model_params.get("train_batch_size", 256))
        self.embedding_size = int(self.model_params.get("hidden_size", 64))
        self.max_seq_len = int(self.model_params.get("max_seq_len", 50))
        
        # Internal state
        self.config = None
        self.dataset = None
        self.train_data = None
        self.model = None
        self.user_history = {} 
        
        print("[Bert4REC] Initialized.")

    def preprocess(self, train_data: pd.DataFrame, **kwargs):
        """
        Converts framework data to RecBole format and builds user history.
        """
        print("[Bert4Rec] Preprocessing data...")
        
        self.data_root = str((Path.cwd() / "recbole_temp").resolve())
        self.dataset_name = f"run_{id(self)}"
        ds_dir = os.path.join(self.data_root, self.dataset_name)
        
        if os.path.exists(ds_dir): 
            shutil.rmtree(ds_dir)
        os.makedirs(ds_dir, exist_ok=True)
        
        time_col = 'time' if 'time' in train_data.columns else 'timestamp'
        
        inter_path = os.path.join(ds_dir, f"{self.dataset_name}.inter")
        train_data[['user', 'item', 'label', time_col]].to_csv(
            inter_path, sep='\t', index=False, 
            header=["user:token", "item:token", "label:float", "time:float"]
        )

        config_dict = {
            "data_path": self.data_root,
            "dataset": self.dataset_name,
            "gpu_id": 0 if torch.cuda.is_available() else -1,
            "USER_ID_FIELD": "user",
            "ITEM_ID_FIELD": "item",
            "TIME_FIELD": "time",
            "RATING_FIELD": "label",
            "load_col": {"inter": ["user", "item", "label", "time"]},
            "MAX_ITEM_LIST_LENGTH": self.max_seq_len,
            "eval_args": {
                "split": {"RS": [1.0, 0.0, 0.0]}, 
                "order": "TO", 
                "mode": "full"
            },
            
            # --- Stability Configuration ---
            "learning_rate": 0.0005,          # Reduced to avoid spikes
            "optimizer": "adam",
            "adam_epsilon": 1e-8,             # Scientific standard
            "grad_clip": {"max_norm": 1.0, "norm_type": 2}, # Prevent explosion
            "weight_decay": 0.01,             
            "initializer_range": 0.02,        # Small init (BERT standard)
            # -------------------------------

            "loss_type": "CE",                
            "train_neg_sample_args": None,    # BERT4Rec uses Cloze Task (masking)
            "epochs": self.epochs,
            "train_batch_size": self.train_batch_size,
            "hidden_size": self.embedding_size,
            "show_progress": False,
        }
        
        self.config = Config(model='BERT4Rec', dataset=self.dataset_name, config_dict=config_dict)
        self.dataset = create_dataset(self.config)
        self.train_data, _, _ = data_preparation(self.config, self.dataset)
        
        u_map = self.dataset.field2token_id['user']
        i_map = self.dataset.field2token_id['item']
        
        # Build history strictly with training data
        for uid, group in train_data.sort_values(time_col).groupby('user'):
            u_int = u_map.get(str(uid), 0)
            if u_int != 0:
                # Convert items to internal RecBole IDs, ignoring unknowns (ID 0)
                self.user_history[u_int] = [
                    i_map.get(str(iid), 0) for iid in group['item'].values 
                    if i_map.get(str(iid), 0) != 0
                ]
                
        print("[Bert4Rec] Preprocessing finished.")

    def fit(self):
        """Trains the RecBole model."""
        print("[Bert4Rec] Starting training (fit)...")
        if self.train_data is None:
            raise RuntimeError("Call preprocess() before fit().")

        self.model = BERT4Rec(self.config, self.dataset).to(self.config['device'])
        trainer = Trainer(self.config, self.model)
        trainer.fit(self.train_data, show_progress=True)
        print("[Bert4Rec] Training finished.")

    def prediction(self, data_input: pd.DataFrame, **kwargs) -> tuple[list[float], list[float]]:
        """
        Predicts scores for user-item pairs using historical sequences.
        """
        device = self.config['device']
        u_map = self.dataset.field2token_id['user']
        i_map = self.dataset.field2token_id['item']

        # Map current batch users/items
        u_ids = [u_map.get(str(u), 0) for u in data_input['user'].values]
        i_ids = [i_map.get(str(i), 0) for i in data_input['item'].values]
        
        curr_len = len(u_ids)
        seqs = np.zeros((curr_len, self.max_seq_len), dtype=np.int64)
        lengths = np.zeros(curr_len, dtype=np.int64)
        
        # Build sequences from history
        for idx, uid in enumerate(u_ids):
            if uid == 0: 
                continue
            # Get last N items
            hist = self.user_history.get(uid, [])[-self.max_seq_len:]
            if hist:
                seqs[idx, :len(hist)] = hist
                lengths[idx] = len(hist)

        self.model.eval()
        with torch.no_grad():
            # Construct RecBole Interaction object manually
            inter_dict = {
                self.dataset.uid_field: torch.tensor(u_ids, device=device),
                self.dataset.iid_field: torch.tensor(i_ids, device=device),
                self.model.ITEM_SEQ: torch.tensor(seqs, device=device),
                self.model.ITEM_SEQ_LEN: torch.tensor(lengths, device=device)
            }
            
            inter = Interaction(inter_dict)
            scores = self.model.predict(inter).cpu().numpy()

        # Handle OOV items (assign minimum score)
        scores[np.array(i_ids) == 0] = -1e9
        
        y_true = data_input['label'].values.astype(float) if 'label' in data_input.columns else np.zeros(curr_len)
        return y_true.tolist(), scores.tolist()