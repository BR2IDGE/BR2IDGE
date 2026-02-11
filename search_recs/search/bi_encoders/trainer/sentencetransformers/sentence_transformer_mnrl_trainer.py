from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import Dataset

from .sentence_transformer_trainer_wrapper import SentenceTransformerTrainerWrapper
from sentence_transformers.trainer import SentenceTransformerTrainer


class SentenceTransformerMnrlTrainer(SentenceTransformerTrainerWrapper):
    def __init__(
        self,
        model_wrapper: SentenceTransformer, 
        train_dataset: Dataset,
        val_dataset: Dataset,
        early_stopping_patience: int,
        early_stopping_threshold: float,
        batch_size: int = 128,
        learning_rate: float = 2e-5,
        num_warmup_factor: float = .1,
        model_dir: str = "model",
        epochs: int = 1,
        logging_steps: int = 200,
        evaluation_steps: int = 200,
        save_steps: int = 200,
    ):
        super().__init__(
            model_wrapper=model_wrapper,    
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_warmup_factor=num_warmup_factor,
            model_dir=model_dir,
            epochs=epochs,
            logging_steps=logging_steps,
            evaluation_steps=evaluation_steps,
            save_steps=save_steps,
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )
        # Use the sentence-transformers loss
        self.train_loss = losses.MultipleNegativesRankingLoss(self.model_wrapper)

        self.trainer = SentenceTransformerTrainer(
            model=self.model_wrapper,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            loss=self.train_loss,
            evaluator=self.evaluator,
        )

    def train(self):
        """
        Fine-tunes the model for semantic search using in-batch negatives.
        """
        self.trainer.train()