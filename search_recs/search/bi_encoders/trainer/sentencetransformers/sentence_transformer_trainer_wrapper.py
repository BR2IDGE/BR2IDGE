from abc import ABC

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from transformers import EarlyStoppingCallback # Add this import to your file
from torch.utils.data import DataLoader, Dataset


class SentenceTransformerTrainerWrapper(ABC):
    """
    This class wraps the Sentence-BERT training process using the
    built-in functionality of the Sentence-Transformer library.
    """

    def __init__(
        self,
        model_wrapper: SentenceTransformer, # Rename it to model_wrapper
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int = 128,
        learning_rate: float = 2e-5,
        num_warmup_factor: float = .1,
        model_dir: str = "model",
        epochs: int = 1,
        logging_steps: int = 200,
        evaluation_steps: int = 200,
        save_steps: int = 200,
        early_stopping_patience: int = 5,
        early_stopping_threshold: float = .0,
    ):
        """
        Args:
            model_wrapper: TorchVectorModel to be trained
            train_dataset: training dataset
            val_dataset: validation dataset
            batch_size: number of batches considered to create the dataloader
            learning_rate: initial learning rate to use in the optimizer
            num_warmup_factor: factor for the learning rate warmup
            model_dir: path to save the model
        """
        self.model_wrapper = model_wrapper  # Rename it to model_wrapper
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_warmup_factor = num_warmup_factor
        self.model_dir = model_dir
        self.epochs = epochs
        self.logging_steps = logging_steps
        self.evaluation_steps = evaluation_steps
        self.save_steps = save_steps
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_thresholdt = early_stopping_threshold

        inner_sbert_model = self.model_wrapper 

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=inner_sbert_model.smart_batching_collate,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=inner_sbert_model.smart_batching_collate,
        )

        self.callbacks_list = []

        try:
            self.evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                self.val_dataloader, 
                name='validation',
            )
            self.early_stop_callback = EarlyStoppingCallback(
                early_stopping_patience=self.early_stopping_patience, 
                early_stopping_threshold=self.early_stopping_threshold
            )
            self.load_best_model_at_end = True
            self.callbacks_list.append(self.early_stop_callback)
        except Exception:
            # Handle case where validation data isn't in Example format (e.g., just triples)
            self.evaluator = None # You must set load_best_model_at_end=False if this happens
            self.load_best_model_at_end = False
            self.early_stop_callback = None

        self.training_args = SentenceTransformerTrainingArguments(
            output_dir=self.model_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            warmup_ratio=self.num_warmup_factor,
            logging_steps=self.logging_steps,
            eval_strategy="steps",
            eval_steps=self.evaluation_steps,
            save_steps=self.save_steps,
            load_best_model_at_end=self.load_best_model_at_end,
            report_to=["tensorboard"],
            fp16=True,
            dataloader_drop_last=True,
        )


    def train(self):
        """
        Fine-tunes the SentenceTransformer model using the built-in fit method.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")