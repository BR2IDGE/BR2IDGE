import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split
from .base_dataloader import BaseSearchDatasetBuilder, BuildConfig

class GoodbooksBuilder(BaseSearchDatasetBuilder):
    def __init__(self, books_path: str, book_tags_path: str, tags_path: str, cfg: BuildConfig = BuildConfig()):
        super().__init__(cfg)
        self.books_path = books_path
        self.book_tags_path = book_tags_path
        self.tags_path = tags_path
        self.books = None
        self.book_tags = None
        self.tags = None

    def load_raw(self) -> None:
        self.books = pd.read_csv(self.books_path)
        self.book_tags = pd.read_csv(self.book_tags_path)
        self.tags = pd.read_csv(self.tags_path)

    @staticmethod
    def _create_document(row: dict) -> str:
        title = str(row.get("title", "")).strip()
        authors = str(row.get("authors", "")).strip()
        year = row.get("original_publication_year")
        parts = [title, authors]
        try:
            import pandas as pd
            if pd.notna(year):
                parts.append(str(int(float(year))))
        except Exception:
            pass
        return "\n".join([p for p in parts if p])

    @staticmethod
    def _create_category(row: dict) -> str:
        return str(row.get("authors", "")).strip() or "unknown"


def load_goodbooks_dataset(
    cfg: BuildConfig,
):
    """
    Reads Goodbooks CSV files, constructs the (search_query, document, category) triplets,
    and returns (train_df, val_df, test_df) ready for the SentenceQueryDocumentDataset.
    """
    books = pd.read_csv("../../data/goodbooks-10k/books.csv")
    book_tags = pd.read_csv("../../data/goodbooks-10k/book_tags.csv")
    tags = pd.read_csv("../../data/goodbooks-10k/tags.csv")

    df = (
        book_tags
        .merge(tags, on="tag_id", how="inner")
        .merge(
            books[["book_id", "title", "authors", "original_publication_year"]],
            left_on="goodreads_book_id",
            right_on="book_id",
            how="inner"
        )
    )

    if cfg.min_tag_count is not None and "count" in df.columns:
        df = df[df["count"] >= cfg.min_tag_count]

    df["document"] = df.apply(GoodbooksBuilder._create_document, axis=1)
    df["category"] = df.apply(GoodbooksBuilder._create_category, axis=1)

    pairs = (
        df.rename(columns={"tag_name": "search_query"})
        .loc[:, ["search_query", "document", "category"]]
        .dropna()
        .drop_duplicates()
    )

    test_size = float(cfg.test_size)
    val_size = float(cfg.val_size)
    
    if not (0.0 < test_size < 1.0) or not (0.0 < val_size < 1.0) or (test_size + val_size) >= 1.0:
        raise ValueError(f"test_size ({test_size}) and val_size ({val_size}) must be in (0,1) and their sum < 1.0")

    total_temp_size = test_size + val_size 
    
    train_df, temp_df = train_test_split(
        pairs,
        test_size=total_temp_size, 
        random_state=cfg.random_state
    )

    relative_test_size = test_size / total_temp_size 

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size, 
        random_state=cfg.random_state 
    )

    if cfg.head_train:
        train_df = train_df.head(cfg.head_train)
    
    if cfg.head_val:
        val_df = val_df.head(cfg.head_val)
        test_df = test_df.head(cfg.head_val) 
        
    return train_df, val_df, test_df