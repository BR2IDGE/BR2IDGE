from typing import Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from search_recs.datasets.beir_files import (
    DEFAULT_SUBSET,
    read_corpus,
    read_qrels,
    read_queries,
    subset_path,
)
from .base_dataloader import BuildConfig

CORPUS_CATEGORY = "corpus"

def _three_way_split(frame: pd.DataFrame, cfg: BuildConfig):
    test_size = float(cfg.test_size)
    val_size = float(cfg.val_size)
    eval_size = test_size + val_size

    train_part, eval_part = train_test_split(
        frame, test_size=eval_size, random_state=cfg.random_state, shuffle=True
    )
    relative_test = test_size / eval_size if eval_size > 0 else 0.5
    val_part, test_part = train_test_split(
        eval_part, test_size=relative_test, random_state=cfg.random_state, shuffle=True
    )
    return train_part, val_part, test_part


def _split_pairs(
    pairs: pd.DataFrame, cfg: BuildConfig, split_by: str = "query"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    split_by = str(split_by or "query").strip().lower()

    if split_by == "pair":
        train_df, val_df, test_df = _three_way_split(pairs, cfg)
    elif split_by == "query":
        queries = pd.Series(pairs["search_query"].unique(), name="search_query")
        train_q, val_q, test_q = _three_way_split(queries.to_frame(), cfg)
        by_query = {name: frame["search_query"] for name, frame in
                    (("train", train_q), ("val", val_q), ("test", test_q))}
        train_df = pairs[pairs["search_query"].isin(set(by_query["train"]))]
        val_df = pairs[pairs["search_query"].isin(set(by_query["val"]))]
        test_df = pairs[pairs["search_query"].isin(set(by_query["test"]))]
    else:
        raise ValueError(f"[BEIR] split_by must be 'query' or 'pair', got '{split_by}'.")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _corpus_filler_rows(
    corpus: pd.DataFrame, already_indexed: set, subset: str, max_snippet_chars: int
) -> pd.DataFrame:

    columns = ["search_query", "document", "document_id", "category"]

    filler = corpus[~corpus["document_id"].isin(already_indexed)].copy()
    if filler.empty:
        return pd.DataFrame(columns=columns)

    snippet = filler["document"].str.slice(0, max_snippet_chars).str.strip()
    filler["search_query"] = filler["title"].where(filler["title"] != "", snippet)
    filler["category"] = CORPUS_CATEGORY

    return filler[columns]


def load_beir_dataset(
    cfg: BuildConfig,
    subset: str = DEFAULT_SUBSET,
    qrels_splits: Optional[Sequence[str]] = None,
    min_score: float = 1.0,
    split_by: str = "query",
    include_full_corpus: bool = True,
    corpus_query_max_chars: int = 120,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset = str(subset or DEFAULT_SUBSET).strip().lower()
    splits = tuple(qrels_splits) if qrels_splits else ("test",)

    if str(kwargs.get("mode", "")).strip().lower() == "hybrid":
        print(
            "[BEIR] Warning: hybrid mode was requested, but BEIR has no user-history "
            "variant -- the four hybrid strategies do not apply to it. Loading the "
            "standard query/document task instead."
        )

    base_path = subset_path(subset)
    print(f"[BEIR] Subset '{subset}' from {base_path}")

    corpus = read_corpus(base_path)
    queries = read_queries(base_path)
    qrels = read_qrels(base_path, splits, min_score)

    print(
        f"[BEIR] Corpus: {len(corpus)} docs | Queries: {len(queries)} | "
        f"Judgements (score >= {min_score}) from {list(splits)}: {len(qrels)}"
    )

    pairs = qrels.merge(queries, on="query_id", how="inner").merge(
        corpus[["document_id", "document"]], on="document_id", how="inner"
    )

    dropped = len(qrels) - len(pairs)
    if dropped > 0:
        print(f"[BEIR] Dropped {dropped} judgement(s) whose query or document is absent from the subset.")

    if pairs.empty:
        raise ValueError(
            f"[BEIR] No usable pairs for subset '{subset}' with splits {list(splits)} "
            f"and min_score={min_score}."
        )

    pairs["category"] = subset
    pairs = pairs[["search_query", "document", "document_id", "category"]]
    pairs = pairs.drop_duplicates(subset=["search_query", "document_id"]).reset_index(drop=True)

    print(f"[BEIR] Total pairs: {len(pairs)} | UNIQUE QUERIES: {pairs['search_query'].nunique()}")

    train_df, val_df, test_df = _split_pairs(pairs, cfg, split_by=split_by)
    print(
        f"[BEIR] Split by {split_by}: train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        f"(test queries: {test_df['search_query'].nunique()})"
    )

    if getattr(cfg, "head_train", None):
        train_df = train_df.head(int(cfg.head_train))
    if getattr(cfg, "head_val", None):
        val_df = val_df.head(int(cfg.head_val))
    if getattr(cfg, "head_test", None):
        test_df = test_df.head(int(cfg.head_test))

    if include_full_corpus:
        filler = _corpus_filler_rows(
            corpus, set(train_df["document_id"]), subset, int(corpus_query_max_chars)
        )
        train_df = pd.concat([train_df, filler], ignore_index=True)
        covered = train_df["document_id"].nunique()
        print(
            f"[BEIR] Appended {len(filler)} corpus document(s) to train; it now "
            f"covers {covered}/{len(corpus)} documents."
        )
        if covered != len(corpus):
            print(f"[BEIR] Warning: train frame does not cover the whole corpus ({covered} != {len(corpus)}).")
    else:
        print("[BEIR] Warning: include_full_corpus=False — the index will only contain judged documents.")

    print(f"[BEIR] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df
