"""Item text for content-aware recommenders, resolved from the dataset parameters.

framework.py builds every recommender with ``{**dataloader_params, **model_config}``, so a model can
find the text of its items from the same parameters its dataloader used. Item ids match the ids the
adaptations put in the interaction matrix (BEIR document ids, MS MARCO passage ids, hybrid ASINs).
"""

from typing import Dict

import pandas as pd


def _beir_texts(params: dict) -> Dict[str, str]:
    from search_recs.datasets import beir_files

    name = str(params.get("dataset_name", "")).lower()
    subset = params.get("subset") or (name[len("beir_"):] if name.startswith("beir_") else "nfcorpus")
    corpus = beir_files.read_corpus(beir_files.subset_path(subset))
    return dict(zip(corpus["document_id"].astype(str), corpus["document"].astype(str)))


def _msmarco_texts(params: dict) -> Dict[str, str]:
    """Passages of the top-1000 pools of the judged queries: the corpus the MS MARCO loaders use."""
    from search_recs.search.dataloader.base_dataloader import BuildConfig
    from search_recs.search.dataloader.msmarco_trec_dl import MsMarcoTrecDlLoader

    ingest = MsMarcoTrecDlLoader(
        BuildConfig(test_size=0.2, val_size=0.1, random_state=42, head_train=None, head_test=None),
        benchmark=params.get("benchmark", "trec-dl-2019"),
        path=params.get("path", "./data/msmarco_trec_dl"),
        candidate_limit=params.get("candidate_limit", 1000),
        qrels_min_relevance=params.get("qrels_min_relevance", 2),
        cache_processed=True,
    )
    paths = ingest._ensure_files()
    queries = ingest._read_queries(paths["queries"])
    qrels = ingest._read_qrels(paths["qrels"])
    judged = [q for q in qrels["query_id"].drop_duplicates().tolist() if q in queries]
    candidates = ingest._read_candidates(paths["top1000"], judged).drop_duplicates("document_id")
    return dict(zip(candidates["document_id"].astype(str), candidates["document"].astype(str)))


def _hybrid_texts(params: dict) -> Dict[str, str]:
    from search_recs.recs.dataloader.hybrid import _dataset_path, item_documents

    root = _dataset_path(params.get("path", "data/hybrid_dataset/hybrid_dataset"))
    docs = item_documents(pd.read_parquet(root / "items.parquet"))
    return dict(zip(docs["document_id"], docs["document"]))


def load_item_texts(params: dict) -> Dict[str, str]:
    """``item id -> text`` for the dataset described by the dataloader parameters."""
    name = str(params.get("dataset_name", "")).strip().lower()
    mode = str(params.get("mode", "")).strip().lower()
    if mode in {"beir", "beir_query"} or name.startswith("beir_"):
        return _beir_texts(params)
    if mode == "msmarco_query" or name.startswith("msmarco_trec_dl"):
        return _msmarco_texts(params)
    if mode == "hybrid" or name in {"hybrid", "hybriddataset"}:
        return _hybrid_texts(params)
    raise ValueError(
        f"[item_text] No item text source for dataset_name='{name}', mode='{mode}'. "
        f"Supported: BEIR subsets, MS MARCO TREC-DL and the hybrid dataset."
    )
