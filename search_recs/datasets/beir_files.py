import json
from pathlib import Path
from typing import List, Sequence

import pandas as pd

from .manager import ensure_dataset

DEFAULT_SUBSET = "nfcorpus"

_QRELS_COLUMN_ALIASES = {
    "query-id": "query_id",
    "query_id": "query_id",
    "qid": "query_id",
    "corpus-id": "document_id",
    "corpus_id": "document_id",
    "doc-id": "document_id",
    "doc_id": "document_id",
    "score": "score",
    "rel": "score",
    "relevance": "score",
}


def subset_path(subset: str) -> Path:
    return ensure_dataset(f"beir-{str(subset).strip().lower()}")


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_document(title, text) -> str:
    title = str(title or "").strip()
    text = str(text or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def read_corpus(base_path: Path) -> pd.DataFrame:
    rows = read_jsonl(base_path / "corpus.jsonl")
    corpus = pd.DataFrame(
        {
            "document_id": [str(r.get("_id", "")).strip() for r in rows],
            "title": [str(r.get("title", "") or "").strip() for r in rows],
            "document": [build_document(r.get("title"), r.get("text")) for r in rows],
        }
    )
    corpus = corpus[(corpus["document_id"] != "") & (corpus["document"].str.strip() != "")]
    return corpus.drop_duplicates(subset=["document_id"]).reset_index(drop=True)


def read_queries(base_path: Path) -> pd.DataFrame:
    rows = read_jsonl(base_path / "queries.jsonl")
    queries = pd.DataFrame(
        {
            "query_id": [str(r.get("_id", "")).strip() for r in rows],
            "search_query": [str(r.get("text", "") or "").strip() for r in rows],
        }
    )
    queries = queries[(queries["query_id"] != "") & (queries["search_query"] != "")]
    return queries.drop_duplicates(subset=["query_id"]).reset_index(drop=True)


def read_qrels(base_path: Path, splits: Sequence[str], min_score: float = 1.0) -> pd.DataFrame:
    qrels_dir = base_path / "qrels"
    frames = []
    missing = []

    for split in splits:
        path = qrels_dir / f"{split}.tsv"
        if not path.exists():
            missing.append(split)
            continue

        df = pd.read_csv(path, sep="\t", dtype=str)
        df.columns = [
            _QRELS_COLUMN_ALIASES.get(str(c).strip().lower(), str(c).strip()) for c in df.columns
        ]

        for required in ("query_id", "document_id", "score"):
            if required not in df.columns:
                raise ValueError(
                    f"[BEIR] {path} is missing the '{required}' column. Found: {list(df.columns)}"
                )

        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
        df = df[df["score"] >= float(min_score)]
        df["qrels_split"] = split
        frames.append(df[["query_id", "document_id", "score", "qrels_split"]])

    if missing:
        print(f"[BEIR] No qrels file for split(s) {missing} (normal for test-only subsets).")

    if not frames:
        raise FileNotFoundError(
            f"[BEIR] None of the requested qrels splits {list(splits)} exist under {qrels_dir}."
        )

    qrels = pd.concat(frames, ignore_index=True)
    qrels["query_id"] = qrels["query_id"].astype(str).str.strip()
    qrels["document_id"] = qrels["document_id"].astype(str).str.strip()
    return qrels
