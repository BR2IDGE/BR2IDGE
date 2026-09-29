import gzip
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .base_dataloader import BaseSearchDatasetBuilder, BuildConfig


COLLECTION_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"

DATASET_FILES = {
    "2019": {
        "label": "trec-dl-2019",
        "queries": {
            "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
            "filename": "msmarco-test2019-queries.tsv.gz",
        },
        "top1000": {
            "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz",
            "filename": "msmarco-passagetest2019-top1000.tsv.gz",
        },
        "qrels": {
            "url": "https://trec.nist.gov/data/deep/2019qrels-pass.txt",
            "filename": "2019qrels-pass.txt",
        },
    },
    "2020": {
        "label": "trec-dl-2020",
        "queries": {
            "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz",
            "filename": "msmarco-test2020-queries.tsv.gz",
        },
        "top1000": {
            "url": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2020-top1000.tsv.gz",
            "filename": "msmarco-passagetest2020-top1000.tsv.gz",
        },
        "qrels": {
            "url": "https://trec.nist.gov/data/deep/2020qrels-pass.txt",
            "filename": "2020qrels-pass.txt",
        },
    },
}


def _repo_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "framework.py").exists():
            return parent
    return Path.cwd()


def _normalize_year(value: str) -> str:
    s = str(value or "").strip().lower()
    aliases = {
        "19": "2019",
        "2019": "2019",
        "dl19": "2019",
        "trec-dl19": "2019",
        "trec-dl-2019": "2019",
        "trec_dl_2019": "2019",
        "20": "2020",
        "2020": "2020",
        "dl20": "2020",
        "trec-dl20": "2020",
        "trec-dl-2020": "2020",
        "trec_dl_2020": "2020",
    }
    if s not in aliases:
        raise ValueError(f"Unsupported MS MARCO/TREC-DL benchmark '{value}'. Use 2019 or 2020.")
    return aliases[s]


class MsMarcoTrecDlLoader(BaseSearchDatasetBuilder):
    """
    Lite MS MARCO Passage loader using TREC-DL 2019/2020 queries, qrels and
    top1000 candidate passages. It intentionally avoids downloading the full
    MS MARCO passage collection and train triples.
    """

    def __init__(self, cfg: BuildConfig, **kwargs):
        super().__init__(cfg)
        benchmark = kwargs.get("benchmark", kwargs.get("year", "2019"))
        self.year = _normalize_year(benchmark)
        self.spec = DATASET_FILES[self.year]
        self.label = self.spec["label"]
        self.data_dir = self._resolve_data_dir(kwargs.get("path", "./data/msmarco_trec_dl"))
        self.candidate_limit = self._opt_int(kwargs.get("candidate_limit"), default=1000)
        self.qrels_min_relevance = int(kwargs.get("qrels_min_relevance", 1))
        self.cache_processed = bool(kwargs.get("cache_processed", True))
        self.corpus_mode = str(kwargs.get("corpus_mode", "lite")).lower()
        if self.corpus_mode not in ("lite", "full"):
            raise ValueError(f"Unsupported corpus_mode '{self.corpus_mode}'. Use 'lite' or 'full'.")

    def _resolve_data_dir(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = _repo_root() / path
        return path

    def _opt_int(self, value, default=None):
        if value is None:
            return default
        try:
            iv = int(value)
        except Exception:
            return default
        return iv if iv > 0 else default

    def _paths(self) -> Dict[str, Path]:
        year_dir = self.data_dir / self.year
        return {
            key: year_dir / meta["filename"]
            for key, meta in self.spec.items()
            if isinstance(meta, dict) and "filename" in meta
        }

    def _download_file(self, url: str, dest: Path) -> None:
        if dest.exists() and dest.stat().st_size > 0:
            return

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"[MSMARCO/TREC-DL] Downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "BR2IDGE-msmarco/1.0"})
        with urllib.request.urlopen(req, timeout=120) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        tmp.replace(dest)

    def _ensure_files(self) -> Dict[str, Path]:
        paths = self._paths()
        for key, path in paths.items():
            self._download_file(self.spec[key]["url"], path)
        return paths

    def _read_queries(self, path: Path) -> Dict[str, str]:
        queries = {}
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2:
                    continue
                queries[str(parts[0])] = parts[1]
        return queries

    def _read_qrels(self, path: Path) -> pd.DataFrame:
        qrels = pd.read_csv(
            path,
            sep=r"\s+",
            names=["query_id", "iteration", "document_id", "relevance"],
            dtype={"query_id": str, "iteration": str, "document_id": str, "relevance": int},
            engine="python",
        )
        return qrels

    def _read_candidates(self, path: Path, judged_qids: Iterable[str]) -> pd.DataFrame:
        judged_qids = set(map(str, judged_qids))
        chunks: List[pd.DataFrame] = []
        names = ["query_id", "document_id", "search_query", "document"]
        for chunk in pd.read_csv(
            path,
            sep="\t",
            names=names,
            dtype=str,
            compression="gzip",
            chunksize=100_000,
            quoting=3,
            on_bad_lines="skip",
        ):
            chunk = chunk[chunk["query_id"].isin(judged_qids)].copy()
            if not chunk.empty:
                chunks.append(chunk)

        if not chunks:
            return pd.DataFrame(columns=names)

        candidates = pd.concat(chunks, ignore_index=True)
        candidates = candidates.dropna(subset=["query_id", "document_id", "search_query", "document"])
        candidates = candidates.drop_duplicates(subset=["query_id", "document_id"], keep="first")

        if self.candidate_limit:
            candidates = (
                candidates.groupby("query_id", sort=False, group_keys=False)
                .head(int(self.candidate_limit))
                .reset_index(drop=True)
            )
        return candidates

    def _limit_queries(self, qids: List[str]) -> List[str]:
        if self.cfg.head_test:
            return qids[: int(self.cfg.head_test)]
        return qids

    def _ensure_collection(self) -> Path:
        full_dir = self.data_dir / "full"
        tsv_path = full_dir / "collection.tsv"
        if tsv_path.exists():
            return tsv_path

        archive = full_dir / "collection.tar.gz"
        self._download_file(COLLECTION_URL, archive)
        print(f"[MSMARCO/TREC-DL] Extracting {archive.name} (full passage collection)...")
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(full_dir)
        archive.unlink(missing_ok=True)

        if not tsv_path.exists():
            found = next(full_dir.rglob("collection.tsv"), None)
            if found is None:
                raise FileNotFoundError(f"collection.tsv not found after extracting {archive}")
            tsv_path = found
        return tsv_path

    def _read_collection(self, path: Path) -> Dict[str, str]:
        print(f"[MSMARCO/TREC-DL] Reading full passage collection: {path.name}")
        collection: Dict[str, str] = {}
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    collection[parts[0]] = parts[1]
        print(f"[MSMARCO/TREC-DL] Full collection loaded: {len(collection):,} passages")
        return collection

    def _build_full(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        collection = self._read_collection(self._ensure_collection())

        paths = self._ensure_files()
        queries = self._read_queries(paths["queries"])
        qrels = self._read_qrels(paths["qrels"])

        judged_qids = [qid for qid in qrels["query_id"].drop_duplicates().tolist() if qid in queries]
        judged_qids = self._limit_queries(judged_qids)
        rel_qrels = qrels[
            qrels["query_id"].isin(judged_qids) & (qrels["relevance"] >= self.qrels_min_relevance)
        ]

        test_rows = [
            {
                "search_query": queries[qid],
                "document_id": pid,
                "document": collection[pid],
                "category": self.label,
            }
            for qid, pid in zip(rel_qrels["query_id"], rel_qrels["document_id"])
            if pid in collection
        ]
        test_df = pd.DataFrame(test_rows)
        if test_df.empty:
            raise ValueError(f"No evaluable queries found for {self.label} (full corpus).")

        gt_ids = set(test_df["document_id"])
        all_ids = list(collection.keys())
        if self.cfg.head_train:
            keep_ids = list(dict.fromkeys(all_ids[: int(self.cfg.head_train)] + list(gt_ids)))
        else:
            keep_ids = all_ids

        train_df = pd.DataFrame(
            {"document_id": keep_ids, "document": [collection[pid] for pid in keep_ids]}
        )
        train_df["search_query"] = train_df["document"]
        train_df["category"] = f"{self.label}-full-corpus"
        train_df = train_df[["search_query", "document", "document_id", "category"]]
        val_df = pd.DataFrame(columns=train_df.columns)

        print(
            f"[MSMARCO/TREC-DL] Built {self.label} (FULL corpus): "
            f"corpus_docs={len(train_df)} eval_rows={len(test_df)} rel_threshold={self.qrels_min_relevance}"
        )
        return train_df.reset_index(drop=True), val_df, test_df.reset_index(drop=True)

    def _build(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cache_path = self.data_dir / self.year / (
            f"processed_{self.corpus_mode}_rel{self.qrels_min_relevance}_cand{self.candidate_limit or 'all'}"
            f"_head{self.cfg.head_test or 'all'}.pkl"
        )
        if self.cache_processed and cache_path.exists():
            print(f"[MSMARCO/TREC-DL] Loading processed cache: {cache_path}")
            try:
                data = pd.read_pickle(cache_path)
                return data["train"], data["val"], data["test"]
            except Exception as e:
                print(
                    "[MSMARCO/TREC-DL] Warning: processed cache is incompatible or corrupted; "
                    f"rebuilding it. Error: {e.__class__.__name__}: {e}"
                )

        if self.corpus_mode == "full":
            train_df, val_df, test_df = self._build_full()
        else:
            paths = self._ensure_files()
            queries = self._read_queries(paths["queries"])
            qrels = self._read_qrels(paths["qrels"])

            judged_qids = [qid for qid in qrels["query_id"].drop_duplicates().tolist() if qid in queries]
            judged_qids = self._limit_queries(judged_qids)
            qrels = qrels[qrels["query_id"].isin(judged_qids)].copy()

            candidates = self._read_candidates(paths["top1000"], judged_qids)
            if candidates.empty:
                raise ValueError(f"No top1000 candidates found for {self.label}.")

            rel_qrels = qrels[qrels["relevance"] >= self.qrels_min_relevance].copy()
            candidate_ids_by_qid = candidates.groupby("query_id")["document_id"].apply(list).to_dict()
            candidate_set_by_qid = {qid: set(ids) for qid, ids in candidate_ids_by_qid.items()}
            relevant_ids_by_qid = rel_qrels.groupby("query_id")["document_id"].apply(list).to_dict()

            test_rows = []
            doc_lookup = candidates.drop_duplicates("document_id").set_index("document_id")["document"].to_dict()

            for qid in judged_qids:
                candidate_ids = candidate_ids_by_qid.get(qid, [])
                candidate_set = candidate_set_by_qid.get(qid, set())
                ground_truth_ids = [
                    doc_id for doc_id in relevant_ids_by_qid.get(qid, []) if doc_id in candidate_set
                ]
                if not candidate_ids or not ground_truth_ids:
                    continue

                first_gt = ground_truth_ids[0]
                test_rows.append(
                    {
                        "query_id": qid,
                        "search_query": queries[qid],
                        "document_id": first_gt,
                        "document": doc_lookup.get(first_gt, ""),
                        "category": self.label,
                        "candidate_ids": json.dumps(candidate_ids),
                        "ground_truth_ids": json.dumps(ground_truth_ids),
                    }
                )

            test_df = pd.DataFrame(test_rows)
            if test_df.empty:
                raise ValueError(f"No evaluable queries found for {self.label}.")

            used_candidate_ids = set()
            for ids in test_df["candidate_ids"]:
                used_candidate_ids.update(json.loads(ids))

            train_df = (
                candidates[candidates["document_id"].isin(used_candidate_ids)]
                .drop_duplicates(subset=["document_id"], keep="first")
                .copy()
            )
            train_df["category"] = f"{self.label}-candidate-corpus"

            if self.cfg.head_train:
                train_df = train_df.head(int(self.cfg.head_train)).copy()

            train_df = train_df[["search_query", "document", "document_id", "category"]].reset_index(drop=True)
            val_df = pd.DataFrame(columns=train_df.columns)
            test_df = test_df.reset_index(drop=True)

            print(
                f"[MSMARCO/TREC-DL] Built {self.label}: "
                f"corpus_docs={len(train_df)} eval_queries={len(test_df)} "
                f"qrels={len(qrels)} rel_threshold={self.qrels_min_relevance}"
            )

        if self.cache_processed:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                pd.to_pickle({"train": train_df, "val": val_df, "test": test_df}, cache_path)
            except Exception as e:
                print(
                    "[MSMARCO/TREC-DL] Warning: could not write processed cache. "
                    f"Continuing without cache. Error: {e.__class__.__name__}: {e}"
                )

        return train_df, val_df, test_df

    def load_raw(self) -> None:
        self._ensure_files()

    def build_pairs(self) -> pd.DataFrame:
        raise NotImplementedError("MS MARCO/TREC-DL uses predefined train/test splits.")

    def build(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self._build()


def load_msmarco_trec_dl_dataset(cfg: BuildConfig, **kwargs):
    return MsMarcoTrecDlLoader(cfg, **kwargs).build()
