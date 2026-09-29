import pandas as pd

from search_recs.recs.dataloader.recs_dataloader import RecsDataLoader
from search_recs.search.dataloader.msmarco_trec_dl import MsMarcoTrecDlLoader
from search_recs.search.dataloader.base_dataloader import BuildConfig


def _opt_int(value):
    if value is None:
        return None
    try:
        iv = int(value)
    except Exception:
        return None
    return iv if iv > 0 else None


class MsMarcoTrecDlRecsDataLoader(RecsDataLoader):
    """Query-as-User hybrid: user=query, item=judged passage, label=graded qrels relevance."""

    def __init__(self, full_config: dict):
        dl = full_config.get("dataloader", full_config) or {}
        dl = {**dl, "dataset_name": dl.get("dataset_name", "msmarco_trec_dl")}
        super().__init__(dl)

        build_cfg = BuildConfig(
            test_size=0.2,
            val_size=0.1,
            random_state=int(dl.get("seed", 42)),
            head_train=None,
            head_test=_opt_int(dl.get("head_test")),
        )
        ingest_kwargs = {
            "benchmark": dl.get("benchmark", "trec-dl-2019"),
            "path": dl.get("path", "./data/msmarco_trec_dl"),
            "candidate_limit": dl.get("candidate_limit", 1000),
            "qrels_min_relevance": dl.get("qrels_min_relevance", 2),
            "cache_processed": True,
        }
        self._ingest = MsMarcoTrecDlLoader(build_cfg, **ingest_kwargs)

    def load_data(self) -> pd.DataFrame:
        return self.hybrid_load_data()

    def hybrid_load_data(self) -> pd.DataFrame:
        paths = self._ingest._ensure_files()
        queries = self._ingest._read_queries(paths["queries"])
        qrels = self._ingest._read_qrels(paths["qrels"])

        judged_qids = [q for q in qrels["query_id"].drop_duplicates().tolist() if q in queries]
        judged_qids = self._ingest._limit_queries(judged_qids)
        qrels = qrels[qrels["query_id"].isin(judged_qids)]

        candidates = self._ingest._read_candidates(paths["top1000"], judged_qids)
        doc_ids = set(candidates["document_id"])

        rows = []
        for i, r in enumerate(qrels.itertuples(index=False)):
            if r.document_id not in doc_ids:
                continue
            rows.append({
                "user": f"query::{r.query_id}",
                "item": str(r.document_id),
                "label": int(r.relevance),
                "time": i,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("[MSMARCO/Recs][QueryAsUser] No qrels pairs overlap with the candidate pool.")

        print(
            f"[MSMARCO/Recs][QueryAsUser] pairs={len(df)} | "
            f"users(queries)={df['user'].nunique()} | items(docs)={df['item'].nunique()}"
        )
        return df
