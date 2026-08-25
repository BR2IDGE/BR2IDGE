import copy
from typing import Optional, Sequence

import pandas as pd

from search_recs.datasets.beir_files import (
    DEFAULT_SUBSET,
    read_qrels,
    read_queries,
    subset_path,
)
from search_recs.recs.dataloader.recs_dataloader import RecsDataLoader

USER_PREFIX = "query::"


class BeirQueryAsUserDataLoader(RecsDataLoader):
    def __init__(self, full_config: dict):
        self.config_copy = copy.deepcopy(full_config)
        dl = self.config_copy.get("dataloader", self.config_copy)

        self.subset = str(dl.get("subset", DEFAULT_SUBSET)).strip().lower()
        dl.setdefault("path", f"beir-{self.subset}")

        super().__init__(dl)

        splits = dl.get("qrels_splits") or ("test",)
        self.qrels_splits: Sequence[str] = tuple(splits)
        self.min_score = float(dl.get("min_score", 1.0))
        self.label_mode = str(dl.get("label_mode", "binary")).strip().lower()
        self.min_user_interactions = int(dl.get("min_user_interactions", 2) or 0)
        self.query_limit: Optional[int] = dl.get("query_limit")
        self.seed = int(dl.get("seed", 42))

        self.dataset_path = subset_path(self.subset)

    def load_data(self) -> pd.DataFrame:
        base_path = self.dataset_path
        qrels = read_qrels(base_path, self.qrels_splits, self.min_score)
        queries = read_queries(base_path)

        df = qrels.merge(queries[["query_id"]], on="query_id", how="inner")
        dropped = len(qrels) - len(df)
        if dropped:
            print(f"[BEIR-QueryAsUser] Dropped {dropped} judgement(s) with no matching query.")

        if df.empty:
            raise ValueError(
                f"[BEIR-QueryAsUser] No judgements for subset '{self.subset}' "
                f"with splits {list(self.qrels_splits)} and min_score={self.min_score}."
            )

        df = df.drop_duplicates(subset=["query_id", "document_id"])

        if self.min_user_interactions > 1:
            counts = df.groupby("query_id")["document_id"].transform("size")
            before_users = df["query_id"].nunique()
            df = df[counts >= self.min_user_interactions]
            after_users = df["query_id"].nunique()
            print(
                f"[BEIR-QueryAsUser] min_user_interactions>={self.min_user_interactions}: "
                f"{before_users} -> {after_users} query-users."
            )
            if df.empty:
                raise ValueError(
                    f"[BEIR-QueryAsUser] Every query-user has fewer than "
                    f"{self.min_user_interactions} judgements, so subset '{self.subset}' cannot "
                    f"build a recs interaction matrix at this threshold. Lower "
                    f"min_user_interactions, pool more qrels_splits, pick a subset with more "
                    f"judgements per query, or use the retrieval-as-user strategy, which "
                    f"synthesises interactions instead of relying on the judgements."
                )

        if self.query_limit:
            limit = int(self.query_limit)
            keep = (
                df["query_id"].drop_duplicates()
                .sample(n=min(limit, df["query_id"].nunique()), random_state=self.seed)
            )
            df = df[df["query_id"].isin(set(keep))]
            print(f"[BEIR-QueryAsUser] query_limit={limit}: kept {df['query_id'].nunique()} query-users.")

        out = pd.DataFrame(
            {
                "user": USER_PREFIX + df["query_id"].astype(str),
                "item": df["document_id"].astype(str),
                "score": df["score"].astype(float),
            }
        )

        out = out.sort_values(["user", "score", "item"], ascending=True, kind="mergesort")
        out["time"] = out.groupby("user", sort=False).cumcount()

        if self.label_mode == "graded":
            out["label"] = out["score"]
        else:
            out["label"] = 1.0

        out = out[["user", "item", "label", "time", "score"]].reset_index(drop=True)

        per_user = out.groupby("user").size()
        print(
            f"[BEIR-QueryAsUser] subset={self.subset} | interactions={len(out)} | "
            f"query-users={out['user'].nunique()} | items={out['item'].nunique()} | "
            f"items/user: min={per_user.min()} mean={per_user.mean():.1f} max={per_user.max()}"
        )
        return out

    def hybrid_load_data(self) -> pd.DataFrame:
        print("[BEIR-QueryAsUser] Hybrid strategy 'query-as-user': queries become users.")
        return self.load_data()
