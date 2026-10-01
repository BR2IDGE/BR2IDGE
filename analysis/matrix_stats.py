"""Structural statistics of the adapted pseudo-user x item matrices (item 2).

Reads every experiment found under artifacts/ (via the runs' full_config.json) and the train/test
matrices saved by framework.py in artifacts/<dataset>/<experiment>/matrices/.

Outputs (analysis/out/):
  structure.csv      one row per experiment x run: size, density, overlap, connectivity, cold test items
  user_features.csv  one row per experiment x run x pseudo-user: degree, neighbours, cold positives ...

Usage:
  python analysis/matrix_stats.py [--artifacts artifacts] [--out analysis/out]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

STRATEGY_SHORT = {"query-as-user": "QaU", "retrieval-as-user": "RaU"}
DATASET_SHORT = {"beir_nfcorpus": "beir", "msmarco_trec_dl": "msmarco", "hybrid": "hybrid"}
CONTEXT_PREFIX = "user::"  # real users injected as training-only context (hybrid dataset, group ctx)


def discover(artifacts: Path) -> pd.DataFrame:
    """One row per experiment: its metadata, matrices dir and the hash dirs holding its runs."""
    rows = {}
    for cfg_path in sorted(artifacts.glob("*/*/*/full_config.json")):
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        exp = cfg.get("experiment") or {}
        name = exp.get("experiment_name")
        if not name or exp.get("task_type") != "recs" or not exp.get("hybrid"):
            continue
        dataset_key = cfg_path.parents[2].name
        matrices = artifacts / dataset_key / name / "matrices"
        if not matrices.is_dir():
            continue
        meta = exp.get("analysis") or {}
        strategy = STRATEGY_SHORT.get(str(exp.get("hybridStrategie", exp.get("hybridStrategy", ""))).lower(), "?")
        dl = cfg.get("dataloader") or {}
        row = rows.setdefault(name, {
            "experiment": name,
            "dataset": meta.get("dataset", DATASET_SHORT.get(dataset_key, dataset_key)),
            "strategy": meta.get("strategy", strategy),
            "group": meta.get("group", "base"),
            "param": meta.get("param"),
            "level": meta.get("level"),
            "seed": meta.get("seed", dl.get("seed")),
            "target_mode": dl.get("target_mode"),
            "top_k": dl.get("top_k"),
            "model": cfg_path.parents[1].name,
            "matrices": str(matrices),
            "hash_dirs": [],
        })
        row["hash_dirs"].append(str(cfg_path.parent))
    return pd.DataFrame(list(rows.values()))


def _binary_matrix(train: pd.DataFrame):
    users = train["user"].astype(str).astype("category")
    items = train["item"].astype(str).astype("category")
    X = sp.csr_matrix(
        (np.ones(len(train), dtype=np.float32), (users.cat.codes.to_numpy(), items.cat.codes.to_numpy())),
        shape=(len(users.cat.categories), len(items.cat.categories)),
    )
    X.data[:] = 1.0  # duplicates collapse to a single interaction
    return X, users.cat.categories, items.cat.categories


def analyse_run(train: pd.DataFrame, test: pd.DataFrame):
    X, users, items = _binary_matrix(train)
    n_users, n_items = X.shape

    user_deg = np.asarray(X.sum(axis=1)).ravel()
    item_deg = np.asarray(X.sum(axis=0)).ravel()
    # per-user statistics describe the pseudo-users (queries); injected real users only add edges
    pseudo = ~pd.Index(users.astype(str)).str.startswith(CONTEXT_PREFIX)

    UU = (X @ X.T).tocsr()
    UU.setdiag(0)
    UU.eliminate_zeros()
    neighbours = np.diff(UU.indptr)

    II = (X.T @ X).tocsr()
    II.setdiag(0)
    II.eliminate_zeros()
    cooc_items = np.diff(II.indptr)

    bipartite = sp.bmat([[None, X], [X.T, None]]).tocsr()
    n_comp, labels = connected_components(bipartite, directed=False)
    comp_sizes = np.bincount(labels)
    lcc = comp_sizes.argmax()
    user_in_lcc = labels[:n_users] == lcc

    item_pop = pd.Series(item_deg, index=items)
    test = test.assign(user=test["user"].astype(str), item=test["item"].astype(str))
    test_pop = test["item"].map(item_pop).fillna(0.0)
    test_cold = test_pop == 0

    structure = {
        "users": n_users,
        "pseudo_users": int(pseudo.sum()),
        "context_users": int((~pseudo).sum()),
        "items": n_items,
        "interactions": int(X.nnz),
        "density": X.nnz / max(n_users * n_items, 1),
        "user_deg_mean": user_deg[pseudo].mean(),
        "user_deg_median": float(np.median(user_deg[pseudo])),
        "item_deg_mean": item_deg.mean(),
        "items_deg1_frac": (item_deg == 1).mean(),
        "users_no_neighbor_frac": (neighbours[pseudo] == 0).mean(),
        "neighbors_per_user": neighbours[pseudo].mean(),
        "items_no_cooc_frac": (cooc_items == 0).mean(),
        "cooc_per_item": cooc_items.mean(),
        "components": int(n_comp),
        "lcc_frac": comp_sizes.max() / (n_users + n_items),
        "test_users": test["user"].nunique(),
        "test_pos": len(test),
        "test_pos_cold_frac": test_cold.mean(),
        "test_users_unknown_frac": (~pd.Series(test["user"].unique()).isin(set(users))).mean(),
    }

    per_user = pd.DataFrame({
        "user": users.astype(str),
        "deg_train": user_deg,
        "n_neighbors": neighbours,
        "in_lcc": user_in_lcc,
    })
    t = test.assign(_cold=test_cold.to_numpy(), _pop=test_pop.to_numpy())
    tu = t.groupby("user").agg(n_test_pos=("item", "size"), frac_test_cold=("_cold", "mean"),
                               mean_test_item_pop=("_pop", "mean")).reset_index()
    per_user = tu.merge(per_user, on="user", how="left")
    per_user["user_known"] = per_user["deg_train"].notna()
    per_user[["deg_train", "n_neighbors"]] = per_user[["deg_train", "n_neighbors"]].fillna(0)
    per_user["in_lcc"] = per_user["in_lcc"].astype("boolean").fillna(False).astype(bool)
    return structure, per_user


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="analysis/out")
    args = ap.parse_args()

    artifacts, out = Path(args.artifacts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    experiments = discover(artifacts)
    if experiments.empty:
        raise SystemExit(f"No hybrid recs experiment with saved matrices found under {artifacts}/.")
    experiments.to_json(out / "experiments.json", orient="records", indent=2)

    id_cols = ["experiment", "model", "dataset", "strategy", "group", "param", "level", "seed", "target_mode", "top_k"]
    structure_rows, user_frames = [], []
    for exp in experiments.itertuples(index=False):
        metas = sorted(Path(exp.matrices).glob("*_meta.json"))
        print(f"[matrix_stats] {exp.experiment}: {len(metas)} run(s)")
        for meta_path in metas:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            stem = meta_path.name[: -len("_meta.json")]
            train = pd.read_parquet(meta_path.with_name(f"{stem}_train.parquet"), columns=["user", "item"])
            test = pd.read_parquet(meta_path.with_name(f"{stem}_test.parquet"), columns=["user", "item"])
            structure, per_user = analyse_run(train, test)

            ids = {c: getattr(exp, c) for c in id_cols}
            ids["run_id"] = meta["run_id"]
            extra = {k: meta[k] for k in ("items_made_cold", "users_without_history") if k in meta}
            structure_rows.append({**ids, **structure, **extra})
            user_frames.append(per_user.assign(**ids))

    structure = pd.DataFrame(structure_rows)
    users = pd.concat(user_frames, ignore_index=True)
    structure.to_csv(out / "structure.csv", index=False)
    users.to_csv(out / "user_features.csv", index=False)

    summary_cols = ["pseudo_users", "context_users", "items", "density", "users_no_neighbor_frac",
                    "neighbors_per_user", "components", "lcc_frac", "test_pos_cold_frac"]
    base = structure[structure["group"] == "base"]
    if not base.empty:
        pd.set_option("display.width", 200)
        print("\n=== base experiments (mean over runs) ===")
        print(base.groupby(["dataset", "strategy"])[summary_cols].mean().round(4).T.to_string())
    print(f"\n[matrix_stats] wrote {out / 'structure.csv'} ({len(structure)} rows) and "
          f"{out / 'user_features.csv'} ({len(users)} rows)")


if __name__ == "__main__":
    main()
