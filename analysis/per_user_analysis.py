"""Per-query and manipulation analysis of the transfer conditions (item 2).

Run analysis/matrix_stats.py first. This script joins its outputs with the per-user metrics
written by framework.py (runs/<run_id>/per_user_metrics.csv) and produces (analysis/out/):

  summary.csv         per experiment: metrics (mean +- std over runs) and structural statistics
  per_user_tests.csv  base experiments: Spearman(feature, metric) and Mann-Whitney contrasts
  logit.csv           logistic regression of NDCG@10 > 0 on structural features, cluster-robust
                      SE by pseudo-user (users whose positives are all cold are excluded: their
                      score is 0 by construction and is reported separately)
  curves.csv          M1/M2/M3: metric vs manipulated level, bootstrap 95% CI over runs, trend test
  fig_*.png           cold-positives vs NDCG@10, manipulation curves, NDCG@10 by neighbours

Usage:
  python analysis/per_user_analysis.py [--artifacts artifacts] [--out analysis/out]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TARGETS = ["NDCG@10", "RECALL@50"]
FEATURES = ["deg_train", "n_neighbors", "frac_test_cold", "mean_test_item_pop", "n_test_pos"]
LOG_FEATURES = {"deg_train", "n_neighbors", "mean_test_item_pop", "n_test_pos"}
GROUP_PARAM = {"m1": "query_limit", "m2": "overlap_break", "m3": "cold_test_frac"}


# ----------------------------------------------------------------------------- loading
def load_metrics(experiments: pd.DataFrame, run_ids: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for exp in experiments.itertuples(index=False):
        wanted = set(run_ids.loc[run_ids["experiment"] == exp.experiment, "run_id"])
        for hash_dir in exp.hash_dirs:
            for path in Path(hash_dir).glob("runs/*/per_user_metrics.csv"):
                run_id = path.parent.name
                if run_id not in wanted:
                    continue
                d = pd.read_csv(path)
                d["col"] = d["metric"].str.upper() + "@" + d["K"].astype(str)
                wide = d.pivot_table(index="user", columns="col", values="value", aggfunc="first").reset_index()
                extra = d.drop_duplicates("user")[["user", "n_pos", "n_pos_unscored"]]
                frames.append(wide.merge(extra, on="user").assign(experiment=exp.experiment, run_id=run_id))
    if not frames:
        raise SystemExit("No per_user_metrics.csv matching the saved matrices was found.")
    out = pd.concat(frames, ignore_index=True)
    out["user"] = out["user"].astype(str)
    return out


# ----------------------------------------------------------------------------- statistics
def bootstrap_ci(values, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def logit_cluster(X: np.ndarray, y: np.ndarray, clusters: np.ndarray, ridge: float = 1e-6, max_iter: int = 100):
    """Logistic regression (Newton-Raphson) with cluster-robust sandwich standard errors."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = expit(X @ beta)
        H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * beta)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = expit(X @ beta)
    H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(X.shape[1])
    bread = np.linalg.inv(H)
    scores = pd.DataFrame(X * (y - p)[:, None]).groupby(clusters).sum().to_numpy()
    n_clusters = scores.shape[0]
    V = bread @ (scores.T @ scores) @ bread * n_clusters / max(n_clusters - 1, 1)
    se = np.sqrt(np.diag(V))
    return beta, se, n_clusters


def per_user_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, strategy), g in df.groupby(["dataset", "strategy"]):
        for target in TARGETS:
            if target not in g:
                continue
            for feat in FEATURES:
                sub = g[[feat, target]].dropna()
                if sub[feat].nunique() < 2 or sub[target].nunique() < 2:
                    continue
                rho, p = stats.spearmanr(sub[feat], sub[target])
                rows.append({"dataset": dataset, "strategy": strategy, "test": "spearman", "feature": feat,
                             "target": target, "n": len(sub), "stat": rho, "p_value": p})
            contrasts = {
                "has_cold_pos (frac_test_cold>0)": g["frac_test_cold"] > 0,
                "all_pos_cold (frac_test_cold==1)": g["frac_test_cold"] >= 1,
                "no_neighbor (n_neighbors==0)": g["n_neighbors"] == 0,
            }
            for label, mask in contrasts.items():
                a, b = g.loc[mask, target].dropna(), g.loc[~mask, target].dropna()
                if len(a) == 0 or len(b) == 0:
                    continue
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                rows.append({"dataset": dataset, "strategy": strategy, "test": "mannwhitney", "feature": label,
                             "target": target, "n": len(a) + len(b), "stat": u, "p_value": p,
                             "mean_true": a.mean(), "mean_false": b.mean(), "n_true": len(a), "n_false": len(b)})
    return pd.DataFrame(rows)


def logit_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled model (dataset x strategy dummies) plus one model per dataset x strategy."""
    base = df[(df["frac_test_cold"] < 1) & df["user_known"]].dropna(subset=["NDCG@10"]).copy()
    base["cell"] = base["dataset"] + "_" + base["strategy"]
    rows = []

    def fit(sub: pd.DataFrame, label: str, with_dummies: bool):
        y = (sub["NDCG@10"] > 0).astype(float).to_numpy()
        if y.min() == y.max() or len(sub) < 30:
            return
        cols, mats = [], []
        for feat in FEATURES:
            x = np.log1p(sub[feat].to_numpy(dtype=float)) if feat in LOG_FEATURES else sub[feat].to_numpy(dtype=float)
            if np.std(x) == 0:
                continue
            mats.append((x - x.mean()) / x.std())
            cols.append(f"{'log1p_' if feat in LOG_FEATURES else ''}{feat} (per SD)")
        if with_dummies:
            cells = sorted(sub["cell"].unique())
            for c in cells[1:]:
                mats.append((sub["cell"] == c).to_numpy(dtype=float))
                cols.append(f"cell={c} (vs {cells[0]})")
        X = np.column_stack([np.ones(len(sub))] + mats)
        cols = ["intercept"] + cols
        clusters = (sub["experiment"] + "|" + sub["user"]).to_numpy()
        beta, se, n_clusters = logit_cluster(X, y, clusters)
        z = beta / se
        for name, b, s, zz in zip(cols, beta, se, z):
            rows.append({"model": label, "term": name, "coef": b, "odds_ratio": np.exp(b), "se_cluster": s,
                         "z": zz, "p_value": 2 * stats.norm.sf(abs(zz)), "n_obs": len(sub),
                         "n_clusters": n_clusters, "positive_rate": y.mean(),
                         "separation_warning": bool(np.abs(beta).max() > 15)})

    fit(base, "pooled", with_dummies=True)
    for cell, sub in base.groupby("cell"):
        fit(sub, cell, with_dummies=False)
    return pd.DataFrame(rows)


def curves(run_level: pd.DataFrame, structure: pd.DataFrame) -> pd.DataFrame:
    rows = []
    beir_base = run_level[(run_level["group"] == "base") & (run_level["dataset"] == "beir")]
    for group, param in GROUP_PARAM.items():
        manip = run_level[run_level["group"] == group]
        if manip.empty:
            continue
        ref = beir_base[beir_base["strategy"].isin(manip["strategy"].unique())].copy()
        if group == "m1":  # reference level = all queries (number of pseudo-users of the base run)
            ref["level"] = ref["users"]
        else:
            ref["level"] = 0.0
        data = pd.concat([manip, ref], ignore_index=True)
        data["level"] = pd.to_numeric(data["level"])
        for strategy, g in data.groupby("strategy"):
            for target in TARGETS:
                rho, p = stats.spearmanr(g["level"], g[target]) if g["level"].nunique() > 1 else (np.nan, np.nan)
                for level, gl in g.groupby("level"):
                    lo, hi = bootstrap_ci(gl[target])
                    rows.append({"group": group, "param": param, "strategy": strategy, "target": target,
                                 "level": level, "n_runs": len(gl), "mean": gl[target].mean(), "ci_low": lo,
                                 "ci_high": hi, "users": gl["users"].mean(),
                                 "users_no_neighbor_frac": gl["users_no_neighbor_frac"].mean(),
                                 "test_pos_cold_frac": gl["test_pos_cold_frac"].mean(),
                                 "trend_spearman": rho, "trend_p_value": p})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def fig_cold(run_level: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    markers = {"base": "o", "step1": "s", "m1": "^", "m2": "D", "m3": "v"}
    for (dataset, strategy), g in run_level.groupby(["dataset", "strategy"]):
        for group, gg in g.groupby("group"):
            ax.scatter(gg["test_pos_cold_frac"], gg["NDCG@10"], alpha=0.7, marker=markers.get(group, "o"),
                       label=f"{dataset} {strategy} ({group})")
    ax.set_xlabel("fraction of test positives that are cold (never in train)")
    ax.set_ylabel("NDCG@10 (mean over pseudo-users, one point per run)")
    ax.set_xlim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_curves(curve_df: pd.DataFrame, out: Path):
    for group, g in curve_df[curve_df["target"] == "NDCG@10"].groupby("group"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for strategy, gs in g.groupby("strategy"):
            gs = gs.sort_values("level")
            ax.errorbar(gs["level"], gs["mean"], yerr=[gs["mean"] - gs["ci_low"], gs["ci_high"] - gs["mean"]],
                        marker="o", capsize=3, label=strategy)
        ax.set_xlabel(g["param"].iloc[0])
        ax.set_ylabel("NDCG@10 (95% bootstrap CI over runs)")
        ax.set_title(f"{group.upper()} on BEIR/NFCorpus")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"fig_curve_{group}.png", dpi=160)
        plt.close(fig)


def fig_neighbors(users: pd.DataFrame, path: Path):
    base = users[(users["group"] == "base") & (users["frac_test_cold"] < 1)].dropna(subset=["NDCG@10"])
    if base.empty:
        return
    base = base.assign(bucket=pd.cut(base["n_neighbors"], [-1, 0, 5, 20, 50, 100, np.inf],
                                     labels=["0", "1-5", "6-20", "21-50", "51-100", ">100"]))
    cells = list(base.groupby(["dataset", "strategy"]))
    fig, axes = plt.subplots(1, len(cells), figsize=(4.5 * len(cells), 4), squeeze=False)
    for ax, ((dataset, strategy), g) in zip(axes[0], cells):
        data = [(str(b), gb["NDCG@10"].to_numpy()) for b, gb in g.groupby("bucket", observed=True)]
        ax.boxplot([d for _, d in data], tick_labels=[b for b, _ in data])
        ax.set_title(f"{dataset} {strategy}")
        ax.set_xlabel("neighbouring pseudo-users")
        ax.set_ylabel("NDCG@10 per pseudo-user")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="analysis/out")
    args = ap.parse_args()
    out = Path(args.out)

    experiments = pd.DataFrame(json.loads((out / "experiments.json").read_text(encoding="utf-8")))
    structure = pd.read_csv(out / "structure.csv")
    features = pd.read_csv(out / "user_features.csv", dtype={"user": str, "run_id": str})
    metrics = load_metrics(experiments, structure[["experiment", "run_id"]])

    users = features.merge(metrics, on=["experiment", "run_id", "user"], how="inner")
    users.to_csv(out / "user_level_joined.csv", index=False)
    print(f"[analysis] joined {len(users)} pseudo-user rows from {users['experiment'].nunique()} experiment(s)")

    # run level: mean over the evaluated pseudo-users of each run
    metric_cols = [c for c in metrics.columns if "@" in c]
    run_level = (users.groupby(["experiment", "run_id"])[metric_cols].mean().reset_index()
                 .merge(structure, on=["experiment", "run_id"], how="left"))
    run_level.to_csv(out / "run_level.csv", index=False)

    id_cols = ["experiment", "dataset", "strategy", "group", "param", "level", "seed", "target_mode", "top_k"]
    agg = run_level.groupby(id_cols, dropna=False)
    summary = agg[metric_cols].mean().add_suffix("_mean").join(agg[metric_cols].std().add_suffix("_std"))
    struct_cols = ["users", "items", "density", "users_no_neighbor_frac", "neighbors_per_user", "lcc_frac",
                   "test_pos_cold_frac", "test_users"]
    summary = summary.join(agg[struct_cols].mean()).join(agg.size().rename("n_runs")).reset_index()
    summary.to_csv(out / "summary.csv", index=False)

    base_users = users[users["group"] == "base"]
    tests = per_user_tests(base_users)
    tests.to_csv(out / "per_user_tests.csv", index=False)
    logit = logit_table(base_users)
    logit.to_csv(out / "logit.csv", index=False)
    curve_df = curves(run_level, structure)
    curve_df.to_csv(out / "curves.csv", index=False)

    fig_cold(run_level, out / "fig_cold_vs_ndcg.png")
    fig_neighbors(users, out / "fig_ndcg_by_neighbors.png")
    if not curve_df.empty:
        fig_curves(curve_df, out)

    # console digest
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    show = ["dataset", "strategy", "group", "param", "level", "n_runs", "test_users", "users_no_neighbor_frac",
            "test_pos_cold_frac", "NDCG@10_mean", "NDCG@10_std", "RECALL@50_mean"]
    print("\n=== summary ===")
    print(summary[[c for c in show if c in summary]].round(4).to_string(index=False))
    cold_all = base_users[base_users["frac_test_cold"] >= 1]
    if len(cold_all):
        print(f"\n[analysis] pseudo-users whose positives are all cold: {len(cold_all)} "
              f"-> NDCG@10 == 0 in {(cold_all['NDCG@10'] == 0).mean():.1%} of them (excluded from the logit)")
    if not logit.empty:
        print("\n=== logit NDCG@10 > 0 (pooled, cluster-robust SE) ===")
        print(logit[logit["model"] == "pooled"][["term", "odds_ratio", "z", "p_value", "n_obs", "n_clusters"]]
              .round(4).to_string(index=False))
    if not curve_df.empty:
        print("\n=== manipulation curves (NDCG@10) ===")
        print(curve_df[curve_df["target"] == "NDCG@10"][["group", "strategy", "level", "n_runs", "mean", "ci_low",
                                                          "ci_high", "test_pos_cold_frac", "trend_spearman",
                                                          "trend_p_value"]].round(4).to_string(index=False))
    print(f"\n[analysis] outputs in {out}/")


if __name__ == "__main__":
    main()
