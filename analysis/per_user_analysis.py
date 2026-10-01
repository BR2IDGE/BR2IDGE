"""Per-query and manipulation analysis of the transfer conditions (item 2).

Run analysis/matrix_stats.py first. This script joins its outputs with the per-user metrics
written by framework.py (runs/<run_id>/per_user_metrics.csv) and produces (analysis/out/):

  summary.csv         per experiment: metrics (mean +- std over runs) and structural statistics
  per_user_tests.csv  base experiments: Spearman(feature, metric) and Mann-Whitney contrasts
  logit.csv           logistic regression of NDCG@10 > 0 on structural features, cluster-robust
                      SE by pseudo-user (users whose positives are all cold are excluded: their
                      score is 0 by construction and is reported separately)
  curves.csv          M1/M2/M3: metric vs manipulated level, bootstrap 95% CI over runs, trend test
  mediation.csv       line NDCG@10 ~ cold fraction fitted on base + M3; residual of each M1/M2/ctx level (CI)
  structure_vs_performance.csv  hypothesis 2: each structural property vs NDCG@10 (Spearman, R2, R2 beyond coverage)
  feature_control.csv criterion 4: same conditions with pure CF (LightFMModel) vs text-aware LightFM
                      (LightFMTextModel: identity + text; LightFMContentModel: text only), tie-neutral metrics
  fig_*.png           cold-positives vs NDCG@10, manipulation curves, NDCG@10 by neighbours

Usage:
  python analysis/per_user_analysis.py [--artifacts artifacts] [--out analysis/out]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TARGETS = ["NDCG@10", "RECALL@50"]
FEATURES = ["deg_train", "n_neighbors", "frac_test_cold", "mean_test_item_pop", "n_test_pos"]
LOG_FEATURES = {"deg_train", "n_neighbors", "mean_test_item_pop", "n_test_pos"}
GROUP_PARAM = {"m1": "query_limit", "m2": "overlap_break", "m3": "cold_test_frac", "ctx": "context_users_frac"}
# the structural properties proposed for hypothesis 2 (+ coverage of the test items)
STRUCT_MEASURES = {
    "pseudo_users": "n. de pseudo-usuarios",
    "user_deg_mean": "itens por pseudo-usuario",
    "neighbors_per_user": "vizinhos por pseudo-usuario (sobreposicao)",
    "users_no_neighbor_frac": "% pseudo-usuarios sem vizinho",
    "density": "densidade",
    "lcc_frac": "fracao no maior componente",
    "cooc_per_item": "coocorrencias por item",
    "items_no_cooc_frac": "% itens sem coocorrencia",
    "test_pos_cold_frac": "% positivos de teste frios (cobertura)",
}
LOG_MEASURES = {"pseudo_users", "user_deg_mean", "neighbors_per_user", "cooc_per_item"}


def _comparable(df: pd.DataFrame) -> pd.DataFrame:
    """Runs on the sampled protocol that measure transfer: no 'all' target mode (memorisation of the
    own history) and no fullrank runs (different candidate protocol)."""
    keep = ((df["target_mode"].fillna("").astype(str) != "all") & (df["group"] != "fullrank")
            & ~df["group"].astype(str).str.startswith("feat_"))  # other recommender: see feature_control()
    return df[keep]


# ----------------------------------------------------------------------------- loading
def load_metrics(experiments: pd.DataFrame, run_ids: pd.DataFrame) -> pd.DataFrame:
    frames, broken = [], []
    for exp in experiments.itertuples(index=False):
        wanted = set(run_ids.loc[run_ids["experiment"] == exp.experiment, "run_id"])
        for hash_dir in exp.hash_dirs:
            for path in Path(hash_dir).glob("runs/*/per_user_metrics.csv"):
                run_id = path.parent.name
                if run_id not in wanted:
                    continue
                if path.stat().st_size == 0:
                    broken.append((exp.experiment, run_id))
                    continue
                d = pd.read_csv(path)
                d["col"] = d["metric"].str.upper() + "@" + d["K"].astype(str)
                wide = d.pivot_table(index="user", columns="col", values="value", aggfunc="first").reset_index()
                extra = d.drop_duplicates("user")[["user", "n_pos", "n_pos_unscored"]]
                frames.append(wide.merge(extra, on="user").assign(experiment=exp.experiment, run_id=run_id))
    if broken:
        print(f"[analysis] WARNING: {len(broken)} empty per_user_metrics.csv skipped (incomplete copy?):")
        for name, n in pd.DataFrame(broken, columns=["exp", "run"]).groupby("exp").size().items():
            print(f"             {name}: {n} run(s)")
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
    base = _comparable(run_level[run_level["group"] == "base"])
    for group, param in GROUP_PARAM.items():
        manip = run_level[run_level["group"] == group]
        for (dataset, strategy), gm in manip.groupby(["dataset", "strategy"]):
            ref = base[(base["dataset"] == dataset) & (base["strategy"] == strategy)].copy()
            if group == "m1":  # reference level = all queries (number of pseudo-users of the base run)
                ref["level"] = ref["pseudo_users"] if "pseudo_users" in ref else ref["users"]
            else:
                ref["level"] = 0.0
            g = pd.concat([gm, ref], ignore_index=True)
            g["level"] = pd.to_numeric(g["level"])
            for target in TARGETS:
                rho, p = stats.spearmanr(g["level"], g[target]) if g["level"].nunique() > 1 else (np.nan, np.nan)
                for level, gl in g.groupby("level"):
                    lo, hi = bootstrap_ci(gl[target])
                    rows.append({"group": group, "param": param, "dataset": dataset, "strategy": strategy,
                                 "target": target, "level": level, "n_runs": len(gl), "mean": gl[target].mean(),
                                 "ci_low": lo, "ci_high": hi, "users": gl["users"].mean(),
                                 "users_no_neighbor_frac": gl["users_no_neighbor_frac"].mean(),
                                 "test_pos_cold_frac": gl["test_pos_cold_frac"].mean(),
                                 "trend_spearman": rho, "trend_p_value": p})
    return pd.DataFrame(rows)


def mediation(run_level: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Does the cold-test fraction explain the manipulations? Per dataset x strategy, fit
    NDCG@10 = a + b * test_pos_cold_frac on the base + M3 runs (cold items controlled directly),
    then report the mean residual of every M1 / M2 / ctx level with a bootstrap CI over its runs.
    A CI that excludes 0 = the manipulation acts beyond the change in coverage."""
    rows = []
    data = _comparable(run_level).dropna(subset=["NDCG@10", "test_pos_cold_frac"])
    rng = np.random.default_rng(seed)
    for (dataset, strategy), g in data.groupby(["dataset", "strategy"]):
        ref = g[g["group"].isin(["base", "m3"])]
        if ref["test_pos_cold_frac"].nunique() < 3:
            continue
        x, y = ref["test_pos_cold_frac"].to_numpy(), ref["NDCG@10"].to_numpy()
        b, a = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (a + b * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        base = {"dataset": dataset, "strategy": strategy, "intercept": a, "slope": b, "r2_reference": r2,
                "n_reference_runs": len(ref)}
        rows.append({**base, "group": "reference (base + m3)", "level": np.nan, "n_runs": len(ref),
                     "mean_cold": x.mean(), "mean_ndcg": y.mean(), "mean_residual": 0.0,
                     "ci_low": np.nan, "ci_high": np.nan})
        for group in ("m1", "m2", "ctx"):
            for level, gl in g[g["group"] == group].groupby("level"):
                resid = gl["NDCG@10"].to_numpy() - (a + b * gl["test_pos_cold_frac"].to_numpy())
                boots = rng.choice(resid, size=(n_boot, len(resid)), replace=True).mean(axis=1) if len(resid) > 1 else resid
                lo, hi = np.percentile(boots, [2.5, 97.5]) if len(resid) > 1 else (np.nan, np.nan)
                rows.append({**base, "group": group, "level": level, "n_runs": len(gl),
                             "mean_cold": gl["test_pos_cold_frac"].mean(), "mean_ndcg": gl["NDCG@10"].mean(),
                             "mean_residual": resid.mean(), "ci_low": lo, "ci_high": hi})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["beyond_coverage"] = (out["ci_high"] < 0) | (out["ci_low"] > 0)
    return out


def _r2(columns, y) -> float:
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, dtype=float) for c in columns])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def structure_vs_performance(run_level: pd.DataFrame) -> pd.DataFrame:
    """Hypothesis 2: each structural property of the adapted matrices vs NDCG@10 across runs, per
    strategy (datasets pooled): Spearman, R2 alone, and R2 added on top of the coverage of the test
    items (a property that matters beyond coverage adds R2)."""
    rows = []
    data = _comparable(run_level).dropna(subset=["NDCG@10", "test_pos_cold_frac"])
    for strategy, g in data.groupby("strategy"):
        y = g["NDCG@10"].to_numpy(dtype=float)
        cold = g["test_pos_cold_frac"].to_numpy(dtype=float)
        # without variation in coverage (or too few runs) the "beyond coverage" comparison is undefined
        r2_cold = _r2([cold], y) if (np.std(cold) > 0 and len(g) >= 10) else np.nan
        for col, label in STRUCT_MEASURES.items():
            if col not in g or g[col].nunique() < 2:
                continue
            x = g[col].to_numpy(dtype=float)
            xs = np.log1p(x) if col in LOG_MEASURES else x
            rho, p = stats.spearmanr(x, y)
            rows.append({"strategy": strategy, "datasets": "+".join(sorted(g["dataset"].unique())),
                         "n_runs": len(g), "n_experiments": g["experiment"].nunique(), "measure": col,
                         "label": label, "spearman": rho, "p_value": p, "r2_alone": _r2([xs], y),
                         "r2_coverage": r2_cold,
                         "delta_r2_beyond_coverage": np.nan if col == "test_pos_cold_frac" else _r2([cold, xs], y) - r2_cold})
    return pd.DataFrame(rows)


def _condition(row) -> str:
    """Comparable experimental condition, shared by LightFM and its text-feature variant."""
    group = str(row["group"]).replace("feat_", "")
    if group == "base" or (group == "step1" and str(row.get("target_mode")) == "new"):
        top_k = row.get("top_k")
        return "base" + (f" (top_k={int(top_k)})" if row["dataset"] == "msmarco" and pd.notna(top_k) and int(top_k) == 100 else "")
    if group in ("m3", "m2"):
        return f"{GROUP_PARAM[group]}={float(row['level']):g}"
    return ""


CF_MODEL = "LightFMModel"


def neutral_run_metrics(experiments: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """NDCG@10 / RECALL@50 per run recomputed from eval_scores.parquet with a fixed random order inside
    ties. The framework lists positives first and numpy's tie order differs across machines, which
    matters when a model ties many candidates (pure CF gives -inf to every unseen item)."""
    from search_recs.metric import NdcgAtK, RecallAtK

    ndcg, recall = NdcgAtK(), RecallAtK()
    wanted = set(zip(runs["experiment"], runs["run_id"]))
    rows = []
    for exp in experiments.itertuples(index=False):
        if exp.experiment not in set(runs["experiment"]):
            continue
        for hash_dir in exp.hash_dirs:
            for path in Path(hash_dir).glob("runs/*/eval_scores.parquet"):
                run_id = path.parent.name
                if (exp.experiment, run_id) not in wanted or path.stat().st_size == 0:
                    continue
                scores = pd.read_parquet(path)
                rng = np.random.default_rng(0)
                nd, rc = [], []
                for _, g in scores.groupby("user", sort=False):
                    g = g.iloc[rng.permutation(len(g))]
                    y = (pd.to_numeric(g["label"], errors="coerce").fillna(0) > 0).astype(float).to_numpy()
                    if y.sum() == 0:
                        continue
                    pred = pd.to_numeric(g["_score"], errors="coerce").to_numpy(dtype=float)
                    pred = np.where(np.isfinite(pred), pred, -1e30)
                    nd.append(ndcg.evaluate_metric(y_pred=list(pred), y_true=list(y), topk=10))
                    rc.append(recall.evaluate_metric(y_pred=list(pred), y_true=list(y), topk=50))
                rows.append({"experiment": exp.experiment, "run_id": run_id,
                             "NDCG@10_neutral": float(np.mean(nd)) if nd else np.nan,
                             "RECALL@50_neutral": float(np.mean(rc)) if rc else np.nan})
    return pd.DataFrame(rows, columns=["experiment", "run_id", "NDCG@10_neutral", "RECALL@50_neutral"])


def feature_control(run_level: pd.DataFrame, experiments: pd.DataFrame):
    """Criterion 4: the same conditions with pure CF (LightFMModel: unseen items get -inf) and with
    recommenders that score unseen items through their text (LightFMTextModel: identity + text;
    LightFMContentModel: text only). Metrics are tie-neutral (neutral_run_metrics).
    Returns (wide table per condition, long table per run)."""
    feat = run_level["group"].astype(str).str.startswith("feat_")
    if not feat.any():
        return pd.DataFrame(), pd.DataFrame()
    cf = (run_level["group"].isin(["base", "step1", "m2", "m3"]) & (run_level["model"] == CF_MODEL)
          & (run_level["target_mode"].fillna("").astype(str) != "all"))
    long = run_level[feat | cf].copy()
    long["condition"] = long.apply(_condition, axis=1)
    long = long[long["condition"] != ""]
    keys = set(zip(*[long.loc[long["group"].astype(str).str.startswith("feat_"), c] for c in ("dataset", "strategy", "condition")]))
    long = long[[k in keys for k in zip(long["dataset"], long["strategy"], long["condition"])]]
    long = long.merge(neutral_run_metrics(experiments, long[["experiment", "run_id"]]), on=["experiment", "run_id"], how="left")

    agg = long.groupby(["dataset", "strategy", "condition", "model"]).agg(
        n_runs=("NDCG@10_neutral", "size"), cold=("test_pos_cold_frac", "mean"),
        ndcg10=("NDCG@10_neutral", "mean"), recall50=("RECALL@50_neutral", "mean")).reset_index()
    wide = agg.pivot_table(index=["dataset", "strategy", "condition"], columns="model",
                           values=["n_runs", "cold", "ndcg10", "recall50"])
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    for model in sorted(set(agg["model"]) - {CF_MODEL}):
        for metric in ("ndcg10", "recall50"):
            if f"{metric}_{model}" in wide and f"{metric}_{CF_MODEL}" in wide:
                wide[f"{metric}_gain_{model}"] = wide[f"{metric}_{model}"] - wide[f"{metric}_{CF_MODEL}"]
    return wide, long


# ----------------------------------------------------------------------------- figures
def fig_feature_control(long: pd.DataFrame, path: Path):
    """BEIR: tie-neutral NDCG@10 vs cold test positives, pure CF vs the two text-aware recommenders."""
    data = long[long["dataset"] == "beir"]
    if data.empty:
        return
    colors = {CF_MODEL: "0.55", "LightFMTextModel": "tab:blue", "LightFMContentModel": "tab:orange"}
    labels = {CF_MODEL: "LightFM (pure CF)", "LightFMTextModel": "LightFM identity + text",
              "LightFMContentModel": "LightFM text only"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for ax, strategy in zip(axes[0], ("QaU", "RaU")):
        for model, g in data[data["strategy"] == strategy].groupby("model"):
            ax.scatter(g["test_pos_cold_frac"], g["NDCG@10_neutral"], alpha=0.75, color=colors.get(model, "k"),
                       label=labels.get(model, model))
        ax.set_title(f"BEIR {strategy}")
        ax.set_xlabel("fraction of test positives that are cold")
        ax.set_ylabel("NDCG@10, tie-neutral (one point per run)")
        ax.set_xlim(-0.03, 1.03)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_cold(run_level: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    markers = {"base": "o", "step1": "s", "m1": "^", "m2": "D", "m3": "v", "ctx": "P"}
    for (dataset, strategy), g in _comparable(run_level).groupby(["dataset", "strategy"]):
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
        for (dataset, strategy), gs in g.groupby(["dataset", "strategy"]):
            gs = gs.sort_values("level")
            ax.errorbar(gs["level"], gs["mean"], yerr=[gs["mean"] - gs["ci_low"], gs["ci_high"] - gs["mean"]],
                        marker="o", capsize=3, label=f"{dataset} {strategy}")
        ax.set_xlabel(g["param"].iloc[0])
        ax.set_ylabel("NDCG@10 (95% bootstrap CI over runs)")
        ax.set_title(group.upper())
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
    features = pd.read_csv(out / "user_features.csv", dtype={"user": str, "run_id": str}, low_memory=False)
    metrics = load_metrics(experiments, structure[["experiment", "run_id"]])

    users = features.merge(metrics, on=["experiment", "run_id", "user"], how="inner")
    users.to_csv(out / "user_level_joined.csv", index=False)
    print(f"[analysis] joined {len(users)} pseudo-user rows from {users['experiment'].nunique()} experiment(s)")

    # run level: mean over the evaluated pseudo-users of each run
    metric_cols = [c for c in metrics.columns if "@" in c]
    run_level = (users.groupby(["experiment", "run_id"])[metric_cols].mean().reset_index()
                 .merge(structure, on=["experiment", "run_id"], how="left"))
    run_level.to_csv(out / "run_level.csv", index=False)

    id_cols = [c for c in ["experiment", "model", "dataset", "strategy", "group", "param", "level", "seed",
                           "target_mode", "top_k"] if c in run_level]
    agg = run_level.groupby(id_cols, dropna=False)
    summary = agg[metric_cols].mean().add_suffix("_mean").join(agg[metric_cols].std().add_suffix("_std"))
    struct_cols = [c for c in ["users", "pseudo_users", "context_users", "items", "density", "users_no_neighbor_frac",
                               "neighbors_per_user", "lcc_frac", "test_pos_cold_frac", "test_users"] if c in run_level]
    summary = summary.join(agg[struct_cols].mean()).join(agg.size().rename("n_runs")).reset_index()
    summary.to_csv(out / "summary.csv", index=False)

    base_users = users[users["group"] == "base"]
    tests = per_user_tests(base_users)
    tests.to_csv(out / "per_user_tests.csv", index=False)
    logit = logit_table(base_users)
    logit.to_csv(out / "logit.csv", index=False)
    curve_df = curves(run_level, structure)
    curve_df.to_csv(out / "curves.csv", index=False)
    med = mediation(run_level)
    med.to_csv(out / "mediation.csv", index=False)
    svp = structure_vs_performance(run_level)
    svp.to_csv(out / "structure_vs_performance.csv", index=False)
    if "model" not in run_level:  # structure.csv written before the model column existed
        run_level["model"] = run_level["experiment"].map(dict(zip(experiments["experiment"], experiments["model"])))
    fc, fc_long = feature_control(run_level, experiments)
    if not fc.empty:
        fc.to_csv(out / "feature_control.csv", index=False)
        fc_long.to_csv(out / "feature_control_runs.csv", index=False)
        fig_feature_control(fc_long, out / "fig_feature_control.png")

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
        print(curve_df[curve_df["target"] == "NDCG@10"][["group", "dataset", "strategy", "level", "n_runs", "mean", "ci_low",
                                                          "ci_high", "test_pos_cold_frac", "trend_spearman",
                                                          "trend_p_value"]].round(4).to_string(index=False))
    if not svp.empty:
        print("\n=== hypothesis 2: structure of the adapted matrix vs NDCG@10 (runs pooled per strategy) ===")
        print(svp[["strategy", "datasets", "n_runs", "label", "spearman", "p_value", "r2_alone",
                   "delta_r2_beyond_coverage"]].round(3).to_string(index=False))
    if not fc.empty:
        print("\n=== criterion 4: pure CF vs text-aware LightFM (tie-neutral metrics) ===")
        cols = ["dataset", "strategy", "condition", f"cold_{CF_MODEL}"] + [
            c for m in (CF_MODEL, "LightFMTextModel", "LightFMContentModel") for c in (f"ndcg10_{m}", f"recall50_{m}") if c in fc]
        print(fc[[c for c in cols if c in fc]].round(4).to_string(index=False))
    if not med.empty:
        print("\n=== does the cold fraction explain M1/M2/ctx? (residual vs line fitted on base + M3) ===")
        print(med[["dataset", "strategy", "group", "level", "n_runs", "mean_cold", "mean_ndcg", "mean_residual",
                   "ci_low", "ci_high", "beyond_coverage", "r2_reference"]].round(4).to_string(index=False))
    print(f"\n[analysis] outputs in {out}/")


if __name__ == "__main__":
    main()
