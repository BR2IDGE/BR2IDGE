import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, List, Tuple
from scipy.stats import ttest_rel, wilcoxon


SIGNIFICANCE_TESTS: Dict[str, Callable[..., None]] = {}


def register_test(name: str) -> Callable:
    name = name.lower()

    def decorator(fn: Callable[..., None]) -> Callable[..., None]:
        SIGNIFICANCE_TESTS[name] = fn
        return fn

    return decorator


def _extract_run_id(path: Path, experiment_name: str) -> str:
    stem = path.stem 
    prefix = f"{experiment_name}_"
    suffix = "_metrics"
    core = stem
    if core.startswith(prefix):
        core = core[len(prefix):]
    if core.endswith(suffix):
        core = core[: -len(suffix)]
    return core or stem


def _load_aggregated_metrics(
    metrics_dir: Path, experiment_name: str
) -> pd.DataFrame | None:
    compact = metrics_dir / f"{experiment_name}_metrics_all.csv"
    if compact.exists():
        df = pd.read_csv(compact)
        required = {"RunId", "Model", "K", "Metric", "Value"}
        if not required.issubset(df.columns):
            print(f"[stats] Invalid metrics_all: {compact} columns={df.columns.tolist()}")
            return None
        print(f"[stats] Aggregated metrics (compact): {len(df)} rows, {df['RunId'].nunique()} runs.")
        return df

    pattern = f"{experiment_name}_*_metrics.csv"
    files = sorted(metrics_dir.glob(pattern))
    if not files:
        print(f"[stats] No files found (compact or '{pattern}') in {metrics_dir}")
        return None

    frames: List[pd.DataFrame] = []
    for path in files:
        try:
            tmp = pd.read_csv(path)
        except Exception as e:
            print(f"[stats] Error reading '{path}': {e}")
            continue
        required_cols = {"Model", "K", "Metric", "Value"}
        if not required_cols.issubset(tmp.columns):
            continue
        run_id = _extract_run_id(path, experiment_name)
        tmp["RunId"] = run_id
        frames.append(tmp)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    print(f"[stats] Aggregated metrics (fallback): {len(df)} rows, {df['RunId'].nunique()} runs.")
    return df

# Wilcoxon
@register_test("wilcoxon")
def wilcoxon_signed_rank_test(
    df: pd.DataFrame,
    model_pairs: List[Tuple[str, str]],
    experiment_name: str,
    metrics_dir: Path,
    alpha: float = 0.05,
    symbol: str = "\u25B2",
    min_runs: int = 1,
) -> None:
    required_cols = {"RunId", "Model", "Metric", "K", "Value"}
    if not required_cols.issubset(df.columns):
        print(f"[wilcoxon] DataFrame is missing required columns {required_cols}. Aborting test.")
        return

    df = df.copy()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df = df.dropna(subset=["Value", "K", "Model", "Metric", "RunId"])
    df["K"] = df["K"].astype(int)

    if df.empty:
        print("[wilcoxon] Metrics DataFrame is empty after cleaning.")
        return

    df_mean = (
        df.groupby(["Model", "Metric", "K"], as_index=False)["Value"]
        .mean()
        .rename(columns={"Value": "Mean"})
    )

    df_marked = df_mean.copy()
    df_marked["Value"] = df_marked["Mean"].apply(lambda v: f"{float(v):.6e}")
    df_marked = df_marked.drop(columns=["Mean"])

    diagnostics: List[dict] = []

    for (metric, k), group in df.groupby(["Metric", "K"], sort=False):
        wide = group.pivot_table(
            index="RunId",
            columns="Model",
            values="Value",
            aggfunc="mean",
        )

        for model_a, model_b in model_pairs:
            if model_a not in wide.columns or model_b not in wide.columns:
                continue

            pair_df = wide[[model_a, model_b]].dropna()

            if len(pair_df) < int(min_runs):
                print(
                    f"[wilcoxon] Insufficient paired runs ({len(pair_df)}/{min_runs}) "
                    f"for {model_a} vs {model_b} (Metric={metric}, K={k}). Skipping."
                )
                continue

            x = pair_df[model_a].to_numpy()
            y = pair_df[model_b].to_numpy()
            diff = x - y

            nonzero_mask = diff != 0
            if nonzero_mask.sum() == 0:
                print(
                    f"[wilcoxon] All differences are zero for {model_a} vs {model_b}, "
                    f"Metric={metric}, K={k}. Skipping."
                )
                continue

            stat, pval = wilcoxon(
                x, y,
                zero_method="wilcox",
                alternative="two-sided",
            )

            mean_a = float(np.mean(x))
            mean_b = float(np.mean(y))
            mean_diff = float(np.mean(diff))
            significant = bool(pval < alpha)

            if mean_diff > 0:
                winner = model_a
            elif mean_diff < 0:
                winner = model_b
            else:
                winner = "tie"

            diagnostics.append(
                {
                    "Test": "Wilcoxon signed-rank",
                    "Experiment": experiment_name,
                    "Metric": metric,
                    "K": int(k),
                    "Model_A": model_a,
                    "Model_B": model_b,
                    "N_runs": int(len(pair_df)),
                    "N_effective": int(nonzero_mask.sum()),
                    "statistic": float(stat),
                    "p_value": float(pval),
                    "mean_A": mean_a,
                    "mean_B": mean_b,
                    "mean_diff": mean_diff,
                    "Winner": winner,
                    "Significant": significant,
                    "alpha": float(alpha),
                }
            )

            if significant and winner in {model_a, model_b}:
                mask = (
                    (df_marked["Model"] == winner)
                    & (df_marked["Metric"] == metric)
                    & (df_marked["K"] == int(k))
                )
                df_marked.loc[mask, "Value"] = df_marked.loc[mask, "Value"].astype(str).apply(
                    lambda s: s if str(s).endswith(symbol) else str(s) + symbol
                )

    if not diagnostics:
        print("[wilcoxon] No model pairs had enough paired runs to apply the test.")
        return

    marked_path = metrics_dir / f"{experiment_name}_wilcoxon_marked.csv"
    df_marked.to_csv(marked_path, index=False)
    print(f"[wilcoxon] Marked summary saved to: {marked_path}")

    diag_path = metrics_dir / f"{experiment_name}_wilcoxon_diagnostics.csv"
    pd.DataFrame(diagnostics).to_csv(diag_path, index=False)
    print(f"[wilcoxon] Detailed diagnostics saved to: {diag_path}")


@register_test("ttest")
@register_test("t_test")
@register_test("t-test")
@register_test("paired_ttest")
@register_test("ttest_rel")
def paired_t_test(
    df: pd.DataFrame,
    model_pairs: List[Tuple[str, str]],
    experiment_name: str,
    metrics_dir: Path,
    alpha: float = 0.05,
    symbol: str = "\u25B2",
    min_runs: int = 2,
) -> None:
    required_cols = {"RunId", "Model", "Metric", "K", "Value"}
    if not required_cols.issubset(df.columns):
        print(f"[ttest] DataFrame is missing required columns {required_cols}. Aborting test.")
        return

    df = df.copy()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df = df.dropna(subset=["Value", "K", "Model", "Metric", "RunId"])
    df["K"] = df["K"].astype(int)

    if df.empty:
        print("[ttest] Metrics DataFrame is empty after cleaning.")
        return

    df_mean = (
        df.groupby(["Model", "Metric", "K"], as_index=False)["Value"]
        .mean()
        .rename(columns={"Value": "Mean"})
    )

    df_marked = df_mean.copy()
    df_marked["Value"] = df_marked["Mean"].apply(lambda v: f"{float(v):.6e}")
    df_marked = df_marked.drop(columns=["Mean"])

    diagnostics: List[dict] = []
    required_runs = max(int(min_runs), 2)

    for (metric, k), group in df.groupby(["Metric", "K"], sort=False):
        wide = group.pivot_table(
            index="RunId",
            columns="Model",
            values="Value",
            aggfunc="mean",
        )

        for model_a, model_b in model_pairs:
            if model_a not in wide.columns or model_b not in wide.columns:
                continue

            pair_df = wide[[model_a, model_b]].dropna()

            if len(pair_df) < required_runs:
                print(
                    f"[ttest] Insufficient paired runs ({len(pair_df)}/{required_runs}) "
                    f"for {model_a} vs {model_b} (Metric={metric}, K={k}). Skipping."
                )
                continue

            x = pair_df[model_a].to_numpy()
            y = pair_df[model_b].to_numpy()
            diff = x - y

            res = ttest_rel(x, y, nan_policy="omit")
            try:
                stat = float(res.statistic)
                pval = float(res.pvalue)
            except AttributeError:
                stat = float(res[0])
                pval = float(res[1])

            if not np.isfinite(stat) or not np.isfinite(pval):
                if np.allclose(diff, 0.0):
                    stat = 0.0
                    pval = 1.0
                else:
                    print(
                        f"[ttest] Non-finite result for {model_a} vs {model_b} "
                        f"(Metric={metric}, K={k}). Skipping."
                    )
                    continue

            mean_a = float(np.mean(x))
            mean_b = float(np.mean(y))
            mean_diff = float(np.mean(diff))
            significant = bool(pval < alpha)

            if mean_diff > 0:
                winner = model_a
            elif mean_diff < 0:
                winner = model_b
            else:
                winner = "tie"

            diagnostics.append(
                {
                    "Test": "Paired t-test",
                    "Experiment": experiment_name,
                    "Metric": metric,
                    "K": int(k),
                    "Model_A": model_a,
                    "Model_B": model_b,
                    "N_runs": int(len(pair_df)),
                    "N_effective": int(len(pair_df)),
                    "statistic": float(stat),
                    "p_value": float(pval),
                    "mean_A": mean_a,
                    "mean_B": mean_b,
                    "mean_diff": mean_diff,
                    "Winner": winner,
                    "Significant": significant,
                    "alpha": float(alpha),
                }
            )

            if significant and winner in {model_a, model_b}:
                mask = (
                    (df_marked["Model"] == winner)
                    & (df_marked["Metric"] == metric)
                    & (df_marked["K"] == int(k))
                )
                df_marked.loc[mask, "Value"] = df_marked.loc[mask, "Value"].astype(str).apply(
                    lambda s: s if str(s).endswith(symbol) else str(s) + symbol
                )

    if not diagnostics:
        print("[ttest] No model pairs had enough paired runs to apply the test.")
        return

    marked_path = metrics_dir / f"{experiment_name}_ttest_marked.csv"
    df_marked.to_csv(marked_path, index=False)
    print(f"[ttest] Marked summary saved to: {marked_path}")

    diag_path = metrics_dir / f"{experiment_name}_ttest_diagnostics.csv"
    pd.DataFrame(diagnostics).to_csv(diag_path, index=False)
    print(f"[ttest] Detailed diagnostics saved to: {diag_path}")


# Orchestrator
def run_significance_analysis(
    metrics_dir: Path,
    experiment_name: str,
    model_pairs: List[Tuple[str, str]],
    tests: List[str] | None = None,
    alpha: float = 0.05,
    symbol: str = "\u25B2",
    min_runs: int = 6,
) -> None:
    if not metrics_dir.exists():
        print(f"[stats] Metrics directory not found: {metrics_dir}")
        return

    tests = tests if isinstance(tests, list) else []
    if len(tests) == 0:
        print("[stats] Empty tests list. No statistical test will be executed.")
        return

    df = _load_aggregated_metrics(metrics_dir, experiment_name)
    if df is None or df.empty:
        print("[stats] Metrics are empty or incompatible; no statistical test will be executed.")
        return

    if not model_pairs:
        print("[stats] No model pairs provided for comparison.")
        return

    for test_name in tests:
        test_name_l = str(test_name).lower().strip()
        fn = SIGNIFICANCE_TESTS.get(test_name_l)
        if fn is None:
            print(
                f"[stats] Test '{test_name}' not recognized. "
                f"Available: {sorted(SIGNIFICANCE_TESTS.keys())}"
            )
            continue

        print(f"[stats] Running statistical test: {test_name_l}")
        fn(
            df=df.copy(),
            model_pairs=model_pairs,
            experiment_name=experiment_name,
            metrics_dir=metrics_dir,
            alpha=alpha,
            symbol=symbol,
            min_runs=min_runs,
        )
