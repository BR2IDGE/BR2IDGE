import argparse
import copy
import json
import sys
import importlib
import joblib
import csv
import re
import contextlib
from pathlib import Path
from time import perf_counter
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from checkpoint.paths import config_hash, ensure_run_dir
from checkpoint.api import resume_if_exists, save_if_enabled

def _load_valid_models(module_path):
    try:
        mod = importlib.import_module(module_path)
        valid_names = getattr(mod, "__all__", [])
        return set(valid_names)
    except ImportError as e:
        print(f"[warning] Could not import {module_path} for validation: {e}")
        return set()

VALID_SEARCH_MODELS = _load_valid_models("search_recs.search.model")
VALID_RECS_MODELS = _load_valid_models("search_recs.recs.model")

print(f"[system] Search models loaded: {sorted(list(VALID_SEARCH_MODELS))}")
print(f"[system] Recs models loaded: {sorted(list(VALID_RECS_MODELS))}")

_ICONS = {"run": "▶", "ok": "✓", "cache": "⎇", "skip": "↷", "error": "✗"}

def stage_start(name: str) -> float:
    print(f"\n{'=' * 70}")
    print(f"{_ICONS['run']} {name.upper()}")
    print(f"{'=' * 70}", flush=True)
    return perf_counter()

def stage_end(name: str, t0: float, tag: str = "ok"):
    dt = perf_counter() - t0
    if tag == "cache":
        suffix = " (cache)"
    elif tag == "skip":
        suffix = " (skip)"
    elif tag == "na":
        suffix = " (n/a)"
    else:
        suffix = f" ({dt:.2f}s)"
    print(f"{_ICONS['ok']} {name}{suffix}", flush=True)

def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

def _safe_len(obj):
    try:
        return len(obj)
    except Exception:
        return None

def _normalize_dataset_name(exp_config: dict) -> str:
    ds = exp_config.get("dataset")
    if isinstance(ds, dict):
        return (ds.get("name") or ds.get("dataset_name") or "").strip()
    if isinstance(ds, str):
        return ds.strip()
    return (exp_config.get("dataset_name") or "").strip()


def _normalize_model_class_name(model_entry) -> str:
    if isinstance(model_entry, dict):
        return (
            model_entry.get("name")
            or model_entry.get("class_name")
            or model_entry.get("model_type")
            or model_entry.get("model_name")
            or ""
        ).strip()
    if isinstance(model_entry, str):
        return model_entry.strip()
    return ""

def validate_models_for_task(model_names: list[str], task_type: str):
    print(f"[validation] Verifying {len(model_names)} model(s) for task '{task_type}'...")
    
    for name in model_names:
        if task_type == "search":
            if name not in VALID_SEARCH_MODELS:
                if name in VALID_RECS_MODELS:
                    raise ValueError(f"CONFIGURATION ERROR: Model '{name}' is for RECOMMENDATION, but current task_type is SEARCH.")
                if not VALID_SEARCH_MODELS:
                    print(f"[warning] Validation skipped for '{name}' (search list empty).")
                else:
                    raise ValueError(f"ERROR: Model '{name}' is not recognized as a valid SEARCH model.\nValid: {sorted(list(VALID_SEARCH_MODELS))}")
                
        elif task_type == "recs":
            if name not in VALID_RECS_MODELS:
                if name in VALID_SEARCH_MODELS:
                    raise ValueError(f"CONFIGURATION ERROR: Model '{name}' is for SEARCH, but current task_type is RECS.")
                if not VALID_RECS_MODELS:
                    print(f"[warning] Validation skipped for '{name}' (recs list empty).")
                else:
                    raise ValueError(f"ERROR: Model '{name}' is not recognized as a valid RECS model.\nValid: {sorted(list(VALID_RECS_MODELS))}")
    
    print(f"{_ICONS['ok']} Model validation successful.")

# CHECKPOINT helpers 

def _has_any_checkpoint_files(run_dir: Path) -> bool:
    if not run_dir.exists() or not run_dir.is_dir():
        return False

    if (run_dir / "model.pkl").exists():
        return True

    if (run_dir / "last.pt").exists():
        return True

    if (run_dir / "keras_weights.weights.h5").exists():
        return True

    if (run_dir / "lightfm.npz").exists():
        return True

    if (run_dir / "als_factors.npz").exists():
        return True
    if (run_dir / "nmf_factors.npz").exists():
        return True

    if any(run_dir.glob("libreco_model*")):
        return True

    return False


def _try_resume_from_previous_runs(model_key: str, model, current_run_dir: Path):
    runs_root = current_run_dir.parent  
    if not runs_root.exists() or not runs_root.is_dir():
        return model, False

    candidates = []
    for path in runs_root.iterdir():
        if not path.is_dir():
            continue
        if path.resolve() == current_run_dir.resolve():
            continue
        if _has_any_checkpoint_files(path):
            candidates.append(path)

    candidates = sorted(candidates, key=lambda x: x.name, reverse=True)

    for candidate in candidates:
        print(f"[checkpoint] Attempting RESUME from previous run: {candidate}")
        model2, resumed = resume_if_exists(model_key, model, candidate)
        if resumed:
            print(f"[checkpoint] Resume OK from: {candidate}")
            return model2, True

    return model, False


def _find_prev_run_with_file(current_run_dir: Path, filename: str) -> Path | None:
    try:
        path_current = current_run_dir / filename
        if path_current.exists():
            return path_current

        runs_root = current_run_dir.parent 
        if not runs_root.exists():
            return None

        candidates = []
        for path in runs_root.iterdir():
            if not path.is_dir():
                continue
            if path.resolve() == current_run_dir.resolve():
                continue
            file_path = path / filename
            if file_path.exists():
                candidates.append(file_path)

        candidates = sorted(candidates, key=lambda x: x.parent.name, reverse=True)
        return candidates[0] if candidates else None
    except Exception:
        return None


def _is_hybrid_retrieval_as_user(exp_config: dict, task_type: str) -> bool:
    if task_type != "recs":
        return False
    if not bool(exp_config.get("hybrid", False)):
        return False

    strat = exp_config.get("hybridStrategie", exp_config.get("hybridStrategy", ""))
    s = str(strat).strip().lower()
    s = s.replace("_", "-")
    s = s.replace(" ", "-")
    s = re.sub(r"-+", "-", s)

    return s in { "retrieval-as-user", "retrievalasuser", "retrieval-asuser", "retrieval-as-user", "retrieval-as-user ", }


_SEARCH_HYBRID_STRATEGIES = {"centroid-vector", "tag-query"}


def _norm_search_hybrid_strategy(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s)
    s = s.replace("querry", "query")
    return s


def _inject_search_hybrid_task(real_model_config: dict, exp_config: dict, task_type: str) -> dict:
    if task_type != "search":
        return real_model_config
    exp_cfg = exp_config or {}
    if not bool(exp_cfg.get("hybrid", False)):
        return real_model_config
    strategy = exp_cfg.get("hybridStrategie", exp_cfg.get("hybridStrategy", ""))
    norm = _norm_search_hybrid_strategy(str(strategy))
    if norm not in _SEARCH_HYBRID_STRATEGIES:
        if norm:
            print(f"[model] Warning: unrecognized hybridStrategy '{strategy}' for search. Valid: {sorted(_SEARCH_HYBRID_STRATEGIES)}")
        return real_model_config
    config = copy.deepcopy(real_model_config)
    config.setdefault("parameters", {})["task"] = "hybrid"
    print(f"[model] Hybrid search: hybridStrategy='{norm}' → injecting task='hybrid' into model parameters")
    return config


def _norm_key_simple(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _resolve_retrieval_as_user_dataset_cfg(dataset_key: str, ds_dir: Path) -> Path:
    k = _norm_key_simple(dataset_key)

    if "movie" in k or "ml25" in k or "movielens" in k:
        suffix = "ml25m"
    elif "lastfm" in k:
        suffix = "lastfm"
    elif "amazonelectronics" in k:
        suffix = "amazon"
    else:
        suffix = k

    direct = ds_dir / f"retrieval_as_user_{suffix}.json"
    if direct.exists():
        return direct

    cands = sorted(ds_dir.glob("retrieval_as_user_*.json"))
    if not cands:
        raise FileNotFoundError(f"No retrieval_as_user_*.json found in {ds_dir}")

    for p in cands:
        stem = _norm_key_simple(p.stem)
        if _norm_key_simple(suffix) in stem or k in stem:
            return p

    raise FileNotFoundError(
        f"Hybrid retrieval-as-user requested, but could not map dataset '{dataset_key}' "
        f"to a retrieval_as_user_*.json file in {ds_dir}."
    )


def resolve_dataset_config(exp_config: dict, task_type: str, base_path: Path) -> Path:
    ds_entry = exp_config.get("dataset", {})
    if isinstance(ds_entry, dict) and "config_path" in ds_entry:
        cfg_path = Path(ds_entry["config_path"])
        return cfg_path if cfg_path.is_absolute() else base_path / cfg_path

    dataset_key = _normalize_dataset_name(exp_config)
    if not dataset_key:
        raise ValueError("General config requires 'dataset.name' or 'dataset'.")

    ds_dir = base_path / "config_files" / "datasets"
    if not ds_dir.exists():
        raise FileNotFoundError(f"Datasets folder not found: {ds_dir}")

    if _is_hybrid_retrieval_as_user(exp_config, task_type):
        p = _resolve_retrieval_as_user_dataset_cfg(dataset_key, ds_dir)
        return p
    
    matches = []
    for p in ds_dir.glob("*.json"):
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        dl = cfg.get("dataloader", {})
        ds_name = (
            dl.get("dataset_name")
            or dl.get("dataset")
            or dl.get("name")
            or ""
        )
        if str(ds_name).lower() == dataset_key.lower():
            matches.append(p)

    if not matches:
        raise FileNotFoundError(
            f"No dataset config found for '{dataset_key}' in {ds_dir}"
        )

    if len(matches) == 1:
        return matches[0]

    def is_search_file(p: Path) -> bool:
        return "search" in p.stem.lower()

    if task_type == "search":
        search_candidates = [p for p in matches if is_search_file(p)]
        return search_candidates[0] if search_candidates else matches[0]
    else:
        non_search = [p for p in matches if not is_search_file(p)]
        return non_search[0] if non_search else matches[0]


def resolve_model_config(model_entry, base_path: Path) -> tuple[Path, str]:
    model_class_name = _normalize_model_class_name(model_entry)
    
    if isinstance(model_entry, dict) and "config_path" in model_entry:
        cfg_path = Path(model_entry["config_path"])
        cfg_path = cfg_path if cfg_path.is_absolute() else base_path / cfg_path
        return cfg_path, model_class_name

    if not model_class_name:
        raise ValueError(f"Invalid model entry: {model_entry}")

    models_dir = base_path / "config_files" / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"Models folder not found: {models_dir}")

    matches = []
    search_key = model_class_name.lower()
    for p in models_dir.glob("*.json"):
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        m_cfg = cfg.get("model", cfg)
        names = []
        for k in ["model_type", "model_name", "name"]:
            v = m_cfg.get(k)
            if isinstance(v, str):
                names.append(v.lower())

        if search_key in names:
            matches.append(p)

    if not matches:
        raise FileNotFoundError(
            f"No model config found for '{model_class_name}' in {models_dir}"
        )

    return matches[0], model_class_name


def get_model_class(model_class_name: str, task_type: str):
    module_path = (
        "search_recs.search.model" if task_type == "search" else "search_recs.recs.model"
    )
    module = importlib.import_module(module_path)
    try:
        return getattr(module, model_class_name)
    except AttributeError as e:
        raise ImportError(f"Model class '{model_class_name}' not found in {module_path}") from e


def get_dataloader_entry(dataset_name: str, task_type: str):
    module_path = (
        "search_recs.search.dataloader" if task_type == "search" else "search_recs.recs.dataloader"
    )
    dl_module = importlib.import_module(module_path)

    if hasattr(dl_module, "get_loader"):
        return dl_module.get_loader(dataset_name)

    if hasattr(dl_module, "REGISTRY"):
        registry = dl_module.REGISTRY
        if dataset_name in registry:
            return registry[dataset_name]
        if dataset_name.lower() in registry:
            return registry[dataset_name.lower()]
        raise ValueError(
            f"Dataset '{dataset_name}' is not in {module_path}.REGISTRY. "
            f"Options: {list(registry.keys())}"
        )

    raise ImportError(f"Module {module_path} has neither 'get_loader' nor 'REGISTRY'.")


_TABLE_DECIMALS = 12
_LONG_DECIMALS = 12

def _latex_escape(s: str) -> str:
    s = str(s)
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("_", "\\_")
    return s

def _latex_label(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "table"

def _fmt_score(v, decimals: int) -> str:
    try:
        v = float(v)
    except Exception:
        return str(v)
    return f"{v:.{decimals}f}"

def normalize_results_dict(results: dict) -> dict[int, dict[str, float]]:

    out: dict[int, dict[str, float]] = {}
    for k, md in (results or {}).items():
        kk = int(k)
        out[kk] = {}
        for m, v in (md or {}).items():
            out[kk][str(m).upper().strip()] = float(v)
    return out

def _results_to_wide_df(results_norm: dict[int, dict[str, float]], metrics_order: list[str]) -> pd.DataFrame:
    ks = sorted(list(results_norm.keys()))
    cols = [m.upper().strip() for m in metrics_order]
    rows = []
    for k in ks:
        row = {"K": int(k)}
        for m in cols:
            row[m] = results_norm.get(k, {}).get(m, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("K").reset_index(drop=True)

def append_metrics_long_csv(csv_path: Path, model_name: str, results_norm: dict[int, dict[str, float]], run_id: str):

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for k, md in results_norm.items():
        for metric, value in (md or {}).items():
            rows.append({
                "RunId": run_id,  
                "Model": model_name,
                "K": int(k),
                "Metric": str(metric).upper(),
                "Value": _fmt_score(value, _LONG_DECIMALS),
            })

    file_exists = csv_path.exists()
    with open(csv_path, mode="a" if file_exists else "w", newline="", encoding="utf-8") as f:
        fieldnames = ["RunId", "Model", "K", "Metric", "Value"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerows(rows)

def write_metrics_wide_csv(csv_path: Path, results_norm: dict[int, dict[str, float]], metrics_order: list[str], meta: dict | None = None):
    df = _results_to_wide_df(results_norm, metrics_order)

    for c in df.columns:
        if c == "K":
            df[c] = df[c].astype(int)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8",
        float_format=f"%.{_TABLE_DECIMALS}f",
        na_rep="",
    )

def write_metrics_wide_latex(tex_path: Path, results_norm: dict[int, dict[str, float]], metrics_order: list[str], caption: str, meta: dict | None = None):

    df = _results_to_wide_df(results_norm, metrics_order)
    cols = [c for c in df.columns if c != "K"]
    align = "l" + ("r" * len(cols))
    ncols_total = 1 + len(cols) 

    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    lines.append(f"\\label{{tab:{_latex_label(caption)}}}")

    if len(cols) >= 6:
        lines.append("\\resizebox{\\linewidth}{!}{%")

    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append("\\toprule")

    if meta:
        meta_line = (
            f"Dataset: {meta.get('dataset','')}  "
            f"| Model: {meta.get('model','')}  "
            f"| Task: {meta.get('task','')}  "
            f"| Run: {meta.get('run','')}"
        )
        lines.append(f"\\multicolumn{{{ncols_total}}}{{c}}{{\\textbf{{{_latex_escape(meta_line)}}}}} \\\\")
        lines.append("\\midrule")

    header = ["\\textbf{K}"] + [f"\\textbf{{{_latex_escape(c)}}}" for c in cols]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for _, r in df.iterrows():
        row_vals = [str(int(r["K"]))]
        for c in cols:
            v = r[c]
            row_vals.append("" if pd.isna(v) else _fmt_score(v, _TABLE_DECIMALS))
        lines.append(" & ".join(row_vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    if len(cols) >= 6:
        lines.append("}")  

    lines.append("\\end{table}")

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[latex] Wide table generated at: {tex_path}")


def _colkey(metric: str, k: int) -> str:
    return f"{metric.upper()}@{int(k)}"

def write_summary_csv(csv_path: Path, df_mean: pd.DataFrame):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_mean.to_csv(csv_path, index=True, encoding="utf-8")

def write_summary_tex_like_image(tex_path: Path, dataset_title: str, df_mean: pd.DataFrame, metrics_order: list[str], top_ks: list[int], caption: str, bold_best: bool = True):

    metrics_order = [m.upper().strip() for m in metrics_order]
    top_ks = [int(k) for k in top_ks]

    ordered_cols = []
    for m in metrics_order:
        for k in top_ks:
            ck = _colkey(m, k)
            if ck in df_mean.columns:
                ordered_cols.append(ck)

    df = df_mean.copy()
    df = df[ordered_cols] if ordered_cols else df

    best = {}
    if bold_best:
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            best[c] = float(s.max()) if len(s.dropna()) else None

    align = "|l|" + "|".join(["r"] * len(df.columns)) + "|"

    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    lines.append(f"\\label{{tab:{_latex_label(caption)}}}")
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append("\\hline")

    lines.append(
        "\\textbf{Dataset} & "
        + f"\\multicolumn{{{len(df.columns)}}}{{c|}}{{\\textbf{{{_latex_escape(dataset_title)}}}}} \\\\"
    )
    lines.append("\\hline")

    metric_groups = []
    col_cursor = 0
    for m in metrics_order:
        cols_m = [c for c in df.columns if c.startswith(m + "@")]
        if not cols_m:
            continue
        metric_groups.append((m, len(cols_m)))

    row = ["\\textbf{Metrics}"]
    for (m, span) in metric_groups:
        row.append(f"\\multicolumn{{{span}}}{{c|}}{{\\textbf{{{_latex_escape(m)}}}}}")
        col_cursor += span
    missing = len(df.columns) - sum(span for _, span in metric_groups)
    if missing > 0:
        row.append(f"\\multicolumn{{{missing}}}{{c|}}{{}}")
    lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")

    row_k = ["\\textbf{K}"]
    for m in metrics_order:
        for k in top_ks:
            ck = _colkey(m, k)
            if ck in df.columns:
                row_k.append(f"\\textbf{{{int(k)}}}")
    while len(row_k) < 1 + len(df.columns):
        row_k.append("")
    lines.append(" & ".join(row_k) + " \\\\")
    lines.append("\\hline")

    for model_name, r in df.iterrows():
        vals = [str(model_name)]
        for c in df.columns:
            v = r[c]
            fv = "" if pd.isna(v) else _fmt_score(v, _TABLE_DECIMALS)
            if bold_best and best.get(c) is not None:
                try:
                    if abs(float(v) - best[c]) <= 1e-12:
                        fv = f"\\textbf{{{fv}}}"
                except Exception:
                    pass
            vals.append(fv)
        lines.append(" & ".join(vals) + " \\\\")
        lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[latex] Summary (image style) generated at: {tex_path}")


def generate_compact_reports(metrics_all_path: Path, out_dir: Path, experiment_name: str, dataset_title: str, metrics_order: list[str], top_ks: list[int]):

    if not metrics_all_path.exists():
        print(f"[compact] metrics_all does not exist: {metrics_all_path}")
        return

    df = pd.read_csv(metrics_all_path)
    required = {"RunId", "Model", "K", "Metric", "Value"}
    if not required.issubset(df.columns):
        print(f"[compact] metrics_all invalid. Columns: {df.columns.tolist()}")
        return

    df["K"] = pd.to_numeric(df["K"], errors="coerce").astype(int)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Metric"] = df["Metric"].astype(str).str.upper()

    for model in sorted(df["Model"].unique()):
        g = df[df["Model"] == model].copy()
        g["Col"] = g.apply(lambda r: _colkey(r["Metric"], r["K"]), axis=1)
        wide = g.pivot_table(index="RunId", columns="Col", values="Value", aggfunc="mean").sort_index()

        ordered_cols = []
        for m in [x.upper().strip() for x in metrics_order]:
            for k in [int(x) for x in top_ks]:
                ck = _colkey(m, k)
                if ck in wide.columns:
                    ordered_cols.append(ck)
        wide = wide[ordered_cols] if ordered_cols else wide

        model_csv = out_dir / f"{experiment_name}_{model}_runs.csv"
        wide.to_csv(model_csv, index=True, encoding="utf-8")
        print(f"[compact] Runs CSV: {model_csv}")

        model_tex = out_dir / f"{experiment_name}_{model}_runs.tex"
        lines = []
        align = "|l|" + "|".join(["r"] * len(wide.columns)) + "|"
        lines.append("\\begin{table}[ht]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\renewcommand{\\arraystretch}{1.1}")
        lines.append(f"\\caption{{{_latex_escape(experiment_name)} — Runs — { _latex_escape(model) }}}")
        lines.append(f"\\label{{tab:{_latex_label(experiment_name + '_runs_' + model)}}}")
        lines.append(f"\\begin{{tabular}}{{{align}}}")
        lines.append("\\hline")
        lines.append("\\textbf{RunId} & " + " & ".join([f"\\textbf{{{_latex_escape(c)}}}" for c in wide.columns]) + " \\\\")
        lines.append("\\hline")
        for run_id, rr in wide.iterrows():
            row = [str(run_id)]
            for c in wide.columns:
                v = rr[c]
                row.append("" if pd.isna(v) else _fmt_score(v, _TABLE_DECIMALS))
            lines.append(" & ".join(row) + " \\\\")
            lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        model_tex.write_text("\n".join(lines), encoding="utf-8")
        print(f"[compact] Runs TEX: {model_tex}")

    mean_df = (
        df.groupby(["Model", "Metric", "K"], as_index=False)["Value"]
        .mean()
    )
    mean_df["Col"] = mean_df.apply(lambda r: _colkey(r["Metric"], r["K"]), axis=1)
    summary = mean_df.pivot_table(index="Model", columns="Col", values="Value", aggfunc="mean")

    ordered_cols = []
    for m in [x.upper().strip() for x in metrics_order]:
        for k in [int(x) for x in top_ks]:
            ck = _colkey(m, k)
            if ck in summary.columns:
                ordered_cols.append(ck)
    summary = summary[ordered_cols] if ordered_cols else summary

    summary_csv = out_dir / f"{experiment_name}_summary.csv"
    write_summary_csv(summary_csv, summary)
    print(f"[compact] Summary CSV: {summary_csv}")

    summary_tex = out_dir / f"{experiment_name}_summary.tex"
    caption = f"{experiment_name} — Results (mean of runs)"
    write_summary_tex_like_image(
        tex_path=summary_tex,
        dataset_title=dataset_title,
        df_mean=summary,
        metrics_order=metrics_order,
        top_ks=top_ks,
        caption=caption,
        bold_best=True,
    )

# RECS helpers

_CAND_USER = ["user", "userId", "uid", "user_id"]
_CAND_ITEM = ["item", "itemId", "movieId", "iid", "item_id"]
_CAND_LABEL = ["label", "rating", "stars", "score"]
_CAND_TIME = ["time", "timestamp", "ts", "datetime"]


def _pick_col(df: pd.DataFrame, preferred: str | None, candidates: list[str]) -> str | None:
    if preferred and preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_recs_schema(df: pd.DataFrame, dl_params: dict) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError("[recs-schema] DataFrame empty after load_data().")

    user_src = _pick_col(df, dl_params.get("user_col"), _CAND_USER)
    item_src = _pick_col(df, dl_params.get("item_col"), _CAND_ITEM)
    label_src = _pick_col(df, dl_params.get("rating_col") or dl_params.get("label_col"), _CAND_LABEL)
    time_src = _pick_col(df, dl_params.get("time_col"), _CAND_TIME)

    out = df
    if "user" not in out.columns:
        if not user_src:
            raise ValueError(f"[recs-schema] User column not found. Columns: {list(out.columns)}")
        out = out.copy()
        out["user"] = out[user_src]

    if "item" not in out.columns:
        if not item_src:
            raise ValueError(f"[recs-schema] Item column not found. Columns: {list(out.columns)}")
        if out is df:
            out = out.copy()
        out["item"] = out[item_src]

    if "label" not in out.columns:
        if not label_src:
            raise ValueError(f"[recs-schema] Rating/label column not found. Columns: {list(out.columns)}")
        if out is df:
            out = out.copy()
        out["label"] = out[label_src]

    if "time" not in out.columns and time_src is not None:
        if out is df:
            out = out.copy()
        out["time"] = out[time_src]

    return out


def apply_min_rating_filter_recs(df: pd.DataFrame, dl_params: dict) -> pd.DataFrame:
    min_rating = dl_params.get("min_rating", None)
    if min_rating is None:
        return df

    thr = float(min_rating)
    if "label" not in df.columns:
        raise ValueError("[recs-min] Column 'label' does not exist (ensure_recs_schema should have created it).")

    s = pd.to_numeric(df["label"], errors="coerce")
    bad = int(s.isna().sum())
    if bad > 0:
        print(f"[recs-min] Warning: {bad} non-numeric labels became NaN and will be removed.")

    df2 = df.copy()
    df2["label"] = s
    df2 = df2.dropna(subset=["label"])

    before = len(df2)
    df2 = df2[df2["label"] >= thr].copy()
    after = len(df2)
    print(f"[recs-min] min_rating>={thr} applied on 'label': {before} -> {after} (removed={before-after})")

    if after == 0:
        raise ValueError(f"[recs-min] Dataset became empty after min_rating>={thr}.")
    return df2


def evaluate_search_task(model, test_data, metrics_list, top_ks):
    from search_recs import metric as metric_module

    METRIC_MAP = {
        "PRECISION": getattr(metric_module, "PrecisionAtK", None),
        "RECALL": getattr(metric_module, "RecallAtK", None),
        "MRR": getattr(metric_module, "ReciprocalRank", None),
        "MAP": getattr(metric_module, "AveragePrecision", None),
        "NDCG": getattr(metric_module, "NdcgAtK", None),
    }

    active_metrics = {}
    for m in (metrics_list or []):
        key = str(m).upper().strip()
        cls = METRIC_MAP.get(key)
        if cls is not None:
            active_metrics[key] = cls()

    if not active_metrics:
        print("[eval] No valid metrics configured. Skipping evaluation.")
        return {}

    top_ks = [int(k) for k in (top_ks or [10])]

    print(f"[eval] Metrics: {list(active_metrics.keys())}")
    print(f"[eval] Top-Ks: {top_ks}")
    print(f"[eval] Calling model.prediction(test_data) with {_safe_len(test_data)} examples...")

    all_y_true, all_y_pred = model.prediction(test_data)

    results = {}
    for k in top_ks:
        results[int(k)] = {}
        for name, metric_obj in active_metrics.items():
            scores = []
            for y_true, y_pred in zip(all_y_true, all_y_pred):
                scores.append(metric_obj.evaluate_metric(y_pred=y_pred, y_true=y_true, topk=int(k)))
            results[int(k)][name] = float(np.mean(scores))
            print(f"[eval] {name}@{k}: {results[int(k)][name]:.8f} (based on {len(scores)} queries)")
    return results

def _predict_scores_recs(model, df: pd.DataFrame, batch_size: int = 4096) -> np.ndarray:

    n_total = len(df)
    scores_list = []
    
    def extract_score_array(prediction_output, chunk_len, chunk_labels=None):
    
        def as_1d(x):
            arr = np.asarray(x)
            if arr.ndim == 1 and len(arr) == chunk_len:
                return arr.astype(float)
            return None

        if isinstance(prediction_output, tuple) and len(prediction_output) == 2:
            a, b = prediction_output
            aa, bb = as_1d(a), as_1d(b)
            
            if chunk_labels is not None:
                if aa is not None and np.allclose(aa, chunk_labels, equal_nan=True):
                    return bb if bb is not None else aa
                if bb is not None and np.allclose(bb, chunk_labels, equal_nan=True):
                    return aa if aa is not None else bb
            
            return aa if aa is not None else bb
        
        return as_1d(prediction_output)

    print(f"[Prediction] Processing {n_total} rows in batches of {batch_size}...")
    
    for start_idx in range(0, n_total, batch_size):
        end_idx = min(start_idx + batch_size, n_total)
        chunk = df.iloc[start_idx:end_idx].copy()
        
        try:
            out_chunk = model.prediction(chunk)
        except Exception as e:
            print(f"[Error] Prediction failure in batch {start_idx}-{end_idx}: {e}")
            raise e
            
        labels_chunk = None
        if "label" in chunk.columns:
            labels_chunk = pd.to_numeric(chunk["label"], errors="coerce").fillna(0.0).to_numpy()

        scores_chunk = extract_score_array(out_chunk, len(chunk), labels_chunk)
        
        if scores_chunk is None:
             raise ValueError(f"[recs-eval] Batch {start_idx}-{end_idx}: Prediction output invalid or incorrect size.")
             
        scores_list.append(scores_chunk)
        
        del chunk, out_chunk
        
    full_scores = np.concatenate(scores_list)
    
    if len(full_scores) != n_total:
         raise ValueError(f"[recs-eval] Concatenation error: Input={n_total}, Output={len(full_scores)}")
         
    return full_scores


def build_recs_eval_df(train_df: pd.DataFrame, test_df: pd.DataFrame, n_neg_samples: int | None = None, seed: int = 42):
    rng = np.random.default_rng(seed)
    
    all_items = pd.unique(pd.concat([train_df["item"], test_df["item"]]))
    
    train_seen = train_df.groupby("user")["item"].agg(set).to_dict()
    test_pos = test_df.groupby("user")["item"].agg(set).to_dict()
    
    users = list(test_pos.keys())
    if n_neg_samples is None:
        print(f"[Evaluation] Generating ALL possible negatives for {len(users)} users (no limit)...")
    else:
        print(f"[Evaluation] Generating {n_neg_samples} fixed negatives for {len(users)} users...")

    neg_users = []
    neg_items = []

    for u in users:
  
        seen_by_u = train_seen.get(u, set()) | test_pos.get(u, set())
        
        possible_negs = np.setdiff1d(all_items, list(seen_by_u), assume_unique=True)
        if len(possible_negs) == 0:
            continue

        if n_neg_samples is None:
            chosen_negs = possible_negs
        else:
            n_to_draw = min(len(possible_negs), int(n_neg_samples))
            if n_to_draw <= 0:
                continue
            chosen_negs = rng.choice(possible_negs, size=n_to_draw, replace=False)

        neg_users.extend([u] * len(chosen_negs))
        neg_items.extend(chosen_negs)
        
    neg_df = pd.DataFrame({
        "user": neg_users,
        "item": neg_items,
        "label": 0.0
    })
    
    for col in test_df.columns:
        if col not in neg_df.columns:
            neg_df[col] = 0

    eval_df = pd.concat([test_df, neg_df], ignore_index=True)
    
    print(f"[Evaluation] Final DataFrame: {eval_df.shape[0]} rows "
          f"({len(test_df)} positives + {len(neg_df)} negatives)")
    
    return eval_df

def evaluate_recs_userwise(model, train_df, test_df, eval_conf: dict, seed: int):

    from search_recs import metric as metric_module

    top_ks = eval_conf.get("top_ks", [10])
    metrics_list = [m.upper().strip() for m in eval_conf.get("metrics", ["NDCG", "RECALL", "PRECISION", "MAP", "MRR"])]
    n_neg = _opt_int(eval_conf.get("n_neg_samples", None))
    
    METRIC_MAP = {
        "PRECISION": getattr(metric_module, "PrecisionAtK", None),
        "RECALL": getattr(metric_module, "RecallAtK", None),
        "MRR": getattr(metric_module, "ReciprocalRank", None),
        "MAP": getattr(metric_module, "AveragePrecision", None),
        "NDCG": getattr(metric_module, "NdcgAtK", None),
    }

    active_metrics_objs = {}
    for m_name in metrics_list:
        cls = METRIC_MAP.get(m_name)
        if cls is not None:
            active_metrics_objs[m_name] = cls()
        else:
            print(f"[warning] Metric '{m_name}' not found in search_recs.metric")

    if not active_metrics_objs:
        raise ValueError("[eval-recs] No valid metrics configured in evaluation.metrics.")

    eval_df = build_recs_eval_df(train_df, test_df, n_neg_samples=n_neg, seed=seed)

    print(f"[eval-recs] Generating scores for {len(eval_df)} pairs (positives + negatives)...")
    b_size = int(eval_conf.get("eval_batch_size", 1024))
    scores = _predict_scores_recs(model, eval_df, batch_size=b_size)

    eval_df = eval_df.copy()
    eval_df["_score"] = scores

    users_used = 0
    sums = {k: {m: 0.0 for m in active_metrics_objs} for k in top_ks}

    print(f"[eval-recs] Calculating user-wise metrics for {eval_df['user'].nunique()} users...")
    
    for _, group in eval_df.groupby("user", sort=False):
 
        y_true = pd.to_numeric(group["label"], errors="coerce").fillna(0.0).tolist()
        y_pred = pd.to_numeric(group["_score"], errors="coerce").fillna(-1e30).tolist()

        if sum(y_true) == 0:
            continue

        users_used += 1

        for k in top_ks:
            for name, metric_obj in active_metrics_objs.items():
                try:
                    score = metric_obj.evaluate_metric(y_pred=y_pred, y_true=y_true, topk=int(k))
                    sums[k][name] += float(score)
                except Exception as e:
                    if users_used == 1:
                        print(f"[metric-error] Failed to calculate {name} at K={k}: {e}")

    if users_used == 0:
        raise ValueError("[eval-recs] No valid user (with positives) found for evaluation.")

    results = {}
    for k in top_ks:
        k_str = str(k)
        results[k_str] = {}
        for name in active_metrics_objs:
            results[k_str][name] = float(sums[k][name] / users_used)

    print(f"[eval-recs] Evaluation successfully completed for {users_used} users.")
    return results

def load_experiment_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_run_dir(exp_config: dict, ds_cfg: dict, md_cfg: dict, dataset_key: str, model_class_name: str) -> Path:
    full_config = {**ds_cfg, **md_cfg, "experiment": exp_config}
    exp_for_hash = dict(exp_config)
    exp_for_hash.pop("execution", None)
    
    full_config_for_hash = {**ds_cfg, **md_cfg, "experiment": exp_for_hash}

    cfg_hash_val = config_hash(full_config_for_hash)
    base_dir = Path(exp_config.get("output_dir", "artifacts"))
    
    run_dir = ensure_run_dir(dataset_key, model_class_name, cfg_hash_val, base_dir=base_dir)
    print(f"[run] Model: {model_class_name} | hash={cfg_hash_val}")
    print(f"[run] dir={run_dir}")
    save_json(run_dir / "full_config.json", full_config)
    return run_dir


def _make_run_id(base_ts: str, run_idx: int) -> str:
    return f"{base_ts}_r{run_idx:02d}"


def _auto_seed() -> int:
    rng = np.random.default_rng()
    return int(rng.integers(1, 2**31 - 1))


def _get_seed(dl_params: dict, fallback_seed: int) -> int:
    s = dl_params.get("seed", None)
    if s is None or str(s).strip() == "":
        return int(fallback_seed)
    try:
        return int(s)
    except Exception:
        return int(fallback_seed)
    
def _opt_int(v):
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"", "none", "null", "inf", "infinite", "infinity", "all", "unlimited", "no_limit", "nolimit"}:
            return None
        try:
            v = int(s)
        except Exception:
            return None
    try:
        iv = int(v)
    except Exception:
        return None
    if iv <= 0:
        return None
    return iv

def run_preprocess(task_type, ds_cfg, dataset_key, splits_path, force_preprocess, no_save, skip_preprocess=False, exp_config=None):

    train_data = val_data = test_data = None
    t0 = stage_start("preprocess")

    if skip_preprocess and not force_preprocess:
        if splits_path and splits_path.exists():
            try:
                print(f"[preprocess] (skip) Using cached splits: {splits_path}")
                data = joblib.load(splits_path)
                train_data = data.get("train")
                val_data = data.get("val")
                test_data = data.get("test")
                
                if train_data is not None and test_data is not None:
                    stage_end("preprocess", t0, tag="skip")
                    return train_data, val_data, test_data
            except Exception as e:
                print(
                    f"[preprocess] Warning: skip requested, but failed to load splits. "
                    f"Generating splits. Error: {e.__class__.__name__}: {e}"
                )
        else:
            print("[preprocess] Warning: skip requested, but splits.pkl does not exist. Generating splits...")

    print(f"[preprocess] Generating splits for {task_type}...")
    dl_params = ds_cfg.get("dataloader", {})
    loader_name = (
        dl_params.get("dataset_name")
        or dl_params.get("dataset")
        or dl_params.get("name")
        or dataset_key
    )

    try:
        loader_entry = get_dataloader_entry(str(loader_name), task_type)
    except Exception:
        loader_entry = get_dataloader_entry(dataset_key, task_type)

    if task_type == "search":
        exp_cfg = exp_config or {}
        if bool(exp_cfg.get("hybrid", False)):
            strategy = exp_cfg.get("hybridStrategie", exp_cfg.get("hybridStrategy", ""))
            norm = _norm_search_hybrid_strategy(str(strategy))
            if norm in _SEARCH_HYBRID_STRATEGIES:
                dl_params = {**dl_params, "mode": "hybrid"}
                print(f"[preprocess] Hybrid search: hybridStrategy='{norm}' → overriding dataloader mode to 'hybrid'")

        from search_recs.search.dataloader.base_dataloader import BuildConfig
        seed = int(dl_params.get("seed", 0))
        h_train = _opt_int(dl_params.get("head_train"))
        h_test = _opt_int(dl_params.get("head_test"))

        build_cfg = BuildConfig(
            test_size=dl_params.get("test_size", 0.3),
            val_size=dl_params.get("val_size", 0.1),
            random_state=seed,
            head_train=h_train,
            head_test=h_test
        )
        try:
            train_data, val_data, test_data = loader_entry(build_cfg, **dl_params)
        except TypeError:
            train_data, val_data, test_data = loader_entry(build_cfg)

    else:
        loader_instance = loader_entry(ds_cfg)

        # Optional hybrid loader selection (e.g., query-as-user / tag-as-user)
        exp_cfg = exp_config or {}
        hybrid_enabled = bool(exp_cfg.get("hybrid", False))
        hybrid_strategy = exp_cfg.get("hybridStrategie", exp_cfg.get("hybridStrategy", None))
        strategy_norm = str(hybrid_strategy).strip().lower() if hybrid_strategy is not None else ""

        if hybrid_enabled and strategy_norm in {"query-as-user", "querry-as-user", "query_as_user", "queryasuser"}:
            if hasattr(loader_instance, "hybrid_load_data"):
                print(f"[preprocess] Hybrid mode enabled: strategy={hybrid_strategy} -> using hybrid_load_data()")
                full_data = loader_instance.hybrid_load_data()
            elif hasattr(loader_instance, "load__data"):
                print(f"[preprocess] Hybrid mode enabled: strategy={hybrid_strategy} -> using load__data()")
                full_data = loader_instance.load__data()
            else:
                raise AttributeError(
                    f"[preprocess] Hybrid mode requested (strategy={hybrid_strategy}), but dataloader "
                    f"{loader_instance.__class__.__name__} does not implement hybrid_load_data() or load__data()."
                )
        else:
            if hybrid_enabled and strategy_norm:
                print(f"[preprocess] Hybrid mode enabled: strategy={hybrid_strategy} -> using standard load_data()")
            full_data = loader_instance.load_data()

        full_data = ensure_recs_schema(full_data, dl_params)
        full_data = apply_min_rating_filter_recs(full_data, dl_params)
        
        stage_end("preprocess", t0)
        return full_data, None, None

    if not no_save and splits_path:
        try:
            print(f"[preprocess] Saving splits to cache: {splits_path}")
            joblib.dump({"train": train_data, "val": val_data, "test": test_data}, splits_path)
        except Exception as e:
            print(f"[preprocess] Warning: Could not save cache: {e}")

    stage_end("preprocess", t0)
    return train_data, val_data, test_data

def run_model_init(task_type, model_class_name, real_model_config, full_config):
    t0 = stage_start("model_init")
    ModelCls = get_model_class(model_class_name, task_type)
    
    dataset_params = full_config.get("dataloader", {})
    
    model_init_config = {**dataset_params, **real_model_config}
    
    if task_type == "search":
        model = ModelCls(model_init_config)
    else:
        try:
            model = ModelCls(model_init_config, full_config.get("features", {}))
        except TypeError as e:
            if "takes" in str(e) and "positional arguments" in str(e):
                print(f"[model] {model_class_name} does not accept features. Initializing only with config (dataset+model).")
                model = ModelCls(model_init_config)
            else:
                raise e

    print(f"[model] Instance created: {model}")
    stage_end("model_init", t0)
    return model


def run_features(task_type, model, train_data, val_data, test_data, skip: set, force: set, model_key: str, run_dir: Path, will_eval: bool,):
    t0 = stage_start("features")
    force_features = "features" in force
    skip_features = ("features" in skip) and not force_features

    if skip_features:
        model2, resumed = resume_if_exists(model_key, model, run_dir)
        if not resumed:
            model2, resumed = _try_resume_from_previous_runs(model_key, model, run_dir)

        if resumed:
            setattr(model2, "_ckpt_resumed", True)
            setattr(model2, "_features_ready", True)
            stage_end("features", t0, tag="cache")
            return model2

        if ("train" not in skip) or ("train" in force):
            print("[features] Warning: skip requested, but no checkpoint for resume and TRAIN is active. Executing preprocess().")
        else:
            if not will_eval:
                stage_end("features", t0, tag="skip")
                return model
            print("[features] Warning: checkpoint not loaded and EVAL active; executing preprocess().")

    if task_type == "search":
        model.preprocess(train_data=train_data, test_data=test_data, val_data=val_data)
    else:
        model.preprocess(train_data=train_data)

    setattr(model, "_features_ready", True)

    stage_end("features", t0)
    return model


def run_train(task_type, model, skip, run_dir, no_save, force, model_key: str, will_eval: bool):
    t0 = stage_start("train")
    force_train = "train" in force
    skip_train = ("train" in skip) and not force_train
    no_save_train = "train" in no_save 

    if skip_train and getattr(model, "_ckpt_resumed", False):
        stage_end("train", t0, tag="cache")
        return model

    if skip_train:
        model2, resumed = resume_if_exists(model_key, model, run_dir)
        if resumed:
            setattr(model2, "_ckpt_resumed", True)
            stage_end("train", t0, tag="cache")
            return model2

        model3, resumed_prev = _try_resume_from_previous_runs(model_key, model, run_dir)
        if resumed_prev:
            setattr(model3, "_ckpt_resumed", True)
            stage_end("train", t0, tag="cache")
            return model3

        if not will_eval:
            print("[train] Warning: checkpoint not loaded. Training will be SKIPPED because EVAL will not be executed.")
            stage_end("train", t0, tag="skip")
            return model

        print(
            "[train] Warning: checkpoint not loaded; "
            "since EVAL is active, it is not safe to skip TRAINING. Training..."
        )

    model.fit()

    save_if_enabled(model_key, model, run_dir, enabled=(not no_save_train))

    stage_end("train", t0)
    return model


def run_index(task_type, model, skip: set, force: set, will_eval: bool):
    t0 = stage_start("index")
    force_index = "index" in force
    skip_index = ("index" in skip) and not force_index

    if task_type != "search":
        stage_end("index", t0, tag="na")
        return model

    if not hasattr(model, "build_index"):
        stage_end("index", t0, tag="na")
        return model

    if skip_index:
        if not will_eval:
            stage_end("index", t0, tag="skip")
            return model

        index_ready = bool(
            getattr(model, "_index_ready", False)
            or getattr(model, "index_ready", False)
            or getattr(model, "index_built", False)
            or (hasattr(model, "index") and getattr(model, "index") is not None)
        )

        if index_ready:
            stage_end("index", t0, tag="skip")
            return model

        print("[index] Warning: skip requested, but index does not seem to exist. Executing build_index()...")

    model.build_index()
    setattr(model, "_index_ready", True)
    stage_end("index", t0)
    return model


def run_eval( task_type, model, test_data, exp_config, run_dir, skip: set, force: set, no_save: set, model_name, dataset_key, full_config, execution_timestamp, train_data=None, emit_long_metrics: bool = False, run_seed: int | None = None,):
    t0 = stage_start("eval")
    force_eval = "eval" in force
    skip_eval = ("eval" in skip) and not force_eval
    no_save_eval = "eval" in no_save 

    if skip_eval:
        p = _find_prev_run_with_file(run_dir, "metrics.json")
        if p is not None:
            try:
                cached = json.loads(p.read_text(encoding="utf-8"))
                print("[eval] (cache) metrics.json loaded from:", p)
                print("\n=== RESULTS (NORMALIZED) ===")
                print(json.dumps(cached, indent=2, default=str))

                if (p.parent != run_dir) and (not no_save_eval):
                    with contextlib.suppress(Exception):
                        save_json(run_dir / "metrics.json", cached)

                stage_end("eval", t0, tag="cache")
                return
            except Exception as e:
                print(
                    f"[eval] Warning: failed to load metrics.json from cache. "
                    f"Continuing without cache. Error: {e.__class__.__name__}: {e}"
                )

        stage_end("eval", t0, tag="skip")
        return

    eval_conf = exp_config.get("evaluation", {})
    print("[eval] Config:", eval_conf)
    
    if task_type == "search":
        metrics = eval_conf.get("metrics", [])
        top_ks = eval_conf.get("top_ks", [10])
        results = evaluate_search_task(model, test_data, metrics, top_ks)

    else:
        dl_params = full_config.get("dataloader", {})
        seed = _get_seed(dl_params, fallback_seed=(run_seed or 0))

        if train_data is None:
            raise ValueError("[eval-recs] train_data is required for user-wise evaluation (negative generation).")

        results = evaluate_recs_userwise(
            model=model,
            train_df=train_data,
            test_df=test_data,
            eval_conf=eval_conf,
            seed=seed,
        )

    results_norm = normalize_results_dict(results)

    print("\n=== RESULTS (NORMALIZED) ===")
    print(json.dumps(results_norm, indent=2, default=str))

    if no_save_eval:
        print("[eval] NO_SAVE active: results will not be persisted to files.")
        stage_end("eval", t0)
        return

    if results_norm:
        base_output = Path("experimental_results")
        subdir = "search" if task_type == "search" else "recs"
        out_dir = base_output / dataset_key / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        exp_name = exp_config.get("experiment_name", "experiment")
        metrics_order = [str(m).upper().strip() for m in (eval_conf.get("metrics") or ["NDCG", "RECALL", "PRECISION", "MAP", "MRR"])]
        top_ks = [int(k) for k in (eval_conf.get("top_ks") or [10])]

        if emit_long_metrics:
            metrics_all = out_dir / f"{exp_name}_metrics_all.csv"
            append_metrics_long_csv(metrics_all, model_name, results_norm, run_id=execution_timestamp)
            print(f"[eval] metrics_all updated: {metrics_all}")

        else:
            wide_csv = out_dir / f"{exp_name}_{execution_timestamp}_{model_name}_table.csv"
            wide_tex = out_dir / f"{exp_name}_{execution_timestamp}_{model_name}_table.tex"
            caption = f"{exp_name} — {task_type.upper()} — {dataset_key} — {model_name}"
            write_metrics_wide_csv(wide_csv, results_norm, metrics_order)
            print(f"[eval] Wide CSV generated: {wide_csv}")
            write_metrics_wide_latex(wide_tex, results_norm, metrics_order, caption)

    save_json(run_dir / "metrics.json", results_norm)

    stage_end("eval", t0)

def split_recs_dataset_temporal(full_data: pd.DataFrame, fold: int = 0, num_folds: int = 5,
                                test_split: float = 0.2, window_size: float = 0.05, 
                                floatseed: int = 42, max_test_pos_per_user: int | None = None):

    if 'time' in full_data.columns and 'timestamp' not in full_data.columns:
        full_data = full_data.rename(columns={'time': 'timestamp'})

    full_data = full_data.sort_values(["user", "timestamp"], ascending=True)

    if window_size > test_split:
        raise ValueError("Window size must be smaller than test sample size.")

    train_split = 1. - test_split

    if train_split <= 0:
      raise ValueError("Train split must be positive.")

    if test_split <= 0:
      raise ValueError("Test split must be positive.")

    seed = np.random.RandomState(floatseed)

    low = train_split - window_size
    high = train_split + window_size
    train_splits = np.linspace(low, high, num_folds)
    
    if fold >= len(train_splits):
        raise IndexError(f"Fold {fold} is out of index for generated thresholds (size {len(train_splits)}).")
    
    current_train_ratio = train_splits[fold]

    g = full_data.groupby("user", sort=False)
    user_sizes = g["user"].transform("size")
    train_cutoffs = (user_sizes * current_train_ratio).astype(int)
    idx = g.cumcount()
    
    train_df = full_data[idx < train_cutoffs].copy()
    test_raw_df = full_data[idx >= train_cutoffs].copy()

    if max_test_pos_per_user is None:
        test_df = test_raw_df.reset_index(drop=True)
    else:
        m = int(max_test_pos_per_user)
        test_df = test_raw_df.groupby("user", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), m), random_state=seed)
        ).reset_index(drop=True)

    print(f"[Fold {fold}] Train Ratio: {current_train_ratio:.4f} | "
          f"Train: {len(train_df)} | Test: {len(test_df)}")
    
    return train_df, test_df


def main(args):
    config_path = Path(args.config)
    exp_config = load_experiment_config(config_path)

    task_type = exp_config.get("task_type", "recs").lower()
    if task_type not in {"search", "recs"}:
        raise ValueError("task_type must be 'search' or 'recs'.")

    base_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[system] Base timestamp: {base_ts}")

    base_path = Path(__file__).resolve().parent
    dataset_key = _normalize_dataset_name(exp_config)
    ds_cfg_path = resolve_dataset_config(exp_config, task_type, base_path)
    
    print(f"[config] experiment_name: {exp_config.get('experiment_name')}")
    print(f"[config] task_type: {task_type}")
    print(f"[config] dataset: {dataset_key} -> {ds_cfg_path.name}")

    ds_cfg = json.loads(ds_cfg_path.read_text(encoding="utf-8"))

    if task_type == "recs":
        eval_conf = exp_config.get("evaluation", {}) or {}
        max_pos = _opt_int(eval_conf.get("n_pos_samples", None))
        dl_params = ds_cfg.setdefault("dataloader", {})

        n_neg_eval = eval_conf.get("n_neg_samples", None)
        if n_neg_eval is not None:
            try:
                dl_params.setdefault("n_neg_samples", int(n_neg_eval))
            except Exception:
                print("[config-recs] Warning: n_neg_samples in evaluation is not a valid integer.")

    # Get Models
    raw_models = exp_config.get("model")
    if not isinstance(raw_models, list):
        raw_models = [raw_models]

    model_names_queue = [_normalize_model_class_name(m) for m in raw_models]
    validate_models_for_task(model_names_queue, task_type)
    print(f"[config] Model queue: {model_names_queue}")

    # Preprocessing Pipeline
    exec_conf = exp_config.get("execution", {})
    skip = set(exec_conf.get("skip", []))
    force = set(exec_conf.get("force", []))
    no_save = set(exec_conf.get("no_save", []))

    stats_conf = (exp_config.get("evaluation", {}) or {}).get("stats", None)
    tests = None
    if isinstance(stats_conf, dict):
        tests = stats_conf.get("tests", None)
    tests = tests if isinstance(tests, list) else []

    wants_stats = (len(model_names_queue) > 1) and (len(tests) > 0)
    tests_norm = {
        str(t).lower().strip().replace("-", "_").replace(" ", "_")
        for t in tests
    }
    has_wilcoxon = "wilcoxon" in tests_norm
    has_ttest = bool({"ttest", "t_test", "paired_ttest"} & tests_norm)

    n_runs_user = exec_conf.get("n_runs", None)
    if n_runs_user is None:
        if wants_stats and has_wilcoxon:
            n_runs = 6
        elif wants_stats and has_ttest:
            n_runs = 2
        else:
            n_runs = 1
    else:
        n_runs = int(n_runs_user)

    window_size_val = float(exec_conf.get("window_size", 0.05))

    if wants_stats and has_wilcoxon and n_runs < 6:
        print(f"[stats] Warning: Wilcoxon requires multiple runs. Adjusting n_runs: {n_runs} -> 6")
        n_runs = 6
    elif wants_stats and has_ttest and n_runs < 2:
        print(f"[stats] Warning: Paired t-test requires at least 2 runs. Adjusting n_runs: {n_runs} -> 2")
        n_runs = 2

    if n_runs < 1:
        n_runs = 1

    print(f"[execution] n_runs = {n_runs} | wants_stats={wants_stats} | tests={tests}")

    base_output_dir = Path(exp_config.get("output_dir", "artifacts"))
    experiment_name = exp_config.get("experiment_name", "default_experiment")
    exp_context_dir = base_output_dir / dataset_key / experiment_name
    exp_context_dir.mkdir(parents=True, exist_ok=True)
    splits_path = exp_context_dir / "splits.pkl"

    full_recs_data = None
    train_data = val_data = test_data = None

    if task_type == "recs":
        train_data, val_data, test_data = run_preprocess(
            task_type,
            ds_cfg,
            dataset_key,
            splits_path,
            force_preprocess=("preprocess" in force),
            no_save=("preprocess" in no_save),
            skip_preprocess=("preprocess" in skip),
            exp_config=exp_config,
        )
        full_recs_data = train_data
        train_data = None
        test_data = None

    print(f"[data] Loaded shapes: Train={_safe_len(train_data)}, Test={_safe_len(test_data)}, FullRecs={_safe_len(full_recs_data)}")

    for run_idx in range(1, n_runs + 1):
        run_id = _make_run_id(base_ts, run_idx)
        run_seed = _auto_seed()
        print(f"\n{'=' * 80}")
        print(f"[RUN] {run_idx}/{n_runs}  run_id={run_id}  run_seed={run_seed}")
        print(f"{'=' * 80}\n")

        if task_type == "search":
            print(f"[main] Generating search splits for run {run_idx} (random_state={run_seed})...")
            ds_cfg.setdefault("dataloader", {})["seed"] = run_seed
            train_data, val_data, test_data = run_preprocess(
                task_type,
                ds_cfg,
                dataset_key,
                splits_path=None,
                force_preprocess=True,
                no_save=True,
                skip_preprocess=False,
                exp_config=exp_config,
            )
            print(f"[main] Search split complete. Train={_safe_len(train_data)}, Val={_safe_len(val_data)}, Test={_safe_len(test_data)}")

        if task_type == "recs":
            current_fold = run_idx - 1
            print(f"[main] Splitting dataset for fold {current_fold} (run {run_idx})...")
            
            num_folds_cfg = int(dl_params.get("num_folds", 3))
            num_folds_val = max(num_folds_cfg, n_runs)
            if num_folds_val != num_folds_cfg:
                print(f"[split] num_folds ({num_folds_cfg}) < n_runs ({n_runs}); bumping num_folds to {num_folds_val} so every run gets a distinct fold.")
            mode_val = dl_params.get("split_mode", "linear")
            if mode_val not in {"linear", "density", "log"}:
                print(f"[split] Warning: invalid split_mode='{mode_val}'. Using 'linear'.")
                mode_val = "linear"
            seed_val = int(dl_params.get("seed", 42))

            train_data, test_data = split_recs_dataset_temporal(
                full_data=full_recs_data,
                fold=current_fold,
                num_folds=num_folds_val,
                floatseed=seed_val,
                max_test_pos_per_user=max_pos,
                window_size=window_size_val 
            )
            print(f"[main] Split complete. Train={len(train_data)}, Test={len(test_data)}")

        for i, model_entry in enumerate(raw_models):
            tf.compat.v1.reset_default_graph()
            current_model_name = model_names_queue[i]
            print(f"\n{'#' * 40}")
            print(f"### MODEL {i+1}/{len(raw_models)}: {current_model_name} | run_id={run_id}")
            print(f"{'#' * 40}\n")

            md_cfg_path, _ = resolve_model_config(model_entry, base_path)
            md_cfg = json.loads(md_cfg_path.read_text(encoding="utf-8"))
            real_model_config = md_cfg.get("model", md_cfg)
            real_model_config = _inject_search_hybrid_task(real_model_config, exp_config, task_type)
            full_config = {**ds_cfg, **md_cfg, "experiment": exp_config}

            base_run_dir = prepare_run_dir(exp_config, ds_cfg, md_cfg, dataset_key, current_model_name)
            run_dir = base_run_dir / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            save_json(run_dir / "run_meta.json", {"run_id": run_id, "run_seed": run_seed})
            
            will_eval = ("eval" in force) or ("eval" not in skip)
            model = run_model_init(task_type, current_model_name, real_model_config, full_config)
            model = run_features( task_type, model, train_data, val_data, test_data, skip=skip, force=force,model_key=current_model_name, run_dir=run_dir, will_eval=will_eval,)
            model = run_train( task_type, model, skip, run_dir, no_save, force, model_key=current_model_name, will_eval=will_eval)
            model = run_index(task_type, model, skip=skip, force=force, will_eval=will_eval)

            emit_long = wants_stats
            run_eval( task_type=task_type, model=model, test_data=test_data, exp_config=exp_config, run_dir=run_dir, skip=skip, force=force, no_save=no_save, model_name=current_model_name, dataset_key=dataset_key, full_config=full_config, execution_timestamp=run_id, train_data=train_data, emit_long_metrics=emit_long, run_seed=run_seed)

    if wants_stats:
        metrics_subdir = "recs" if task_type == "recs" else "search"
        out_dir = Path("experimental_results") / dataset_key / metrics_subdir

        metrics_all_path = out_dir / f"{experiment_name}_metrics_all.csv"
        generate_compact_reports(
            metrics_all_path=metrics_all_path,
            out_dir=out_dir,
            experiment_name=experiment_name,
            dataset_title=dataset_key,  
            metrics_order=(exp_config.get("evaluation", {}) or {}).get("metrics", ["NDCG","MAP","MRR","PRECISION","RECALL"]),
            top_ks=(exp_config.get("evaluation", {}) or {}).get("top_ks", [10]),
        )

    if wants_stats:
        try:
            from search_recs.stats.significance import run_significance_analysis

            print("[stats] Running statistical comparison tests between models...")
            metrics_subdir = "recs" if task_type == "recs" else "search"
            metrics_dir = Path("experimental_results") / dataset_key / metrics_subdir

            model_pairs = [
                (model_names_queue[i], model_names_queue[j])
                for i in range(len(model_names_queue))
                for j in range(i + 1, len(model_names_queue))
            ]

            alpha = float(stats_conf.get("alpha", 0.05))
            symbol = stats_conf.get("symbol", "\u25B2")
            default_min_runs = 6 if has_wilcoxon else (2 if has_ttest else 1)
            min_runs = int(stats_conf.get("min_runs", default_min_runs))

            run_significance_analysis(metrics_dir=metrics_dir, experiment_name=experiment_name, model_pairs=model_pairs, tests=tests, alpha=alpha, symbol=symbol, min_runs=min_runs)
        except ImportError as e:
            print(f"[stats] Statistical tests module not available: {e}")
    else:
        if len(model_names_queue) <= 1:
            print("[stats] Only one model configured; statistical tests will not be run.")
        else:
            print("[stats] No tests configured in evaluation.stats.tests; statistical tests will not be run.")

    total_time = perf_counter() - args._t0
    print(f"\n[Framework] Finished in {total_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parsed = parser.parse_args()
    setattr(parsed, "_t0", perf_counter())
    main(parsed)
