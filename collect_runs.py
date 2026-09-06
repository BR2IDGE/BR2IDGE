import argparse
import csv
import json
import re
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from framework import generate_compact_reports

FOLD_RE = re.compile(r"_r(\d+)$")


def fold_key(run_id: str) -> str:
    m = FOLD_RE.search(run_id)
    return f"fold{int(m.group(1)) - 1:02d}" if m else run_id


def find_hash_dirs(artifacts: Path, dataset: str, experiment: str) -> list[Path]:
    found = []
    root = artifacts / dataset
    if not root.exists():
        return found
    for cfg_path in sorted(root.glob("*/*/full_config.json")):
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (cfg.get("experiment") or {}).get("experiment_name") == experiment:
            found.append(cfg_path.parent)
    return found


def collect_rows(hash_dirs: list[Path], run_key: str = "fold") -> list[dict]:
    rows = []
    for hash_dir in hash_dirs:
        model = hash_dir.parent.name
        for metrics_path in sorted(hash_dir.glob("runs/*/metrics.json")):
            run_id = metrics_path.parent.name
            if run_key == "fold":
                run_id = fold_key(run_id)
            try:
                results = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[aviso] ignorando {metrics_path}: {exc}")
                continue
            for k, metrics in (results or {}).items():
                for metric, value in (metrics or {}).items():
                    try:
                        v = float(value)
                    except (TypeError, ValueError):
                        continue
                    rows.append({
                        "RunId": run_id,
                        "Model": model,
                        "K": int(k),
                        "Metric": str(metric).upper(),
                        "Value": f"{v:.6f}",
                    })
    return rows


def report_coverage(rows: list[dict]) -> None:
    per_model = {}
    for r in rows:
        per_model.setdefault(r["Model"], set()).add(r["RunId"])
    width = max((len(m) for m in per_model), default=0)
    for model in sorted(per_model):
        print(f"[collect]   {model:<{width}}  {len(per_model[model])} execucao(oes)")
    shared = set.intersection(*per_model.values()) if per_model else set()
    print(f"[collect]   execucoes em comum a todos os modelos: {len(shared)}")
    if len({len(v) for v in per_model.values()}) > 1:
        print("[aviso] modelos com numero diferente de execucoes; "
              "os testes pareados usam so as execucoes em comum.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--dataset", default="beir_nfcorpus")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--task", default="recs", choices=["recs", "search"])
    ap.add_argument("--top-ks", default="5,10,20,50")
    ap.add_argument("--metrics", default="NDCG,PRECISION,RECALL")
    ap.add_argument("--run-key", default="fold", choices=["fold", "raw"])
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--tests", default="ttest_rel")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-runs", type=int, default=6)
    args = ap.parse_args()

    hash_dirs = find_hash_dirs(Path(args.artifacts), args.dataset, args.experiment)
    if not hash_dirs:
        print(f"[erro] nenhum artefato de '{args.experiment}' em {args.artifacts}/{args.dataset}")
        return 1

    rows = collect_rows(hash_dirs, run_key=args.run_key)
    if not rows:
        print("[erro] nenhum metrics.json encontrado nas execucoes")
        return 1

    models = sorted({r["Model"] for r in rows})
    n_runs = len({r["RunId"] for r in rows})
    print(f"[collect] {len(rows)} linhas | {n_runs} execucao(oes) | {len(models)} modelo(s)")
    report_coverage(rows)

    out_dir = Path("experimental_results") / args.dataset / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_all = out_dir / f"{args.experiment}_metrics_all.csv"

    with open(metrics_all, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["RunId", "Model", "K", "Metric", "Value"])
        w.writeheader()
        w.writerows(rows)
    print(f"[collect] {metrics_all}")

    generate_compact_reports(
        metrics_all_path=metrics_all,
        out_dir=out_dir,
        experiment_name=args.experiment,
        dataset_title=args.dataset,
        metrics_order=[m.strip().upper() for m in args.metrics.split(",") if m.strip()],
        top_ks=[int(k) for k in args.top_ks.split(",") if k.strip()],
    )

    if args.stats:
        if len(models) < 2:
            print("[stats] menos de dois modelos coletados; sem comparacao pareada.")
        else:
            from search_recs.stats.significance import run_significance_analysis
            run_significance_analysis(
                metrics_dir=out_dir,
                experiment_name=args.experiment,
                model_pairs=list(combinations(models, 2)),
                tests=[t.strip() for t in args.tests.split(",") if t.strip()],
                alpha=args.alpha,
                min_runs=args.min_runs,
            )

    print(f"\n[ok] tabelas geradas em {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
