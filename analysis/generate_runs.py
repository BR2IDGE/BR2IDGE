"""Generate the experiment configs of the transfer-conditions study (item 2).

Groups:
  step1  complementary Retrieval-as-User runs (target_mode "all", MS MARCO with top_k=100)
  m1     number of pseudo-users (query_limit) on BEIR, both strategies
  m2     co-occurrence removal (overlap_break) on BEIR Query-as-User
  m3     controlled cold test items (cold_test_frac) on BEIR, both strategies

Each variant gets its own dataset config (config_files/datasets/generated/<group>/) and its own
experiment config (runs/<group>/) with a unique experiment_name, so matrices and results of
different variants never overwrite each other. The experiment config carries an "analysis"
block that analysis/*.py uses to recover group / strategy / parameter / level / seed.

Usage:
  python analysis/generate_runs.py            # all groups
  python analysis/generate_runs.py m1 m3      # only some groups
"""

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Base experiments and dataset configs produced in step 0.
BASES = {
    ("beir", "QaU"): ("beir_lightfm_query_as_user.json", "config_files/datasets/beir_nfcorpus_recs.json"),
    ("beir", "RaU"): ("beir_lightfm_retrieval_as_user.json", "config_files/datasets/retrieval_as_user_beir_nfcorpus.json"),
    ("msmarco", "QaU"): ("msmarco_lightfm_query_as_user.json", "config_files/datasets/msmarco_trec_dl_recs_query.json"),
    ("msmarco", "RaU"): ("msmarco_lightfm_retrieval_as_user.json", "config_files/datasets/retrieval_as_user_msmarco.json"),
}

SEEDS = (1, 2, 3)


def _variants(group: str):
    """Yield (dataset, strategy, changes-to-dataloader, analysis-metadata)."""
    if group == "step1":
        yield "beir", "RaU", {"target_mode": "all"}, {"param": "target_mode", "level": "all"}
        yield "msmarco", "RaU", {"target_mode": "new", "top_k": 100}, {"param": "top_k", "level": 100, "target_mode": "new"}
        yield "msmarco", "RaU", {"target_mode": "all", "top_k": 100}, {"param": "top_k", "level": 100, "target_mode": "all"}
    elif group == "m1":
        for strat in ("QaU", "RaU"):
            for level in (25, 42, 100, 200):
                for seed in SEEDS:
                    yield "beir", strat, {"query_limit": level, "seed": seed}, {"param": "query_limit", "level": level, "seed": seed}
    elif group == "m2":
        for level in (0.25, 0.5, 0.75, 1.0):
            for seed in SEEDS:
                yield "beir", "QaU", {"overlap_break": level, "seed": seed}, {"param": "overlap_break", "level": level, "seed": seed}
    elif group == "m3":
        for strat in ("QaU", "RaU"):
            for level in (0.25, 0.5, 0.75, 1.0):
                for seed in SEEDS:
                    yield "beir", strat, {"cold_test_frac": level, "seed": seed}, {"param": "cold_test_frac", "level": level, "seed": seed}
    else:
        raise ValueError(f"Unknown group '{group}'.")


def _name(group, dataset, strat, meta):
    level = str(meta["level"]).replace(".", "p")
    parts = [group.upper(), dataset.upper(), strat, f"{meta['param']}{level}"]
    if "target_mode" in meta:
        parts.append(meta["target_mode"])
    if "seed" in meta:
        parts.append(f"s{meta['seed']}")
    return "_".join(parts)


def generate(group: str) -> list:
    ds_dir = ROOT / "config_files" / "datasets" / "generated" / group
    run_dir = ROOT / "runs" / group
    ds_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for dataset, strat, changes, meta in _variants(group):
        exp_base, ds_base = BASES[(dataset, strat)]
        name = _name(group, dataset, strat, meta)

        ds_cfg = json.loads((ROOT / ds_base).read_text(encoding="utf-8"))
        ds_cfg["dataloader"].update(changes)
        ds_path = ds_dir / f"{name.lower()}.json"
        ds_path.write_text(json.dumps(ds_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        exp = copy.deepcopy(json.loads((ROOT / exp_base).read_text(encoding="utf-8")))
        exp["experiment_name"] = name
        exp["dataset"] = {"name": exp["dataset"]["name"], "config_path": str(ds_path.relative_to(ROOT))}
        exp["analysis"] = {"group": group, "dataset": dataset, "strategy": strat, **meta}
        exp_path = run_dir / f"{name.lower()}.json"
        exp_path.write_text(json.dumps(exp, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(exp_path)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="*", default=["step1", "m1", "m2", "m3"])
    args = ap.parse_args()
    for group in args.groups:
        files = generate(group)
        print(f"[generate] {group}: {len(files)} experiment(s) in runs/{group}/")


if __name__ == "__main__":
    main()
