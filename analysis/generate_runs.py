"""Generate the experiment configs of the transfer-conditions study (item 2).

Groups:
  step1     complementary Retrieval-as-User runs (target_mode "all", MS MARCO with top_k=100)
  m1        number of pseudo-users (query_limit) on BEIR, both strategies
  m2        co-occurrence removal (overlap_break) on BEIR Query-as-User
  m3        controlled cold test items (cold_test_frac) on BEIR and on the hybrid dataset
  fullrank  base runs evaluated against ALL items of the matrix (no sampled negatives), the
            protocol analysis/native_reference.py needs for a meaningful Transfer Ratio
  hybrid    base runs of the hybrid dataset (Query-as-User, Retrieval-as-User new) + its
            Retrieval-as-User "all" run
  ctx       M5 on the hybrid dataset: real users' interactions on the same catalogue added to the
            training matrix (context_users_frac), i.e. co-occurrence added instead of removed
  feat      criterion 4: the same conditions with recommenders that can score items never seen in
            training -- LightFMTextModel (identity + item text) and LightFMContentModel (item text only) --
            on the base runs of the 3 datasets, BEIR cold_test_frac 0.5/1.0 and BEIR overlap_break 1.0
            (block-diagonal); analysis groups feat_base/feat_m3/feat_m2
  rob       robustness of hypothesis 2: the base runs of the 3 datasets with other collaborative recommenders
            -- ItemKNN and LightGCN with unseen items scored 0 (cold_start_score "popularity") or -inf
            ("neg_inf") -- to check that the conclusions are not specific to LightFM and to its -inf for
            unseen items; analysis group rob_base (per_user_analysis.py: robustness.csv)

Storage: generated experiments save no model checkpoints (execution.no_save = ["train"]) and write
eval_scores.parquet only for the groups whose analyses read it (EVAL_SCORES_GROUPS).

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
    ("hybrid", "QaU"): ("hybrid_lightfm_query_as_user.json", "config_files/datasets/hybrid_query_as_user.json"),
    ("hybrid", "RaU"): ("hybrid_lightfm_retrieval_as_user.json", "config_files/datasets/retrieval_as_user_hybrid.json"),
}

ALL_GROUPS = ["step1", "m1", "m2", "m3", "fullrank", "hybrid", "ctx", "feat", "rob"]
# criterion 4 controls: identity + item text (hybrid) and item text only (content-only)
TEXT_MODELS = {"TEXT": "LightFMTextModel", "CONTENT": "LightFMContentModel"}
# robustness: other collaborative recommenders, configs in config_files/models/h2/ -> (class, config, label)
ROBUST_MODELS = {
    "KNN": ("ItemKNNModel", "config_files/models/h2/itemknn.json", "ItemKNN"),
    "GCNPOP": ("LightGCNModel", "config_files/models/h2/lightgcn_popularity.json", "LightGCN (popularity)"),
    "GCNINF": ("LightGCNModel", "config_files/models/h2/lightgcn_neg_inf.json", "LightGCN (neg_inf)"),
}
SEEDS = (1, 2, 3)
# candidate scores are only read by native_reference.py (TR) and by the tie-neutral criterion-4 table
EVAL_SCORES_GROUPS = {"step1", "fullrank", "hybrid", "feat", "m2", "m3", "rob"}


def _variants(group: str):
    """Yield (dataset, strategy, changes-to-dataloader, analysis-metadata).

    Optional metadata keys: "name" (fixed experiment name), "analysis_group" (group recorded for the
    analysis when it differs from the generation group, e.g. the hybrid base runs are "base"),
    "model" (recommender replacing the base experiment's LightFMModel), "model_config" (its config file,
    when not the one found by class name in config_files/models/) and "model_label" (name recorded for
    the analysis, to tell apart two configs of the same class)."""
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
        for dataset, levels in (("beir", (0.25, 0.5, 0.75, 1.0)), ("hybrid", (0.25, 0.5, 0.75))):
            for strat in ("QaU", "RaU"):
                for level in levels:
                    for seed in SEEDS:
                        yield dataset, strat, {"cold_test_frac": level, "seed": seed}, {"param": "cold_test_frac", "level": level, "seed": seed}
    elif group == "fullrank":
        yield "beir", "QaU", {}, {"param": "candidates", "level": "all"}
        yield "beir", "RaU", {"target_mode": "new"}, {"param": "candidates", "level": "all", "target_mode": "new"}
        yield "beir", "RaU", {"target_mode": "all"}, {"param": "candidates", "level": "all", "target_mode": "all"}
        yield "msmarco", "QaU", {}, {"param": "candidates", "level": "all"}
        yield "msmarco", "RaU", {"target_mode": "new", "top_k": 100}, {"param": "candidates", "level": "all", "target_mode": "new"}
        yield "msmarco", "RaU", {"target_mode": "all", "top_k": 100}, {"param": "candidates", "level": "all", "target_mode": "all"}
        yield "hybrid", "QaU", {}, {"param": "candidates", "level": "all"}
        yield "hybrid", "RaU", {"target_mode": "new"}, {"param": "candidates", "level": "all", "target_mode": "new"}
        yield "hybrid", "RaU", {"target_mode": "all"}, {"param": "candidates", "level": "all", "target_mode": "all"}
    elif group == "hybrid":
        yield "hybrid", "QaU", {}, {"param": "base", "level": "-", "name": "HYBRID_LightFM_QueryAsUser",
                                    "analysis_group": "base"}
        yield "hybrid", "RaU", {"target_mode": "new"}, {"param": "base", "level": "-", "name": "HYBRID_LightFM_RetrievalAsUser",
                                                        "analysis_group": "base"}
        yield "hybrid", "RaU", {"target_mode": "all"}, {"param": "target_mode", "level": "all", "name": "STEP1_HYBRID_RaU_target_modeall",
                                                        "analysis_group": "step1"}
    elif group == "ctx":
        for strat in ("QaU", "RaU"):
            for level in (0.1, 0.25, 0.5, 1.0):
                for seed in SEEDS:
                    yield "hybrid", strat, {"context_users_frac": level, "seed": seed}, {"param": "context_users_frac", "level": level, "seed": seed}
    elif group == "feat":
        for tag, model in TEXT_MODELS.items():
            base = {"model": model, "param": "base", "level": "-", "analysis_group": "feat_base"}
            yield "beir", "QaU", {}, {**base, "name": f"FEAT_{tag}_BEIR_QaU_base"}
            yield "beir", "RaU", {"target_mode": "new"}, {**base, "name": f"FEAT_{tag}_BEIR_RaU_base_new"}
            yield "msmarco", "QaU", {}, {**base, "name": f"FEAT_{tag}_MSMARCO_QaU_base"}
            yield "msmarco", "RaU", {"target_mode": "new", "top_k": 100}, {**base, "name": f"FEAT_{tag}_MSMARCO_RaU_base_top100_new"}
            yield "hybrid", "QaU", {}, {**base, "name": f"FEAT_{tag}_HYBRID_QaU_base"}
            yield "hybrid", "RaU", {"target_mode": "new"}, {**base, "name": f"FEAT_{tag}_HYBRID_RaU_base_new"}
            for strat in ("QaU", "RaU"):
                for level in (0.5, 1.0):
                    for seed in SEEDS:
                        yield "beir", strat, {"cold_test_frac": level, "seed": seed}, {
                            "model": model, "param": "cold_test_frac", "level": level, "seed": seed,
                            "analysis_group": "feat_m3", "name": f"FEAT_{tag}_BEIR_{strat}_cold_test_frac{str(level).replace('.', 'p')}_s{seed}"}
            for seed in SEEDS:
                yield "beir", "QaU", {"overlap_break": 1.0, "seed": seed}, {
                    "model": model, "param": "overlap_break", "level": 1.0, "seed": seed,
                    "analysis_group": "feat_m2", "name": f"FEAT_{tag}_BEIR_QaU_overlap_break1p0_s{seed}"}
    elif group == "rob":
        for tag, (model, model_config, label) in ROBUST_MODELS.items():
            base = {"model": model, "model_config": model_config, "model_label": label, "param": "base", "level": "-",
                    "analysis_group": "rob_base"}
            yield "beir", "QaU", {}, {**base, "name": f"ROB_{tag}_BEIR_QaU_base"}
            yield "beir", "RaU", {"target_mode": "new"}, {**base, "name": f"ROB_{tag}_BEIR_RaU_base_new"}
            yield "msmarco", "QaU", {}, {**base, "name": f"ROB_{tag}_MSMARCO_QaU_base"}
            yield "msmarco", "RaU", {"target_mode": "new", "top_k": 100}, {**base, "name": f"ROB_{tag}_MSMARCO_RaU_base_top100_new"}
            yield "hybrid", "QaU", {}, {**base, "name": f"ROB_{tag}_HYBRID_QaU_base"}
            yield "hybrid", "RaU", {"target_mode": "new"}, {**base, "name": f"ROB_{tag}_HYBRID_RaU_base_new"}
    else:
        raise ValueError(f"Unknown group '{group}'. Groups: {ALL_GROUPS}")


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
        meta = dict(meta)
        name = meta.pop("name", None) or _name(group, dataset, strat, meta)
        analysis_group = meta.pop("analysis_group", group)
        model = meta.pop("model", None)
        model_config = meta.pop("model_config", None)
        model_label = meta.pop("model_label", None) or model
        exp_base, ds_base = BASES[(dataset, strat)]

        ds_cfg = json.loads((ROOT / ds_base).read_text(encoding="utf-8"))
        ds_cfg["dataloader"].update(changes)
        ds_path = ds_dir / f"{name.lower()}.json"
        ds_path.write_text(json.dumps(ds_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        exp = copy.deepcopy(json.loads((ROOT / exp_base).read_text(encoding="utf-8")))
        exp["experiment_name"] = name
        exp["dataset"] = {"name": exp["dataset"]["name"], "config_path": str(ds_path.relative_to(ROOT))}
        exp["analysis"] = {"group": analysis_group, "dataset": dataset, "strategy": strat, **meta}
        if model:
            exp["model"] = [{"name": model, **({"config_path": model_config} if model_config else {})}]
            exp["analysis"]["model"] = model_label
        if group == "fullrank":
            # every item of the adapted matrix is a candidate; every held-out positive is kept
            exp["evaluation"].update({"n_neg_samples": "all", "n_pos_samples": "all"})
        # storage: no model checkpoints (every run retrains; nothing resumes from them) and candidate
        # scores only where an analysis reads them. Neither changes any metric.
        exp["execution"]["no_save"] = sorted(set(exp["execution"].get("no_save", [])) | {"train"})
        exp["evaluation"]["save_eval_scores"] = group in EVAL_SCORES_GROUPS
        exp_path = run_dir / f"{name.lower()}.json"
        exp_path.write_text(json.dumps(exp, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(exp_path)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="*", default=ALL_GROUPS)
    args = ap.parse_args()
    for group in args.groups:
        files = generate(group)
        print(f"[generate] {group}: {len(files)} experiment(s) in runs/{group}/")


if __name__ == "__main__":
    main()
