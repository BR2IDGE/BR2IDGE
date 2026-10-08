"""Deliverable 3: tag coverage of the OUTPUT ranking (top-5/10/20 results).

For each model (Tag Query / Centroid Vector), trains/indexes ONCE (the
underlying document index is built from train_df, which does not depend on
history_size), then calls prediction() separately for each history_size's
test_df to measure what fraction of the top-k ranked candidates have a
genome tag.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent))

from movielens_matched_protocol_common import load_exp1_frames, build_has_tag_map, topk_candidate_ids
from search_recs.search.model.bi_encoder import BiEncoderModel
from search_recs.search.model.bm25_sparse import BM25Model
from search_recs.metric import NdcgAtK, RecallAtK, PrecisionAtK

OUT_DIR = REPO_ROOT.parent / "experimental_results" / "movielens" / "search"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "BiEncoderSearchModel": (BiEncoderModel, "config_files/models/bi_encoder.json"),
    "BM25Model": (BM25Model, "config_files/models/bm25_search.json"),
}

REPORTED_NDCG10 = {
    "BiEncoderSearchModel": {5: 0.4437, 10: 0.4422, 20: 0.4322},
    "BM25Model": {5: 0.4964, 10: 0.5317, 20: 0.5590},
}


def load_model_config(path: str) -> dict:
    full = json.loads((REPO_ROOT.parent / path).read_text(encoding="utf-8"))
    return copy.deepcopy(full["model"])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default=None,
                         help="Run only this model (default: both)")
    parser.add_argument("--suffix", default="", help="Output filename suffix")
    args = parser.parse_args()

    models_to_run = {args.model: MODEL_CONFIGS[args.model]} if args.model else MODEL_CONFIGS

    has_tag = build_has_tag_map()
    metrics = {"NDCG": NdcgAtK(), "RECALL": RecallAtK(), "PRECISION": PrecisionAtK()}

    print("[audit-output] Loading Exp1 frames for all history sizes (train_df is shared) ...")
    train_df = None
    test_dfs = {}
    for h in (5, 10, 20):
        tdf, test_df = load_exp1_frames(history_size=h, gt_size=20, history_size_max=20)
        if train_df is None:
            train_df = tdf
        test_dfs[h] = test_df
    print(f"[audit-output] train_df rows (shared across h): {len(train_df)}")

    output_rows = []
    sanity_rows = []

    for model_name, (ModelCls, cfg_path) in models_to_run.items():
        print(f"\n[audit-output] ===== {model_name}: train/index ONCE =====")
        model_cfg = load_model_config(cfg_path)
        model_cfg.setdefault("parameters", {})["task"] = "hybrid"

        model = ModelCls(model_cfg)
        # test_data is only used at preprocess-time as a has_candidate_mode trigger /
        # harmless placeholder for indexing (document_id=-1 rows contribute nothing real);
        # the real document index is built entirely from train_df. Any h's test_df works here.
        model.preprocess(train_data=train_df, test_data=test_dfs[5], val_data=None)
        model.fit()

        for h in (5, 10, 20):
            test_df = test_dfs[h]
            print(f"[audit-output] {model_name}: predicting for h={h} ({len(test_df)} users) ...")
            all_y_true, all_y_pred = model.prediction(test_df)

            ndcg10_scores = [
                metrics["NDCG"].evaluate_metric(y_pred=yp, y_true=yt, topk=10)
                for yt, yp in zip(all_y_true, all_y_pred)
            ]
            recomputed_ndcg10 = float(np.mean(ndcg10_scores))
            reported = REPORTED_NDCG10[model_name][h]
            sanity_rows.append({
                "history_size": h, "model": model_name,
                "recomputed_ndcg10": recomputed_ndcg10, "reported_ndcg10": reported,
                "abs_diff": abs(recomputed_ndcg10 - reported),
            })
            print(f"[audit-output] {model_name} h={h}: recomputed NDCG@10={recomputed_ndcg10:.4f} "
                  f"(reported={reported:.4f}, diff={abs(recomputed_ndcg10-reported):.4f})")

            for k in (5, 10, 20):
                fracs = []
                for i, row in enumerate(test_df.itertuples(index=False)):
                    cand_ids = row.candidate_ids_list
                    y_pred = all_y_pred[i]
                    topk_ids = topk_candidate_ids(cand_ids, y_pred, k)
                    frac = float(np.mean([has_tag.get(int(cid), False) for cid in topk_ids])) if topk_ids else np.nan
                    fracs.append(frac)
                mean_pct = float(np.nanmean(fracs) * 100)
                output_rows.append({
                    "history_size": h, "model": model_name, "k": k,
                    "mean_pct_tagged_topk": mean_pct, "n_users": len(test_df),
                })
                print(f"[audit-output]   top-{k}: mean %% tagged = {mean_pct:.2f}")

    out_df = pd.DataFrame(output_rows)
    out_path = OUT_DIR / f"TagCoverage_Output_by_H_Model_K{args.suffix}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n[audit-output] Saved: {out_path}")
    print(out_df.to_string(index=False))

    sanity_df = pd.DataFrame(sanity_rows)
    sanity_path = OUT_DIR / f"TagCoverage_Output_SanityCheck_NDCG10{args.suffix}.csv"
    sanity_df.to_csv(sanity_path, index=False)
    print(f"\n[audit-output] Saved sanity check: {sanity_path}")
    print(sanity_df.to_string(index=False))


if __name__ == "__main__":
    main()
