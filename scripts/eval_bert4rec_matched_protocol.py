"""Deliverable 1: BERT4Rec under the exact matched Exp1 protocol.

For each history_size in (5, 10, 20): trains BERT4Rec using ONLY the same
pre-GT history items used by Tag Query / Centroid Vector (same 64,907 users),
then evaluates against the SAME ground_truth_ids/candidate_ids (20 GT + up to
200 negatives), producing a true row-by-row comparison with the adaptations.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent))

from movielens_matched_protocol_common import load_exp1_frames
from search_recs.recs.model.bert4rec import Bert4REC
from search_recs.metric import NdcgAtK, RecallAtK, PrecisionAtK

OUT_DIR = REPO_ROOT.parent / "experimental_results" / "movielens" / "search"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_USERS = 2000
TOP_KS = (5, 10, 20, 50)


def build_bert4rec_train_df(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in test_df.itertuples(index=False):
        for t, item in enumerate(row.query_items):
            rows.append({"user": str(row.userId), "item": str(item), "label": 1.0, "time": float(t)})
    return pd.DataFrame(rows)


def check_oov(model: Bert4REC, test_df: pd.DataFrame) -> float:
    i_map = model.dataset.field2token_id["item"]
    vocab_size = len(i_map)
    all_cands = [c for cand in test_df["candidate_ids_list"] for c in cand]
    oov_rate = float(np.mean([i_map.get(str(c), 0) == 0 for c in all_cands]))
    print(f"[bert4rec-matched] vocab_size(items)={vocab_size} | candidate OOV rate={oov_rate:.4f}")
    return oov_rate


def evaluate_in_batches(model: Bert4REC, test_df: pd.DataFrame):
    rows_meta = list(test_df.itertuples(index=False))
    all_y_true, all_y_pred = [], []

    n_batches = (len(rows_meta) + BATCH_USERS - 1) // BATCH_USERS
    t0 = time.time()
    for b, start in enumerate(range(0, len(rows_meta), BATCH_USERS)):
        batch_rows = rows_meta[start:start + BATCH_USERS]
        flat_rows = []
        counts = []
        for row in batch_rows:
            cand = row.candidate_ids_list
            counts.append(len(cand))
            for c in cand:
                flat_rows.append({"user": str(row.userId), "item": str(c)})
        flat_df = pd.DataFrame(flat_rows)
        _y_true_unused, y_pred_flat = model.prediction(flat_df)
        y_pred_flat = np.asarray(y_pred_flat)

        idx = 0
        for row, cnt in zip(batch_rows, counts):
            y_pred_u = y_pred_flat[idx:idx + cnt]
            idx += cnt
            gt_set = set(row.gt_items)
            y_true_u = np.array([1.0 if cid in gt_set else 0.0 for cid in row.candidate_ids_list])
            all_y_true.append(y_true_u)
            all_y_pred.append(y_pred_u)

        if (b + 1) % 5 == 0 or (b + 1) == n_batches:
            elapsed = time.time() - t0
            print(f"[bert4rec-matched] eval batch {b+1}/{n_batches} "
                  f"({start+len(batch_rows)}/{len(rows_meta)} users) | {elapsed:.1f}s elapsed")

    return all_y_true, all_y_pred


def compute_metrics(all_y_true, all_y_pred, top_ks=TOP_KS):
    metric_objs = {"NDCG": NdcgAtK(), "RECALL": RecallAtK(), "PRECISION": PrecisionAtK()}
    results = {}
    for k in top_ks:
        results[k] = {}
        for name, obj in metric_objs.items():
            scores = [obj.evaluate_metric(y_pred=yp, y_true=yt, topk=k) for yt, yp in zip(all_y_true, all_y_pred)]
            results[k][name] = float(np.mean(scores))
            print(f"[bert4rec-matched] {name}@{k}: {results[k][name]:.6f}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_sizes", default="5,10,20", help="Comma-separated history sizes to run")
    args = parser.parse_args()
    history_sizes = [int(x) for x in args.history_sizes.split(",")]

    summary_rows = []
    oov_rows = []

    for h in history_sizes:
        print(f"\n[bert4rec-matched] ===== history_size={h} =====")
        _train_df_unused, test_df = load_exp1_frames(history_size=h, gt_size=20, history_size_max=20)

        train_df_bert = build_bert4rec_train_df(test_df)
        print(f"[bert4rec-matched] BERT4Rec train rows: {len(train_df_bert)} "
              f"({len(test_df)} users x up to {h} history items)")

        model_cfg = json.loads((REPO_ROOT.parent / "config_files/models/recbole_bert4rec.json").read_text())["model"]
        # Restricted-history sequences (5/10/20 items vs the ~150 RecBole normally sees) make it
        # likely, with the default train_batch_size=16, that an entire batch draws zero masked
        # positions under BERT4Rec's Cloze task (P(no mask in one 5-item sequence) = 0.8^5 ~= 33%),
        # which makes RecBole's loss divide-by-zero into NaN (recbole/model/.../bert4rec.py:212).
        # A larger batch makes "every sequence in the batch has zero masks" astronomically unlikely,
        # without touching mask_ratio or any shared wrapper code.
        model_cfg.setdefault("parameters", {})["train_batch_size"] = 256
        model = Bert4REC(model_cfg)
        model.preprocess(train_data=train_df_bert)

        oov_rate = check_oov(model, test_df)
        oov_rows.append({"history_size": h, "oov_rate_candidates": oov_rate})

        t_fit0 = time.time()
        model.fit()
        print(f"[bert4rec-matched] fit() took {time.time()-t_fit0:.1f}s")

        t_eval0 = time.time()
        all_y_true, all_y_pred = evaluate_in_batches(model, test_df)
        print(f"[bert4rec-matched] eval took {time.time()-t_eval0:.1f}s")

        results = compute_metrics(all_y_true, all_y_pred)

        wide_rows = [{"K": k, "NDCG": results[k]["NDCG"], "RECALL": results[k]["RECALL"], "PRECISION": results[k]["PRECISION"]} for k in TOP_KS]
        wide_df = pd.DataFrame(wide_rows)
        wide_path = OUT_DIR / f"MovieLens_Exp1_Context_H{h}_Bert4REC_MatchedProtocol_table.csv"
        wide_df.to_csv(wide_path, index=False)
        print(f"[bert4rec-matched] Saved: {wide_path}")

        for k in TOP_KS:
            for name in ("NDCG", "RECALL", "PRECISION"):
                summary_rows.append({"history_size": h, "k": k, "metric": name, "value": results[k][name]})

    suffix = "_h" + "-".join(str(h) for h in history_sizes)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / f"Bert4REC_MatchedProtocol_summary{suffix}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[bert4rec-matched] Saved summary: {summary_path}")

    oov_df = pd.DataFrame(oov_rows)
    oov_path = OUT_DIR / f"Bert4REC_MatchedProtocol_OOV{suffix}.csv"
    oov_df.to_csv(oov_path, index=False)
    print(f"[bert4rec-matched] Saved OOV report: {oov_path}")
    print(oov_df.to_string(index=False))


if __name__ == "__main__":
    main()
