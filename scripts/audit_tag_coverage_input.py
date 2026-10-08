"""Deliverable 2: tag coverage of the INPUT history (h5/h10/h20).

For each history_size, measures what fraction of each user's history items
(the items fed into Tag Query / Centroid Vector) have a genome tag.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from movielens_matched_protocol_common import load_exp1_frames, build_has_tag_map

OUT_DIR = REPO_ROOT.parent / "experimental_results" / "movielens" / "search"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    has_tag = build_has_tag_map()
    rows = []
    per_user_frames = {}

    for h in (5, 10, 20):
        print(f"[audit-input] Loading Exp1 frames for history_size={h} ...")
        _train_df, test_df = load_exp1_frames(history_size=h, gt_size=20, history_size_max=20)

        fracs = test_df["query_items"].apply(
            lambda items: float(np.mean([has_tag.get(int(i), False) for i in items])) if items else np.nan
        )
        per_user_frames[h] = pd.DataFrame({"userId": test_df["userId"], "tag_coverage_frac": fracs})

        n_users = len(test_df)
        rows.append({
            "history_size": h,
            "n_users": n_users,
            "mean_pct_tagged": float(fracs.mean() * 100),
            "median_pct_tagged": float(fracs.median() * 100),
            "p25_pct_tagged": float(fracs.quantile(0.25) * 100),
            "p75_pct_tagged": float(fracs.quantile(0.75) * 100),
            "pct_users_fully_tagged": float((fracs == 1.0).mean() * 100),
            "pct_users_zero_tagged": float((fracs == 0.0).mean() * 100),
        })
        print(f"[audit-input] h={h}: n_users={n_users} mean_pct_tagged={rows[-1]['mean_pct_tagged']:.2f}")

    summary = pd.DataFrame(rows)
    out_path = OUT_DIR / "TagCoverage_Input_by_H.csv"
    summary.to_csv(out_path, index=False)
    print(f"[audit-input] Saved: {out_path}")
    print(summary.to_string(index=False))

    for h, df in per_user_frames.items():
        per_user_path = OUT_DIR / f"TagCoverage_Input_per_user_H{h}.csv"
        df.to_csv(per_user_path, index=False)
        print(f"[audit-input] Saved: {per_user_path}")


if __name__ == "__main__":
    main()
