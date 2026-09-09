"""
scripts/run_v21_to_v25_sequential.py
=====================================
Runs v21 -> v22 -> v23 -> v24 -> v25 sequentially.
After each run, prints the updated leaderboard vs. the current best.
Auto-tracks whether each new model beats the record.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VARIANTS = [
    {
        "id": "v21",
        "name": "v21_dilated_hybrid",
        "module": "models.v21_dilated_hybrid.train",
        "desc": "Hybrid: ECA@P2/P3 + Dilated-ECA@P4/P5",
    },
    {
        "id": "v22",
        "name": "v22_dilated_cosine",
        "module": "models.v22_dilated_cosine.train",
        "desc": "All-Dilated ECA + 100ep Cosine LR",
    },
    {
        "id": "v23",
        "name": "v23_thermal_aug",
        "module": "models.v23_thermal_aug.train",
        "desc": "v5b arch + Thermal-Specific Augmentation",
    },
    {
        "id": "v24",
        "name": "v24_copy_paste_aug",
        "module": "models.v24_copy_paste_aug.train",
        "desc": "v5b arch + Copy-Paste (class imbalance fix)",
    },
    {
        "id": "v25",
        "name": "v25_dilated_graduated",
        "module": "models.v25_dilated_graduated.train",
        "desc": "Graduated Dilation d=1/1/2/3 per stage",
    },
]

RESULTS_CSV = PROJECT_ROOT / "results" / "results.csv"
CURRENT_BEST = 0.8739  # v5b_eval / v13_eval official test mAP50


def read_leaderboard():
    df = pd.read_csv(RESULTS_CSV)
    eval_rows = df[df["model_id"].str.endswith("_eval")].drop_duplicates(
        subset=["model_id"], keep="last"
    ).sort_values("mAP50", ascending=False)
    return eval_rows


def print_review(variant_id: str, variant_name: str, current_best: float) -> float:
    df = read_leaderboard()
    row = df[df["model_id"] == f"{variant_id}_eval"]
    if row.empty:
        print(f"\n[REVIEW] No eval row found for {variant_id}!")
        return current_best

    r = row.iloc[0]
    test_map50 = float(r["mAP50"])
    is_new_best = test_map50 > current_best

    print("\n" + "=" * 70)
    print(f"  REVIEW: {variant_name}")
    print("=" * 70)
    print(f"  Test mAP50   : {test_map50 * 100:.2f}%")
    print(f"  Precision    : {float(r['precision']) * 100:.2f}%")
    print(f"  Recall       : {float(r['recall']) * 100:.2f}%")
    print(f"  mAP50-95     : {float(r['mAP50_95']) * 100:.2f}%")
    print(f"  Params (M)   : {r['params_M']}")
    print(f"  GFLOPs       : {r['gflops']}")
    print(f"  vs best ({current_best * 100:.2f}%): {'[NEW BEST] +' if is_new_best else '[below]  '}"
          f"{abs(test_map50 - current_best) * 100:.2f}pp {'above' if is_new_best else 'below'}")
    print()

    print("  --- Current Top-10 Leaderboard ---")
    top10 = df.head(10)
    for rank, (_, row_) in enumerate(top10.iterrows(), 1):
        marker = f"  {rank:2d}"
        print(f"  {marker}. {row_['model_id']:<20s} {float(row_['mAP50'])*100:.2f}%  "
              f"P={float(row_['precision'])*100:.1f}%  R={float(row_['recall'])*100:.1f}%")
    print("=" * 70 + "\n")

    return max(current_best, test_map50)


def main():
    current_best = CURRENT_BEST
    print(f"\n{'='*70}")
    print("  SEQUENTIAL v21->v25 TRAINING RUN")
    print(f"  Current best (official test): {current_best*100:.2f}%  [v5b/v13]")
    print(f"{'='*70}\n")

    for v in VARIANTS:
        print(f"\n{'='*70}")
        print(f"  LAUNCHING: {v['name']}  ({v['desc']})")
        print(f"{'='*70}")

        result = subprocess.run(
            [sys.executable, "-m", v["module"]],
            cwd=str(PROJECT_ROOT),
        )

        if result.returncode != 0:
            print(f"\n[ERROR] {v['name']} failed with exit code {result.returncode}. Skipping review.")
            continue

        current_best = print_review(v["id"], v["name"], current_best)

    print("\n" + "=" * 70)
    print("  ALL 5 VARIANTS COMPLETE")
    print(f"  Final best on official test split: {current_best*100:.2f}%")
    print("=" * 70)

    df = read_leaderboard()
    print("\nFULL LEADERBOARD (official test eval rows):")
    print(df[["model_id", "mAP50", "precision", "recall", "mAP50_95", "params_M", "gflops"]]
          .head(25).to_string(index=False))


if __name__ == "__main__":
    main()
