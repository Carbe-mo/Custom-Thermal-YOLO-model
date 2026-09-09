"""
scripts/run_v33_and_baseline_100ep.py
======================================
Trains and evaluates under identical conditions:
  1. v33_v30_concat_eca: v30 Hybrid Dilated Backbone + Concat_ECA Neck Gating (100ep Cosine)
  2. v0b_baseline_100ep: Stock YOLOv8n baseline (100ep Cosine, close_mosaic=20)

Both models are evaluated on the exact same held-out test split (data/splits/test.txt, 108 images).
Directly isolates the net architectural gain of our models over the stock baseline.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VARIANTS = [
    {
        "id": "v33",
        "name": "v33_v30_concat_eca",
        "module": "models.v33_v30_concat_eca.train",
        "desc": "Hybrid Dilated + Concat_ECA Cross-Scale Gating (100ep Cosine)",
    },
    {
        "id": "v0b",
        "name": "v0b_baseline_100ep",
        "module": "models.v0b_baseline_100ep.train",
        "desc": "Stock YOLOv8n Baseline under identical 100ep Cosine conditions",
    },
]

RESULTS_CSV = PROJECT_ROOT / "results" / "results.csv"
CURRENT_BEST = 0.8789  # v30_eval official test mAP50


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
    diff = abs(test_map50 - current_best) * 100
    tag = "[NEW BEST] +" if is_new_best else "[below]  -"
    direction = "above" if is_new_best else "below"
    print(f"  vs best ({current_best * 100:.2f}%): {tag}{diff:.2f}pp {direction}")
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
    print("  SEQUENTIAL TRAINING: v33 (Concat_ECA) & v0b (100ep Baseline)")
    print(f"  Current best (official test): {current_best*100:.2f}%  [v30_v27_cosine_100ep]")
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
    print("  BOTH RUNS COMPLETE -- HEAD-TO-HEAD COMPARISON")
    print("=" * 70)

    df = read_leaderboard()
    target_ids = ["v33_eval", "v30_eval", "v0b_eval", "v0_eval"]
    comparison = df[df["model_id"].isin(target_ids)]
    print(comparison[["model_id", "mAP50", "precision", "recall", "mAP50_95", "params_M"]].to_string(index=False))


if __name__ == "__main__":
    main()
