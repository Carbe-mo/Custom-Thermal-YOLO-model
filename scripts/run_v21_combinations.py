"""
scripts/run_v21_combinations.py
================================
Runs combinations of v21 sequentially:
  1. v26_v21_p4_dilated (Dilation d=2 ONLY at P4, standard ECA at P2/P3/P5)
  2. v27_v21_cosine (v21 Hybrid with Cosine Annealing LR Schedule)
  3. v28_v21_dilated_dual_eca (v21 Hybrid with Dual ECA in dilated P4/P5)
  4. v29_v21_p3p4_dilated (Dilation at P3 and P4, standard at P2/P5)

Evaluates on held-out test split (108 images) after each iteration.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VARIANTS = [
    {
        "id": "v26",
        "name": "v26_v21_p4_dilated",
        "module": "models.v26_v21_p4_dilated.train",
        "desc": "Dilation d=2 only at P4, clean feed to SPPF",
    },
    {
        "id": "v27",
        "name": "v27_v21_cosine",
        "module": "models.v27_v21_cosine.train",
        "desc": "v21 Hybrid arch + Cosine LR Schedule (75 epochs)",
    },
    {
        "id": "v28",
        "name": "v28_v21_dilated_dual_eca",
        "module": "models.v28_v21_dilated_dual_eca.train",
        "desc": "v21 Hybrid + Dual ECA at P4/P5 dilated stages",
    },
    {
        "id": "v29",
        "name": "v29_v21_p3p4_dilated",
        "module": "models.v29_v21_p3p4_dilated.train",
        "desc": "Dilation at P3 and P4, standard at P2 and P5",
    },
]

RESULTS_CSV = PROJECT_ROOT / "results" / "results.csv"
CURRENT_BEST = 0.8741  # v21_eval official test mAP50


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
    print("  SEQUENTIAL v21 COMBINATIONS TRAINING RUN")
    print(f"  Current best (official test): {current_best*100:.2f}%  [v21_dilated_hybrid]")
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
    print("  ALL v21 COMBINATIONS COMPLETE")
    print(f"  Final best on official test split: {current_best*100:.2f}%")
    print("=" * 70)

    df = read_leaderboard()
    print("\nFULL LEADERBOARD (official test eval rows):")
    print(df[["model_id", "mAP50", "precision", "recall", "mAP50_95", "params_M", "gflops"]]
          .head(25).to_string(index=False))


if __name__ == "__main__":
    main()
