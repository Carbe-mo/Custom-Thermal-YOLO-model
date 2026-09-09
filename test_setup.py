"""
test_setup.py — Verify dataset + results_logger before any real training.

1. Populates data/ from the source dataset.
2. Generates canonical splits.
3. Builds thermal + rgb data YAMLs.
4. Logs two fake runs and prints the CSV + xlsx paths.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.dataset import populate_data, generate_splits, build_yaml, get_paired_paths
from common.results_logger import log_run, CSV_PATH, XLSX_PATH


def main():
    print("=" * 60)
    print("STEP 1: Populate data/ from source dataset")
    print("=" * 60)
    populate_data()

    print()
    print("=" * 60)
    print("STEP 2: Generate canonical splits (seed=42)")
    print("=" * 60)
    splits = generate_splits(force=True)
    for name, stems in splits.items():
        print(f"  {name}: {len(stems)} images  (first 3: {stems[:3]})")

    print()
    print("=" * 60)
    print("STEP 3: Build Ultralytics data YAMLs")
    print("=" * 60)
    thm_yaml = build_yaml("thermal")
    rgb_yaml = build_yaml("rgb")
    print()
    print(f"  Thermal YAML contents:")
    print(f"  {thm_yaml.read_text()}")

    print()
    print("=" * 60)
    print("STEP 4: Paired paths sample (for v4_rgb_distill)")
    print("=" * 60)
    pairs = get_paired_paths("train")
    print(f"  Total paired training samples: {len(pairs)}")
    for thm, rgb, lbl in pairs[:3]:
        print(f"    THM={thm.name}  RGB={rgb.name}  LBL={lbl.name}")

    print()
    print("=" * 60)
    print("STEP 5: Log two fake runs to results_logger")
    print("=" * 60)

    row1 = log_run(
        model_id="v0",
        model_name="v0_baseline (YOLOv8n)",
        folder="models/v0_baseline",
        params_M=3.01,
        gflops=8.1,
        epochs=50,
        mAP50=0.8693,
        mAP50_95=0.6192,
        precision=0.8508,
        recall=0.7864,
        inference_ms=2.7,
        train_time_min=12.5,
        notes="fake test run 1",
    )

    row2 = log_run(
        model_id="v2",
        model_name="v2_cbam_backbone (YOLOv8n+CBAM)",
        folder="models/v2_cbam_backbone",
        params_M=3.25,
        gflops=8.9,
        epochs=50,
        mAP50=0.9012,
        mAP50_95=0.6580,
        precision=0.8720,
        recall=0.8190,
        inference_ms=3.1,
        train_time_min=14.2,
        notes="fake test run 2 — should become best",
    )

    print()
    print("=" * 60)
    print("STEP 6: Verify output files")
    print("=" * 60)
    print(f"\n  CSV path: {CSV_PATH}")
    print(f"  CSV contents:\n")
    print(CSV_PATH.read_text())

    print(f"\n  XLSX path: {XLSX_PATH}")
    print(f"  XLSX exists: {XLSX_PATH.exists()}")
    if XLSX_PATH.exists():
        print(f"  XLSX size:   {XLSX_PATH.stat().st_size} bytes")

    # Clean up fake data so real runs start fresh
    print()
    print("=" * 60)
    print("STEP 7: Clean up fake rows (so real training starts clean)")
    print("=" * 60)
    CSV_PATH.write_text("")  # empty the CSV
    if XLSX_PATH.exists():
        XLSX_PATH.unlink()
    print("  results.csv emptied, results_highlighted.xlsx deleted.")
    print()
    print("All checks passed. Ready to train.")


if __name__ == "__main__":
    main()
