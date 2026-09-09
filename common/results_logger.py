"""
common/results_logger.py — Append-only results logger
=====================================================

Rules (from CLAUDE.md):
- results/results.csv is append-only, written ONLY by this module.
- Every train.py calls log_run(...) with real metrics from the trainer.
- Never hand-edit results.csv.

This module also regenerates results/results_highlighted.xlsx on every
log_run() call, highlighting the current best mAP50 row.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CSV_PATH = RESULTS_DIR / "results.csv"
XLSX_PATH = RESULTS_DIR / "results_highlighted.xlsx"

COLUMNS = [
    "run_id",
    "model_id",
    "model_name",
    "folder",
    "params_M",
    "gflops",
    "epochs",
    "mAP50",
    "mAP50_95",
    "precision",
    "recall",
    "inference_ms",
    "train_time_min",
    "date",
    "notes",
    "is_best",
]


def _read_all_rows() -> list[dict[str, str]]:
    """Read every row from results.csv, return list of dicts."""
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        return []
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_all_rows(rows: list[dict[str, str]]) -> None:
    """Rewrite the entire CSV (used only to flip is_best flags)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _next_run_id(rows: list[dict[str, str]]) -> int:
    """Return the next sequential run_id."""
    if not rows:
        return 1
    return max(int(r.get("run_id", 0)) for r in rows) + 1


def _regenerate_xlsx(rows: list[dict[str, str]]) -> None:
    """
    Build results_highlighted.xlsx from the current CSV data.
    The row with is_best=True gets a green highlight on mAP50.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    except ImportError:
        print("[results_logger] openpyxl not installed — skipping .xlsx generation")
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "Results"

    # Header styling
    header_fill = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True, size=11)
    thin_border = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin"),
    )

    # Write headers
    for col_idx, col_name in enumerate(COLUMNS, start=1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")
        cell.border = thin_border

    # Highlight for best row
    best_fill = PatternFill(start_color="92D050", end_color="92D050", fill_type="solid")
    best_font = Font(bold=True, size=11)

    # Write data rows
    for row_idx, row_data in enumerate(rows, start=2):
        is_best = str(row_data.get("is_best", "")).lower() == "true"
        for col_idx, col_name in enumerate(COLUMNS, start=1):
            value = row_data.get(col_name, "")
            # Try to convert numeric values
            try:
                if col_name in ("run_id", "epochs"):
                    value = int(value)
                elif col_name in ("params_M", "gflops", "mAP50", "mAP50_95",
                                  "precision", "recall", "inference_ms", "train_time_min"):
                    value = float(value)
            except (ValueError, TypeError):
                pass

            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center")

            if is_best:
                cell.font = best_font
                if col_name == "mAP50":
                    cell.fill = best_fill

    # Auto-width columns
    for col_idx, col_name in enumerate(COLUMNS, start=1):
        max_len = len(col_name)
        for row_idx in range(2, len(rows) + 2):
            val = str(ws.cell(row=row_idx, column=col_idx).value or "")
            max_len = max(max_len, len(val))
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = max_len + 3

    wb.save(XLSX_PATH)
    print(f"[results_logger] Wrote {XLSX_PATH}")


def log_run(
    model_id: str,
    model_name: str,
    folder: str,
    params_M: float,
    gflops: float,
    epochs: int,
    mAP50: float,
    mAP50_95: float,
    precision: float,
    recall: float,
    inference_ms: float = 0.0,
    train_time_min: float = 0.0,
    notes: str = "",
) -> dict[str, Any]:
    """
    Append one row to results/results.csv.

    Recomputes is_best by comparing the new mAP50 against the current max.
    If the new run is the best, flips the previous best's flag off.
    Regenerates results_highlighted.xlsx afterward.

    Returns the row dict that was written.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Read existing rows
    rows = _read_all_rows()
    run_id = _next_run_id(rows)

    # Determine if this is the new best
    current_best_mAP50 = 0.0
    for r in rows:
        try:
            val = float(r.get("mAP50", 0))
            if val > current_best_mAP50:
                current_best_mAP50 = val
        except (ValueError, TypeError):
            pass

    is_best = mAP50 > current_best_mAP50

    # If new run is best, demote the previous best
    if is_best:
        for r in rows:
            if str(r.get("is_best", "")).lower() == "true":
                r["is_best"] = "False"

    # Build the new row
    new_row = {
        "run_id": str(run_id),
        "model_id": model_id,
        "model_name": model_name,
        "folder": folder,
        "params_M": f"{params_M:.2f}",
        "gflops": f"{gflops:.1f}",
        "epochs": str(epochs),
        "mAP50": f"{mAP50:.4f}",
        "mAP50_95": f"{mAP50_95:.4f}",
        "precision": f"{precision:.4f}",
        "recall": f"{recall:.4f}",
        "inference_ms": f"{inference_ms:.2f}",
        "train_time_min": f"{train_time_min:.1f}",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "notes": notes,
        "is_best": str(is_best),
    }

    rows.append(new_row)

    # Write the full CSV (needed to update is_best flags)
    _write_all_rows(rows)
    print(f"[results_logger] Logged run {run_id}: {model_name} "
          f"mAP50={mAP50:.4f} {'** NEW BEST **' if is_best else ''}")

    # Regenerate XLSX
    _regenerate_xlsx(rows)

    return new_row


# ---------------------------------------------------------------------------
# CLI: python -m common.results_logger
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("results_logger module loaded. Use log_run() to append results.")
    print(f"CSV path:  {CSV_PATH}")
    print(f"XLSX path: {XLSX_PATH}")
