"""
common/dataset.py — Dataset utilities for thermal-yolo
======================================================

Two responsibilities:
1. generate_splits()  — scans data/thermal/images/, pairs with RGB,
   writes data/splits/{train,val,test}.txt once with a fixed seed.
2. build_yaml()       — writes a temporary Ultralytics-compatible
   data.yaml pointing at the right image/label dirs for a given mode.

The split files store one thermal image **stem** per line (e.g. "FLIR7852").
Every train.py reads the same splits; none ever regenerates them.

Modes
-----
- "thermal"  (default) — Ultralytics trains on thermal images only.
- "rgb"                — Ultralytics trains on RGB images only (teacher for v4).
- "paired"             — Returns both paths; used only by v4's distillation
                         DataLoader wrapper, not by stock Ultralytics.
"""

from __future__ import annotations

import os
import random
import shutil
import yaml
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Paths (relative to project root thermal-yolo/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_ROOT / "splits"

# Source dataset (the already-prepared 4cls data)
SOURCE_DATASET = Path(r"D:\Yolo\Data\dataset_4cls")

CLASS_NAMES = {0: "person", 1: "trolley", 2: "bag", 3: "unattended"}
NC = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# 1. Populate data/ from the source dataset
# ---------------------------------------------------------------------------
def populate_data(force: bool = False) -> None:
    """
    Symlink (or copy) images and labels from the original dataset_4cls
    into data/thermal/ and data/rgb/ with a flat layout:
        data/thermal/images/  ← all thermal .jpg
        data/thermal/labels/  ← all thermal .txt
        data/rgb/images/      ← all rgb .jpg
        data/rgb/labels/      ← all rgb .txt

    Flat layout lets us point Ultralytics at subsets via split .txt files
    without maintaining separate train/val/test folders.
    """
    for modality in ("thermal", "rgb"):
        img_dst = DATA_ROOT / modality / "images"
        lbl_dst = DATA_ROOT / modality / "labels"
        if img_dst.exists() and any(img_dst.iterdir()) and not force:
            continue  # already populated
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for split in ("train", "val", "test"):
            src_img = SOURCE_DATASET / "images" / modality / split
            src_lbl = SOURCE_DATASET / "labels" / modality / split
            if not src_img.exists():
                continue
            for f in src_img.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    dst = img_dst / f.name
                    if not dst.exists():
                        shutil.copy2(f, dst)
            if src_lbl.exists():
                for f in src_lbl.iterdir():
                    if f.suffix == ".txt":
                        dst = lbl_dst / f.name
                        if not dst.exists():
                            shutil.copy2(f, dst)

    print(f"[dataset] data/thermal/images: "
          f"{len(list((DATA_ROOT / 'thermal' / 'images').glob('*.jpg')))} files")
    print(f"[dataset] data/rgb/images:     "
          f"{len(list((DATA_ROOT / 'rgb' / 'images').glob('*.jpg')))} files")


# ---------------------------------------------------------------------------
# 2. Generate canonical splits
# ---------------------------------------------------------------------------
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
# TEST_RATIO = 0.10  (remainder)


def generate_splits(force: bool = False) -> dict[str, list[str]]:
    """
    Scan data/thermal/images/, extract stems, shuffle with fixed seed,
    split 70/20/10, and write data/splits/{train,val,test}.txt.

    Returns dict of {"train": [...], "val": [...], "test": [...]}.
    """
    train_file = SPLITS_DIR / "train.txt"
    val_file = SPLITS_DIR / "val.txt"
    test_file = SPLITS_DIR / "test.txt"

    # If all three exist and force is False, just read them back
    if all(f.exists() and f.stat().st_size > 0 for f in (train_file, val_file, test_file)) and not force:
        splits = {}
        for name, path in [("train", train_file), ("val", val_file), ("test", test_file)]:
            splits[name] = [l.strip() for l in path.read_text().splitlines() if l.strip()]
        print(f"[dataset] Splits already exist — "
              f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        return splits

    # Discover all thermal image stems
    img_dir = DATA_ROOT / "thermal" / "images"
    if not img_dir.exists() or not any(img_dir.iterdir()):
        raise FileNotFoundError(
            f"No images in {img_dir}. Run populate_data() first."
        )

    stems = sorted(f.stem for f in img_dir.glob("*.jpg"))
    random.seed(SEED)
    random.shuffle(stems)

    n = len(stems)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": stems[:n_train],
        "val": stems[n_train : n_train + n_val],
        "test": stems[n_train + n_val :],
    }

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, stem_list in splits.items():
        (SPLITS_DIR / f"{name}.txt").write_text("\n".join(stem_list) + "\n")

    print(f"[dataset] Splits generated (seed={SEED}): "
          f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    return splits


# ---------------------------------------------------------------------------
# 3. Build Ultralytics-compatible data YAML
# ---------------------------------------------------------------------------
def build_yaml(
    mode: Literal["thermal", "rgb"] = "thermal",
    output_path: str | Path | None = None,
) -> Path:
    """
    Write a data.yaml that Ultralytics can consume.

    The YAML uses absolute paths and points train/val/test at text files
    listing image paths (one per line), which Ultralytics natively supports.

    Args:
        mode: "thermal" or "rgb"
        output_path: where to write the YAML; defaults to
                     data/<mode>_data.yaml

    Returns:
        Path to the written YAML.
    """
    img_dir = DATA_ROOT / mode / "images"
    splits = generate_splits()  # reads existing, never regenerates

    if output_path is None:
        output_path = DATA_ROOT / f"{mode}_data.yaml"
    output_path = Path(output_path)

    # Write split list files with full image paths
    list_dir = DATA_ROOT / mode
    for split_name, stem_list in splits.items():
        lines = []
        for stem in stem_list:
            if mode == "rgb":
                # RGB pair: thermal stem is even, RGB is even+1
                num = int(stem.replace("FLIR", ""))
                rgb_name = f"FLIR{num + 1}.jpg"
                img_path = img_dir / rgb_name
                if img_path.exists():
                    lines.append(str(img_path))
            else:
                img_path = img_dir / f"{stem}.jpg"
                if img_path.exists():
                    lines.append(str(img_path))
        (list_dir / f"{split_name}.txt").write_text("\n".join(lines) + "\n")

    data_cfg = {
        "path": str(DATA_ROOT / mode),
        "train": str(list_dir / "train.txt"),
        "val": str(list_dir / "val.txt"),
        "test": str(list_dir / "test.txt"),
        "nc": NC,
        "names": CLASS_NAMES,
    }

    output_path.write_text(yaml.dump(data_cfg, default_flow_style=False, sort_keys=False))
    print(f"[dataset] Wrote {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 4. Paired iterator (for v4_rgb_distill only)
# ---------------------------------------------------------------------------
def get_paired_paths(split: str = "train") -> list[tuple[Path, Path, Path]]:
    """
    Returns list of (thermal_img, rgb_img, label_txt) tuples for a split.
    Used by v4_rgb_distill's custom DataLoader, not by stock Ultralytics.
    """
    splits = generate_splits()
    stems = splits[split]
    triples = []
    for stem in stems:
        thm_img = DATA_ROOT / "thermal" / "images" / f"{stem}.jpg"
        num = int(stem.replace("FLIR", ""))
        rgb_img = DATA_ROOT / "rgb" / "images" / f"FLIR{num + 1}.jpg"
        lbl = DATA_ROOT / "thermal" / "labels" / f"{stem}.txt"
        if thm_img.exists() and rgb_img.exists():
            triples.append((thm_img, rgb_img, lbl))
    return triples


# ---------------------------------------------------------------------------
# CLI: python -m common.dataset
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Populating data/ from source dataset_4cls ...")
    populate_data()
    print()
    print("Generating canonical splits ...")
    splits = generate_splits(force=True)
    print()
    print("Building thermal data.yaml ...")
    build_yaml("thermal")
    print()
    print("Building rgb data.yaml ...")
    build_yaml("rgb")
    print()
    print("Paired paths (train, first 3):")
    for thm, rgb, lbl in get_paired_paths("train")[:3]:
        print(f"  THM: {thm.name}  RGB: {rgb.name}  LBL: {lbl.name}")
    print()
    print("Done.")
