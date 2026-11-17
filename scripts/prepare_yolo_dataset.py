"""
Convert HAM10000 lesion masks + diagnosis metadata into a binary YOLO dataset.

Classes:
  0 = benign     (nv, bkl, df, vasc)
  1 = malignant  (mel, bcc, akiec)

Split: lesion-level stratified 70% train / 15% val / 15% test, seed=42.
All images of the same lesion_id stay in one split (no leakage).

Usage (from project root):
  python scripts/prepare_yolo_dataset.py
  python scripts/prepare_yolo_dataset.py --balance-hard-pairs
  python scripts/prepare_yolo_dataset.py --balance-revolume
  python scripts/prepare_yolo_dataset.py --balance-hard-pairs --mel-multiplier 6.0 --malignant-cap-ratio 1.2
  python scripts/prepare_yolo_dataset.py --balance-train --malignant-ratio 1.2
  python scripts/prepare_yolo_dataset.py --balance-train --malignant-ratio 1.2 --preprocess-derm
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMAGES_DIR = RAW_DIR / "images"
MASKS_DIR = RAW_DIR / "masks"
META_PATH = RAW_DIR / "metadata" / "HAM10000_metadata.csv"
PROCESSED = PROJECT_ROOT / "data" / "processed"
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
REPORT_PATH = PROCESSED / "prepare_report.json"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MALIGNANT = {"mel", "bcc", "akiec"}
BENIGN = {"nv", "bkl", "df", "vasc"}
DX_TO_CLASS = {dx: 1 for dx in MALIGNANT} | {dx: 0 for dx in BENIGN}
CLASS_NAMES = {0: "benign", 1: "malignant"}


def find_mask_for_image(image_stem: str) -> Path | None:
    """Match common HAM mask naming conventions."""
    candidates = [
        MASKS_DIR / f"{image_stem}_segmentation.png",
        MASKS_DIR / f"{image_stem}.png",
        MASKS_DIR / f"{image_stem}_mask.png",
    ]
    for path in candidates:
        if path.is_file():
            return path
    # Fallback: any file starting with stem
    matches = list(MASKS_DIR.glob(f"{image_stem}*"))
    return matches[0] if matches else None


def mask_to_yolo_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Return normalized YOLO bbox (xc, yc, w, h) from a binary mask."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary = (mask > 0).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    h, w = binary.shape[:2]
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    if bw <= 1 or bh <= 1:
        return None
    xc = (x1 + x2) / 2.0 / w
    yc = (y1 + y2) / 2.0 / h
    nw = bw / w
    nh = bh / h
    # Clamp to [0, 1]
    xc = float(np.clip(xc, 0.0, 1.0))
    yc = float(np.clip(yc, 0.0, 1.0))
    nw = float(np.clip(nw, 0.0, 1.0))
    nh = float(np.clip(nh, 0.0, 1.0))
    return xc, yc, nw, nh


def reset_processed_dirs() -> None:
    if PROCESSED.exists():
        shutil.rmtree(PROCESSED)
    for split in ("train", "val", "test"):
        (PROCESSED / "images" / split).mkdir(parents=True, exist_ok=True)
        (PROCESSED / "labels" / split).mkdir(parents=True, exist_ok=True)


def gray_world_normalize(bgr: np.ndarray) -> np.ndarray:
    """Simple Gray-World color constancy on a BGR uint8 image."""
    img = bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    means = np.maximum(means, 1e-6)
    gray = float(means.mean())
    scales = gray / means
    out = img * scales.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def light_hair_removal(bgr: np.ndarray) -> np.ndarray:
    """
    Mild black-hat hair removal + inpaint.
    Falls back to input if little/no hair-like structure is detected.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    if int(mask.sum()) < 500:
        return bgr
    return cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)


def preprocess_derm_bgr(bgr: np.ndarray) -> np.ndarray:
    """Apply Gray-World normalization then light hair removal."""
    normalized = gray_world_normalize(bgr)
    return light_hair_removal(normalized)


def write_image(dst: Path, src: Path, preprocess_derm: bool) -> None:
    """Copy or preprocess+write an image to dst."""
    if not preprocess_derm:
        shutil.copy2(src, dst)
        return
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        shutil.copy2(src, dst)
        return
    out = preprocess_derm_bgr(img)
    cv2.imwrite(str(dst), out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def save_preprocess_samples(df: pd.DataFrame, n: int = 6) -> list[str]:
    """Save a few before/after pairs for visual QA."""
    sample_dir = PROCESSED / "preprocess_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    rows = df.sample(n=min(n, len(df)), random_state=SEED)
    for row in rows.itertuples(index=False):
        src = Path(row.image_path)
        raw = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if raw is None:
            continue
        proc = preprocess_derm_bgr(raw)
        before = sample_dir / f"{src.stem}_before.jpg"
        after = sample_dir / f"{src.stem}_after.jpg"
        cv2.imwrite(str(before), raw, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(str(after), proc, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved.append(src.stem)
    return saved


def write_data_yaml() -> None:
    # Absolute path helps Ultralytics / YOLO repos resolve reliably on Windows
    data = {
        "path": str((PROJECT_ROOT / "data" / "processed").resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 2,
        "names": {0: "benign", 1: "malignant"},
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def count_split_classes(split_name: str) -> dict[str, int]:
    label_dir = PROCESSED / "labels" / split_name
    counts = Counter()
    for path in label_dir.glob("*.txt"):
        line = path.read_text(encoding="utf-8").strip().splitlines()
        if not line:
            continue
        cls = int(line[0].split()[0])
        counts[CLASS_NAMES.get(cls, str(cls))] += 1
    return {
        CLASS_NAMES[0]: int(counts.get(CLASS_NAMES[0], 0)),
        CLASS_NAMES[1]: int(counts.get(CLASS_NAMES[1], 0)),
        "total": int(sum(counts.values())),
    }


def _clean_balance_copies(split: str = "train") -> int:
    """Remove previously generated *_bal* image/label copies from a split."""
    img_dir = PROCESSED / "images" / split
    lbl_dir = PROCESSED / "labels" / split
    removed = 0
    for lbl_path in list(lbl_dir.glob("*.txt")):
        if "_bal" not in lbl_path.stem:
            continue
        lbl_path.unlink(missing_ok=True)
        for ext in (".jpg", ".jpeg", ".png"):
            img_path = img_dir / f"{lbl_path.stem}{ext}"
            if img_path.is_file():
                img_path.unlink()
        removed += 1
    return removed


def _find_train_image(stem: str) -> Path | None:
    img_dir = PROCESSED / "images" / "train"
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = img_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _augment_bgr(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Lightweight OpenCV augment so oversampled copies are not identical pixels."""
    out = img.copy()
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    angle = rng.uniform(-15.0, 15.0)
    h, w = out.shape[:2]
    matrix = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.0)
    out = cv2.warpAffine(
        out,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-4.0, 4.0)) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.75, 1.25), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.75, 1.25), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def _load_train_dx_groups(dx_by_image: dict[str, str]) -> dict[str, list[tuple[Path, Path]]]:
    """Group unique (non-_bal) train samples by dx for malignant subtypes."""
    lbl_dir = PROCESSED / "labels" / "train"
    groups: dict[str, list[tuple[Path, Path]]] = {dx: [] for dx in sorted(MALIGNANT)}
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        if "_bal" in lbl_path.stem:
            continue
        cls = int(lbl_path.read_text(encoding="utf-8").strip().split()[0])
        if cls != 1:
            continue
        img_path = _find_train_image(lbl_path.stem)
        if img_path is None:
            continue
        dx = dx_by_image.get(lbl_path.stem)
        if dx not in groups:
            continue
        groups[dx].append((img_path, lbl_path))
    return groups


def _count_train_dx(dx_by_image: dict[str, str]) -> dict[str, int]:
    lbl_dir = PROCESSED / "labels" / "train"
    counts: Counter = Counter()
    for lbl_path in lbl_dir.glob("*.txt"):
        base = lbl_path.stem.split("_bal")[0]
        dx = dx_by_image.get(base, "unknown")
        counts[dx] += 1
    return dict(counts)


def balance_train_malignant(target_ratio: float, seed: int = SEED) -> dict:
    """
    Legacy: oversample malignant via identical file copies until ratio vs benign.
    Prefer balance_hard_pairs for mel-focused augmented oversampling.
    """
    _clean_balance_copies("train")
    img_dir = PROCESSED / "images" / "train"
    lbl_dir = PROCESSED / "labels" / "train"

    before = count_split_classes("train")
    benign_imgs = []
    malignant_imgs = []
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        if "_bal" in lbl_path.stem:
            continue
        cls = int(lbl_path.read_text(encoding="utf-8").strip().split()[0])
        img_path = _find_train_image(lbl_path.stem)
        if img_path is None:
            continue
        if cls == 1:
            malignant_imgs.append((img_path, lbl_path))
        else:
            benign_imgs.append((img_path, lbl_path))

    n_benign = len(benign_imgs)
    n_mal = len(malignant_imgs)
    if n_benign == 0 or n_mal == 0:
        raise SystemExit("Cannot balance train: missing benign or malignant samples.")

    target_mal = int(math.ceil(n_benign * target_ratio))
    need = max(0, target_mal - n_mal)
    added = 0
    if need > 0:
        for i in range(need):
            src_img, src_lbl = malignant_imgs[i % n_mal]
            suffix = f"_bal{i}"
            dst_img = img_dir / f"{src_img.stem}{suffix}{src_img.suffix}"
            dst_lbl = lbl_dir / f"{src_lbl.stem}{suffix}.txt"
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)
            added += 1

    after = count_split_classes("train")
    return {
        "enabled": True,
        "method": "identical_copy",
        "target_ratio_malignant_to_benign": target_ratio,
        "before": before,
        "after": after,
        "malignant_added": added,
        "unique_malignant_sources": n_mal,
        "unique_benign_sources": n_benign,
    }


def balance_hard_pairs(
    dx_by_image: dict[str, str],
    mel_multiplier: float = 6.0,
    malignant_cap_ratio: float = 1.2,
    other_multiplier: float = 1.5,
    seed: int = SEED,
) -> dict:
    """
    Mel-first OpenCV-augmented oversampling restoring malignant volume.

    - Clean old *_bal* copies
    - Raise mel toward mel_multiplier × unique mel (augmented copies)
    - Mild bcc/akiec boost (other_multiplier)
    - Fill any remaining room under malignant_cap_ratio × benign with
      round-robin bcc/akiec augmented copies (keeps total mal volume ≈ v1)
    """
    rng = random.Random(seed)
    removed = _clean_balance_copies("train")
    img_dir = PROCESSED / "images" / "train"
    lbl_dir = PROCESSED / "labels" / "train"

    before = count_split_classes("train")
    dx_before = _count_train_dx(dx_by_image)
    n_benign = before[CLASS_NAMES[0]]
    n_mal = before[CLASS_NAMES[1]]
    if n_benign == 0 or n_mal == 0:
        raise SystemExit("Cannot hard-pair balance: missing benign or malignant samples.")

    cap_mal = int(math.floor(n_benign * malignant_cap_ratio))
    groups = _load_train_dx_groups(dx_by_image)
    unique_by_dx = {dx: len(items) for dx, items in groups.items()}

    added_by_dx: Counter = Counter()
    bal_idx = 0

    def room_left() -> int:
        cur = count_split_classes("train")[CLASS_NAMES[1]]
        return max(0, cap_mal - cur)

    def add_one(dx: str, src_img: Path, src_lbl: Path) -> bool:
        nonlocal bal_idx
        if room_left() <= 0:
            return False
        raw = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
        if raw is None:
            return False
        aug = _augment_bgr(raw, rng)
        suffix = f"_bal{bal_idx}"
        bal_idx += 1
        dst_img = img_dir / f"{src_img.stem}{suffix}{src_img.suffix}"
        dst_lbl = lbl_dir / f"{src_lbl.stem}{suffix}.txt"
        cv2.imwrite(str(dst_img), aug, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        shutil.copy2(src_lbl, dst_lbl)
        added_by_dx[dx] += 1
        return True

    def add_augmented(dx: str, target_total: int) -> None:
        items = groups.get(dx) or []
        if not items:
            return
        current = unique_by_dx.get(dx, 0) + added_by_dx[dx]
        need = max(0, target_total - current)
        need = min(need, room_left())
        for i in range(need):
            src_img, src_lbl = items[i % len(items)]
            if not add_one(dx, src_img, src_lbl):
                break

    # 1) Mel first toward multiplier × unique
    mel_unique = unique_by_dx.get("mel", 0)
    mel_target = int(math.ceil(mel_unique * mel_multiplier))
    add_augmented("mel", mel_target)

    # 2) Mild bcc/akiec toward other_multiplier
    for dx in ("bcc", "akiec"):
        u = unique_by_dx.get(dx, 0)
        target = int(math.ceil(u * other_multiplier))
        add_augmented(dx, target)

    # 3) Fill remaining headroom under cap (round-robin bcc/akiec)
    fill_pool: list[tuple[str, Path, Path]] = []
    for dx in ("bcc", "akiec"):
        for pair in groups.get(dx) or []:
            fill_pool.append((dx, pair[0], pair[1]))
    if not fill_pool:
        for pair in groups.get("mel") or []:
            fill_pool.append(("mel", pair[0], pair[1]))
    remaining = room_left()
    for i in range(remaining):
        if not fill_pool:
            break
        dx, src_img, src_lbl = fill_pool[i % len(fill_pool)]
        if not add_one(dx, src_img, src_lbl):
            break

    after = count_split_classes("train")
    dx_after = _count_train_dx(dx_by_image)
    ratio = (
        after[CLASS_NAMES[1]] / after[CLASS_NAMES[0]]
        if after[CLASS_NAMES[0]]
        else None
    )
    return {
        "enabled": True,
        "method": "hard_pair_revolume",
        "mel_multiplier": mel_multiplier,
        "other_multiplier": other_multiplier,
        "malignant_cap_ratio": malignant_cap_ratio,
        "cap_malignant": cap_mal,
        "achieved_malignant_to_benign": ratio,
        "removed_old_bal_copies": removed,
        "before": before,
        "after": after,
        "dx_counts_before": dx_before,
        "dx_counts_after": dx_after,
        "unique_by_dx": unique_by_dx,
        "augmented_added_by_dx": dict(added_by_dx),
        "malignant_added": int(sum(added_by_dx.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare binary YOLO HAM10000 dataset")
    parser.add_argument(
        "--balance-train",
        action="store_true",
        help="Legacy: identical-copy oversample malignant toward ratio vs benign",
    )
    parser.add_argument(
        "--malignant-ratio",
        type=float,
        default=1.0,
        help="Target malignant/benign count ratio in train when --balance-train is set",
    )
    parser.add_argument(
        "--balance-hard-pairs",
        action="store_true",
        help="Mel-focused OpenCV-augmented oversampling (preferred over --balance-train)",
    )
    parser.add_argument(
        "--balance-revolume",
        action="store_true",
        help="Alias for mel-first revolume (same as --balance-hard-pairs with defaults 6.0 / 1.2)",
    )
    parser.add_argument(
        "--mel-multiplier",
        type=float,
        default=6.0,
        help="Target mel count ≈ multiplier × unique mel when --balance-hard-pairs",
    )
    parser.add_argument(
        "--malignant-cap-ratio",
        type=float,
        default=1.2,
        help="Cap total malignant ≤ ratio × benign when --balance-hard-pairs",
    )
    parser.add_argument(
        "--other-multiplier",
        type=float,
        default=1.5,
        help="Mild bcc/akiec multiplier after mel when --balance-hard-pairs",
    )
    parser.add_argument(
        "--preprocess-derm",
        action="store_true",
        help="Apply Gray-World color norm + light hair removal when writing images",
    )
    args = parser.parse_args()
    if args.balance_revolume:
        args.balance_hard_pairs = True
    if args.malignant_ratio <= 0:
        raise SystemExit("--malignant-ratio must be > 0")
    if args.mel_multiplier <= 0:
        raise SystemExit("--mel-multiplier must be > 0")
    if args.malignant_cap_ratio <= 0:
        raise SystemExit("--malignant-cap-ratio must be > 0")
    if args.balance_train and args.balance_hard_pairs:
        raise SystemExit("Use only one of --balance-train or --balance-hard-pairs/--balance-revolume")

    if not META_PATH.is_file():
        raise SystemExit(
            f"Missing metadata: {META_PATH}\nRun: python scripts/download_ham10000.py"
        )
    if not IMAGES_DIR.is_dir() or not any(IMAGES_DIR.iterdir()):
        raise SystemExit(f"No images in {IMAGES_DIR}. Run download first.")
    if not MASKS_DIR.is_dir() or not any(MASKS_DIR.iterdir()):
        raise SystemExit(f"No masks in {MASKS_DIR}. Run download first.")

    meta = pd.read_csv(META_PATH)
    required_cols = {"image_id", "dx", "lesion_id"}
    missing_cols = required_cols - set(meta.columns)
    if missing_cols:
        raise SystemExit(
            f"HAM10000_metadata.csv missing columns: {sorted(missing_cols)}"
        )

    meta = meta.drop_duplicates(subset=["image_id"]).copy()
    meta["dx"] = meta["dx"].astype(str).str.lower().str.strip()
    meta["lesion_id"] = meta["lesion_id"].astype(str).str.strip()
    meta["class_id"] = meta["dx"].map(DX_TO_CLASS)
    unknown = meta[meta["class_id"].isna()]
    if len(unknown):
        print(f"Warning: dropping {len(unknown)} rows with unknown dx: {unknown['dx'].unique()}")
        meta = meta.dropna(subset=["class_id"])
    meta["class_id"] = meta["class_id"].astype(int)

    records = []
    skipped = []
    bbox_areas = []

    for row in tqdm(meta.itertuples(index=False), total=len(meta), desc="Building labels"):
        image_id = row.image_id
        class_id = int(row.class_id)
        dx = row.dx
        lesion_id = row.lesion_id

        image_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = IMAGES_DIR / f"{image_id}{ext}"
            if candidate.is_file():
                image_path = candidate
                break
        if image_path is None:
            skipped.append({"image_id": image_id, "reason": "missing_image"})
            continue

        mask_path = find_mask_for_image(image_id)
        if mask_path is None:
            skipped.append({"image_id": image_id, "reason": "missing_mask"})
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            skipped.append({"image_id": image_id, "reason": "unreadable_mask"})
            continue

        bbox = mask_to_yolo_bbox(mask)
        if bbox is None:
            skipped.append({"image_id": image_id, "reason": "empty_mask"})
            continue

        xc, yc, bw, bh = bbox
        bbox_areas.append(bw * bh)
        records.append(
            {
                "image_id": image_id,
                "lesion_id": lesion_id,
                "image_path": str(image_path),
                "class_id": class_id,
                "dx": dx,
                "label_line": f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n",
            }
        )

    if not records:
        raise SystemExit("No valid samples produced. Check raw images/masks.")

    df = pd.DataFrame(records)

    # One row per lesion for stratified split; all images of a lesion share that split.
    lesion_df = (
        df.groupby("lesion_id", as_index=False)
        .agg(class_id=("class_id", "first"), n_images=("image_id", "count"))
    )
    mixed = (
        df.groupby("lesion_id")["class_id"].nunique().reset_index(name="n_classes")
    )
    mixed_lesions = int((mixed["n_classes"] > 1).sum())
    if mixed_lesions:
        print(f"Warning: {mixed_lesions} lesions have mixed class labels; using first seen.")

    train_val_lesions, test_lesions = train_test_split(
        lesion_df,
        test_size=TEST_RATIO,
        random_state=SEED,
        stratify=lesion_df["class_id"],
    )
    relative_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_lesions, val_lesions = train_test_split(
        train_val_lesions,
        test_size=relative_val,
        random_state=SEED,
        stratify=train_val_lesions["class_id"],
    )

    lesion_to_split = {}
    for split_name, split_lesions in (
        ("train", train_lesions),
        ("val", val_lesions),
        ("test", test_lesions),
    ):
        for lid in split_lesions["lesion_id"]:
            lesion_to_split[lid] = split_name

    df["split"] = df["lesion_id"].map(lesion_to_split)
    assert df["split"].notna().all(), "Every image must map to a lesion split"

    # Leakage check: each lesion appears in exactly one split
    lesion_split_counts = df.groupby("lesion_id")["split"].nunique()
    leaking_lesions = int((lesion_split_counts > 1).sum())
    if leaking_lesions:
        raise SystemExit(f"Lesion leakage detected for {leaking_lesions} lesions")

    splits = {
        "train": df[df["split"] == "train"],
        "val": df[df["split"] == "val"],
        "test": df[df["split"] == "test"],
    }
    reset_processed_dirs()

    split_counts = {}
    lesion_counts = {}
    for split_name, split_df in splits.items():
        class_counter = Counter(split_df["class_id"].tolist())
        split_counts[split_name] = {
            CLASS_NAMES[0]: int(class_counter.get(0, 0)),
            CLASS_NAMES[1]: int(class_counter.get(1, 0)),
            "total": int(len(split_df)),
        }
        lesion_class = Counter(
            split_df.drop_duplicates("lesion_id")["class_id"].tolist()
        )
        lesion_counts[split_name] = {
            CLASS_NAMES[0]: int(lesion_class.get(0, 0)),
            CLASS_NAMES[1]: int(lesion_class.get(1, 0)),
            "total": int(split_df["lesion_id"].nunique()),
        }
        for row in tqdm(
            split_df.itertuples(index=False),
            total=len(split_df),
            desc=f"Writing {split_name}",
        ):
            src = Path(row.image_path)
            dst_img = PROCESSED / "images" / split_name / src.name
            dst_lbl = PROCESSED / "labels" / split_name / f"{src.stem}.txt"
            write_image(dst_img, src, preprocess_derm=args.preprocess_derm)
            dst_lbl.write_text(row.label_line, encoding="utf-8")

    preprocess_samples: list[str] = []
    if args.preprocess_derm:
        print("\nSaving preprocess before/after samples ...")
        preprocess_samples = save_preprocess_samples(df, n=6)

    dx_by_image = {str(r.image_id): str(r.dx) for r in df.itertuples(index=False)}

    balance_info: dict = {"enabled": False}
    if args.balance_hard_pairs:
        print(
            f"\nHard-pair balancing: mel_multiplier={args.mel_multiplier:.2f}, "
            f"cap_ratio={args.malignant_cap_ratio:.2f}, other_multiplier={args.other_multiplier:.2f} ..."
        )
        balance_info = balance_hard_pairs(
            dx_by_image=dx_by_image,
            mel_multiplier=args.mel_multiplier,
            malignant_cap_ratio=args.malignant_cap_ratio,
            other_multiplier=args.other_multiplier,
            seed=SEED,
        )
        split_counts["train"] = balance_info["after"]
        print(json.dumps(balance_info, indent=2))
    elif args.balance_train:
        print(
            f"\nBalancing train malignant toward ratio={args.malignant_ratio:.2f} vs benign ..."
        )
        balance_info = balance_train_malignant(args.malignant_ratio, seed=SEED)
        split_counts["train"] = balance_info["after"]
        print(json.dumps(balance_info, indent=2))

    write_data_yaml()

    report = {
        "seed": SEED,
        "split_level": "lesion_id",
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "class_mapping": {
            "benign": sorted(BENIGN),
            "malignant": sorted(MALIGNANT),
        },
        "total_valid": len(df),
        "total_lesions": int(df["lesion_id"].nunique()),
        "leaking_lesions": leaking_lesions,
        "mixed_class_lesions": mixed_lesions,
        "skipped": len(skipped),
        "skipped_reasons": dict(Counter(s["reason"] for s in skipped)),
        "split_counts": split_counts,
        "lesion_counts": lesion_counts,
        "train_balance": balance_info,
        "preprocess_derm": bool(args.preprocess_derm),
        "preprocess_samples": preprocess_samples,
        "bbox_area_mean": float(np.mean(bbox_areas)) if bbox_areas else None,
        "bbox_area_median": float(np.median(bbox_areas)) if bbox_areas else None,
        "config": str(CONFIG_PATH),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    skipped_path = PROCESSED / "skipped.json"
    skipped_path.write_text(json.dumps(skipped, indent=2), encoding="utf-8")

    print("\n=== Prepare summary ===")
    print(json.dumps(report, indent=2))
    print(f"\nWrote YOLO dataset to {PROCESSED}")
    print(f"Updated config: {CONFIG_PATH}")


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    main()
