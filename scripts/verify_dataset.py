"""
Statistical + visual verification of the processed YOLO skin-cancer dataset.

Usage (from project root):
  python scripts/verify_dataset.py
  python scripts/verify_dataset.py --samples 12
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
OUT_DIR = PROJECT_ROOT / "runs" / "verify"


def load_names() -> dict[int, str]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {0: "benign", 1: "malignant"})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def parse_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = parts
        boxes.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return boxes


def draw_yolo_boxes(image_bgr, boxes, names: dict[int, str]):
    h, w = image_bgr.shape[:2]
    colors = {0: (40, 180, 40), 1: (40, 40, 220)}  # BGR: green benign, red malignant
    for cls, xc, yc, bw, bh in boxes:
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        color = colors.get(cls, (255, 255, 0))
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        label = names.get(cls, str(cls))
        cv2.putText(
            image_bgr,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return image_bgr


def summarize_split(split: str, names: dict[int, str]) -> dict:
    img_dir = PROCESSED / "images" / split
    lbl_dir = PROCESSED / "labels" / split
    images = sorted(img_dir.glob("*.*"))
    class_counts = Counter()
    missing_labels = 0
    empty_labels = 0
    bad_boxes = 0
    areas = []

    for img in images:
        lbl = lbl_dir / f"{img.stem}.txt"
        boxes = parse_label(lbl)
        if not lbl.is_file():
            missing_labels += 1
            continue
        if not boxes:
            empty_labels += 1
            continue
        for cls, xc, yc, bw, bh in boxes:
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                bad_boxes += 1
            class_counts[names.get(cls, str(cls))] += 1
            areas.append(bw * bh)

    return {
        "images": len(images),
        "class_counts": dict(class_counts),
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "bad_boxes": bad_boxes,
        "mean_bbox_area": float(sum(areas) / len(areas)) if areas else None,
    }


def save_mosaic(samples: int, names: dict[int, str], seed: int = 42) -> Path:
    rng = random.Random(seed)
    all_images = []
    for split in ("train", "val", "test"):
        all_images.extend((split, p) for p in (PROCESSED / "images" / split).glob("*.*"))
    if not all_images:
        raise SystemExit("No processed images found. Run prepare_yolo_dataset.py first.")

    chosen = rng.sample(all_images, k=min(samples, len(all_images)))
    cols = 4
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax in axes:
        ax.axis("off")

    for ax, (split, img_path) in zip(axes, chosen):
        image = cv2.imread(str(img_path))
        if image is None:
            ax.set_title(f"{split}/{img_path.name} (unreadable)")
            continue
        boxes = parse_label(PROCESSED / "labels" / split / f"{img_path.stem}.txt")
        vis = draw_yolo_boxes(image.copy(), boxes, names)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{split} | {img_path.stem}", fontsize=9)
        ax.axis("off")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "sample_mosaic.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify YOLO HAM10000 binary dataset")
    parser.add_argument("--samples", type=int, default=12, help="Mosaic sample count")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not PROCESSED.exists():
        raise SystemExit(f"Missing processed dataset at {PROCESSED}")

    names = load_names()
    print("=== Dataset verification ===")
    for split in ("train", "val", "test"):
        summary = summarize_split(split, names)
        print(f"\n[{split}]")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    mosaic = save_mosaic(args.samples, names, seed=args.seed)
    print(f"\nSaved visual mosaic: {mosaic}")
    print("OK: verification finished.")


if __name__ == "__main__":
    main()
