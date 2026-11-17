"""
Full image-level failure audit for a YOLO binary skin model.

Uses ground-truth labels from data/processed/labels/<split>
and images from data/raw/images (raw domain for yolov5su_balanced).

Usage:
  python scripts/audit_failures.py
  python scripts/audit_failures.py --split val --tag yolov5su_balanced_val
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

from test_generalization import CLASS_NAMES, PROJECT_ROOT, load_gt_labels, predict_top_class

META_PATH = PROJECT_ROOT / "data" / "raw" / "metadata" / "HAM10000_metadata.csv"
RAW_IMAGES = PROJECT_ROOT / "data" / "raw" / "images"
DEFAULT_WEIGHTS = (
    PROJECT_ROOT / "runs" / "detect" / "yolov5su_balanced" / "weights" / "best.pt"
)
OUT_ROOT = PROJECT_ROOT / "runs" / "error_audit"

MALIGNANT_DX = {"mel", "bcc", "akiec"}
BENIGN_DX = {"nv", "bkl", "df", "vasc"}


def load_dx_map() -> dict[str, str]:
    meta = pd.read_csv(META_PATH).drop_duplicates("image_id")
    return {
        str(row.image_id): str(row.dx).lower()
        for row in meta.itertuples(index=False)
    }


def find_raw_image(stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = RAW_IMAGES / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def summarize(records: list[dict], n_benign: int, n_mal: int) -> dict:
    confusion = Counter()
    by_dx_error: dict[str, Counter] = defaultdict(Counter)
    conf_correct: list[float] = []
    conf_wrong: list[float] = []
    failures: list[dict] = []

    for r in records:
        key = f"{r['true_name']}->{r['pred_name']}"
        confusion[key] += 1
        if r["correct"]:
            conf_correct.append(r["conf"])
        else:
            conf_wrong.append(r["conf"])
            failures.append(r)
            err_key = key
            by_dx_error[r["dx"]][err_key] += 1

    mal_acc = confusion["malignant->malignant"] / n_mal if n_mal else 0.0
    ben_acc = confusion["benign->benign"] / n_benign if n_benign else 0.0
    overall = (
        (confusion["benign->benign"] + confusion["malignant->malignant"])
        / (n_benign + n_mal)
        if (n_benign + n_mal)
        else 0.0
    )
    mal_to_ben = confusion["malignant->benign"] / n_mal if n_mal else 0.0
    ben_to_mal = confusion["benign->malignant"] / n_benign if n_benign else 0.0

    # High vs low confidence wrong predictions
    high_conf_wrong = [f for f in failures if f["conf"] >= 0.7]
    low_conf_wrong = [f for f in failures if 0 < f["conf"] < 0.5]

    return {
        "n_images": n_benign + n_mal,
        "n_benign": n_benign,
        "n_malignant": n_mal,
        "confusion": dict(confusion),
        "benign_accuracy": ben_acc,
        "malignant_accuracy": mal_acc,
        "overall_class_accuracy": overall,
        "malignant_to_benign": mal_to_ben,
        "benign_to_malignant": ben_to_mal,
        "mean_conf_correct": float(np.mean(conf_correct)) if conf_correct else None,
        "mean_conf_wrong": float(np.mean(conf_wrong)) if conf_wrong else None,
        "n_failures": len(failures),
        "n_high_conf_wrong_ge_0_7": len(high_conf_wrong),
        "n_low_conf_wrong_lt_0_5": len(low_conf_wrong),
        "errors_by_dx": {dx: dict(c) for dx, c in sorted(by_dx_error.items())},
        "failures": sorted(failures, key=lambda x: (-x["conf"], x["image"])),
    }


def save_grids(failures: list[dict], out_dir: Path, per_grid: int = 16) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    groups = {
        "mal_to_ben": [f for f in failures if f["true_name"] == "malignant" and f["pred_name"] == "benign"],
        "ben_to_mal": [f for f in failures if f["true_name"] == "benign" and f["pred_name"] == "malignant"],
        "mal_to_miss": [f for f in failures if f["true_name"] == "malignant" and f["pred_name"] == "miss"],
    }
    for group_name, items in groups.items():
        if not items:
            continue
        n_pages = max(1, math.ceil(len(items) / per_grid))
        for page in range(n_pages):
            chunk = items[page * per_grid : (page + 1) * per_grid]
            cols = 4
            rows = math.ceil(len(chunk) / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
            axes = np.array(axes).reshape(-1)
            for ax in axes:
                ax.axis("off")
            for ax, item in zip(axes, chunk):
                img = cv2.imread(item["image_path"])
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(rgb)
                ax.set_title(
                    f"{item['image']}\n{item['dx']} conf={item['conf']:.2f}",
                    fontsize=8,
                )
                ax.axis("off")
            out = out_dir / f"{group_name}_{page + 1:02d}.jpg"
            fig.tight_layout()
            fig.savefig(out, dpi=120)
            plt.close(fig)
            saved.append(str(out))
    return saved


def audit(
    weights: Path,
    split: str,
    conf: float,
    iou: float,
    tag: str,
) -> dict:
    gt = load_gt_labels(split)
    dx_map = load_dx_map()
    model = YOLO(str(weights))

    records: list[dict] = []
    missing_images = 0
    for stem, true_cls in sorted(gt.items()):
        img_path = find_raw_image(stem)
        if img_path is None:
            missing_images += 1
            continue
        pred_cls, pred_conf = predict_top_class(model, img_path, conf, iou)
        true_name = CLASS_NAMES[true_cls]
        if pred_cls is None:
            pred_name = "miss"
            pred_conf_f = 0.0
            correct = False
        else:
            pred_name = CLASS_NAMES.get(pred_cls, str(pred_cls))
            pred_conf_f = float(pred_conf)
            correct = pred_cls == true_cls
        dx = dx_map.get(stem, "unknown")
        records.append(
            {
                "image": stem,
                "image_path": str(img_path),
                "dx": dx,
                "true_cls": true_cls,
                "true_name": true_name,
                "pred_cls": pred_cls,
                "pred_name": pred_name,
                "conf": pred_conf_f,
                "correct": correct,
            }
        )

    n_benign = sum(1 for r in records if r["true_cls"] == 0)
    n_mal = sum(1 for r in records if r["true_cls"] == 1)
    summary = summarize(records, n_benign, n_mal)
    report = {
        "tag": tag,
        "split": split,
        "weights": str(weights),
        "image_source": str(RAW_IMAGES),
        "conf": conf,
        "iou": iou,
        "missing_raw_images": missing_images,
        "summary": {k: v for k, v in summary.items() if k != "failures"},
        "failures": summary["failures"],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Full failure audit on raw images")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--tag", type=str, default="yolov5su_balanced")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"Missing weights: {weights}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    report = audit(weights, args.split, args.conf, args.iou, args.tag)
    grids = save_grids(report["failures"], OUT_ROOT / "grids" / args.tag)
    report["grids"] = grids

    out_json = OUT_ROOT / f"{args.tag}_{args.split}_failures.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    s = report["summary"]
    print(f"Wrote {out_json}")
    print(f"grids={len(grids)}")
    print(
        f"overall={s['overall_class_accuracy']:.3f} "
        f"mal_acc={s['malignant_accuracy']:.3f} "
        f"mal->ben={s['malignant_to_benign']:.3f} "
        f"ben->mal={s['benign_to_malignant']:.3f}"
    )
    print(f"failures={s['n_failures']} high_conf_wrong={s['n_high_conf_wrong_ge_0_7']}")
    print("errors_by_dx=", json.dumps(s["errors_by_dx"], indent=2))


if __name__ == "__main__":
    main()
