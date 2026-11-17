"""
Image-level confusion stratified by HAM10000 diagnosis (dx).

Uses the same top-confidence decision policy as test_generalization.py.

Usage:
  python scripts/analyze_confusion_by_dx.py
  python scripts/analyze_confusion_by_dx.py --weights runs/detect/yolov5su_balanced/weights/best.pt --tag balanced
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from test_generalization import (
    CLASS_NAMES,
    OUT_DIR,
    PROCESSED,
    PROJECT_ROOT,
    load_gt_labels,
    predict_top_class,
)

META_PATH = PROJECT_ROOT / "data" / "raw" / "metadata" / "HAM10000_metadata.csv"
DEFAULT_WEIGHTS = (
    PROJECT_ROOT / "runs" / "detect" / "yolov5su_balanced" / "weights" / "best.pt"
)

MALIGNANT_DX = {"mel", "bcc", "akiec"}
BENIGN_DX = {"nv", "bkl", "df", "vasc"}


def load_dx_map() -> dict[str, str]:
    meta = pd.read_csv(META_PATH).drop_duplicates("image_id")
    return {
        str(row.image_id): str(row.dx).lower()
        for row in meta.itertuples(index=False)
    }


def analyze(
    weights: Path,
    conf: float,
    iou: float,
    tag: str,
) -> dict:
    gt = load_gt_labels("test")
    dx_map = load_dx_map()
    model = YOLO(str(weights))
    img_dir = PROCESSED / "images" / "test"

    # gt_dx -> pred_label -> count
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hard_ids: dict[str, list[dict]] = {
        "mel_to_benign": [],
        "nv_bkl_to_malignant": [],
        "other_malignant_to_benign": [],
        "other_benign_to_malignant": [],
    }
    conf_by_bucket: dict[str, list[float]] = defaultdict(list)

    for stem, true_cls in sorted(gt.items()):
        dx = dx_map.get(stem, "unknown")
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = img_dir / f"{stem}{ext}"
            if candidate.is_file():
                img_path = candidate
                break
        if img_path is None:
            continue

        pred_cls, pred_conf = predict_top_class(model, img_path, conf, iou)
        if pred_cls is None:
            pred_name = "miss"
        else:
            pred_name = CLASS_NAMES.get(pred_cls, str(pred_cls))

        matrix[dx][pred_name] += 1
        conf_by_bucket[f"{dx}->{pred_name}"].append(float(pred_conf))

        true_name = CLASS_NAMES[true_cls]
        if true_name == "malignant" and pred_name == "benign":
            item = {"image": stem, "dx": dx, "conf": float(pred_conf)}
            if dx == "mel":
                hard_ids["mel_to_benign"].append(item)
            else:
                hard_ids["other_malignant_to_benign"].append(item)
        elif true_name == "benign" and pred_name == "malignant":
            item = {"image": stem, "dx": dx, "conf": float(pred_conf)}
            if dx in {"nv", "bkl"}:
                hard_ids["nv_bkl_to_malignant"].append(item)
            else:
                hard_ids["other_benign_to_malignant"].append(item)

    # Rates per diagnosis
    dx_rates: dict[str, dict] = {}
    for dx, preds in sorted(matrix.items()):
        total = sum(preds.values())
        if dx in MALIGNANT_DX:
            correct = preds.get("malignant", 0)
            wrong_benign = preds.get("benign", 0)
            miss = preds.get("miss", 0)
            dx_rates[dx] = {
                "n": total,
                "role": "malignant",
                "accuracy": correct / total if total else 0.0,
                "to_benign": wrong_benign / total if total else 0.0,
                "to_miss": miss / total if total else 0.0,
                "preds": dict(preds),
            }
        elif dx in BENIGN_DX:
            correct = preds.get("benign", 0)
            wrong_mal = preds.get("malignant", 0)
            miss = preds.get("miss", 0)
            dx_rates[dx] = {
                "n": total,
                "role": "benign",
                "accuracy": correct / total if total else 0.0,
                "to_malignant": wrong_mal / total if total else 0.0,
                "to_miss": miss / total if total else 0.0,
                "preds": dict(preds),
            }
        else:
            dx_rates[dx] = {"n": total, "role": "unknown", "preds": dict(preds)}

    n_mel = dx_rates.get("mel", {}).get("n", 0) or 0
    n_nv = dx_rates.get("nv", {}).get("n", 0) or 0
    n_bkl = dx_rates.get("bkl", {}).get("n", 0) or 0
    mel_to_ben = dx_rates.get("mel", {}).get("preds", {}).get("benign", 0)
    nv_to_mal = dx_rates.get("nv", {}).get("preds", {}).get("malignant", 0)
    bkl_to_mal = dx_rates.get("bkl", {}).get("preds", {}).get("malignant", 0)

    summary = {
        "mel_to_benign_rate": (mel_to_ben / n_mel) if n_mel else None,
        "nv_to_malignant_rate": (nv_to_mal / n_nv) if n_nv else None,
        "bkl_to_malignant_rate": (bkl_to_mal / n_bkl) if n_bkl else None,
        "nv_bkl_to_malignant_rate": (
            (nv_to_mal + bkl_to_mal) / (n_nv + n_bkl) if (n_nv + n_bkl) else None
        ),
        "n_mel_to_benign": mel_to_ben,
        "n_nv_bkl_to_malignant": nv_to_mal + bkl_to_mal,
    }

    mean_conf = {
        k: (sum(v) / len(v) if v else None) for k, v in sorted(conf_by_bucket.items())
    }

    report = {
        "tag": tag,
        "weights": str(weights),
        "conf": conf,
        "iou": iou,
        "n_images": len(gt),
        "matrix_gt_dx_to_pred": {dx: dict(preds) for dx, preds in sorted(matrix.items())},
        "dx_rates": dx_rates,
        "hard_pair_summary": summary,
        "mean_conf_by_dx_pred": mean_conf,
        "hard_ids": {
            k: sorted(v, key=lambda x: -x["conf"]) for k, v in hard_ids.items()
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Confusion by HAM10000 dx")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--tag", type=str, default="yolov5su_balanced")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default: runs/generalization/confusion_by_dx_<tag>.json)",
    )
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"Missing weights: {weights}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.out)
        if args.out
        else OUT_DIR / f"confusion_by_dx_{args.tag}.json"
    )
    # Also always write/merge canonical baseline name when tag is balanced
    report = analyze(weights, args.conf, args.iou, args.tag)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    canonical = OUT_DIR / "confusion_by_dx.json"
    if args.tag == "yolov5su_balanced" or args.out is None:
        if args.tag == "yolov5su_balanced":
            canonical.write_text(json.dumps(report, indent=2), encoding="utf-8")

    hs = report["hard_pair_summary"]
    print(f"Wrote {out_path}")
    if args.tag == "yolov5su_balanced":
        print(f"Also wrote {canonical}")
    print(f"mel→benign={hs['mel_to_benign_rate']}")
    print(f"nv→malignant={hs['nv_to_malignant_rate']}")
    print(f"bkl→malignant={hs['bkl_to_malignant_rate']}")
    print(f"nv+bkl→malignant={hs['nv_bkl_to_malignant_rate']}")
    for dx, info in report["dx_rates"].items():
        print(f"  {dx}: {info}")


if __name__ == "__main__":
    main()
