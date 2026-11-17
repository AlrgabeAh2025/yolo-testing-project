"""
Test generalization and benign/malignant discrimination on the held-out test set.

For each trained model:
  - image-level confusion (GT class vs top prediction)
  - miss / wrong-class / correct rates
  - mean confidence on correct vs confused predictions
  - val→test gap from compare_summary.json (overfitting signal)
  - save a failure mosaic (malignant→benign and benign→malignant samples)

Usage:
  python scripts/test_generalization.py
  python scripts/test_generalization.py --conf 0.25 --iou 0.5
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "configs" / "data.yaml"
COMPARE_PATH = PROJECT_ROOT / "runs" / "compare_summary.json"
OUT_DIR = PROJECT_ROOT / "runs" / "generalization"
PROCESSED = PROJECT_ROOT / "data" / "processed"

MODELS = {
    "yolov5su_balanced": PROJECT_ROOT / "runs" / "detect" / "yolov5su_balanced" / "weights" / "best.pt",
    "yolov8s_balanced": PROJECT_ROOT / "runs" / "detect" / "yolov8s_balanced" / "weights" / "best.pt",
    "yolov3-tinyu_balanced": PROJECT_ROOT
    / "runs"
    / "detect"
    / "yolov3-tinyu_balanced"
    / "weights"
    / "best.pt",
}
CLASS_NAMES = {0: "benign", 1: "malignant"}

# Image-level acceptance thresholds on the held-out test set
GATE = {
    "malignant_accuracy_min": 0.75,
    "malignant_to_benign_max": 0.20,
    "benign_to_malignant_max": 0.10,
    "overall_class_accuracy_min": 0.85,
    "malignant_recall_drop_max": 0.05,
}


def load_gt_labels(split: str = "test") -> dict[str, int]:
    """Map image stem -> ground-truth class (first box; dataset has 1 lesion/image)."""
    label_dir = PROCESSED / "labels" / split
    gt = {}
    for path in sorted(label_dir.glob("*.txt")):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            continue
        cls = int(lines[0].split()[0])
        gt[path.stem] = cls
    return gt


def predict_top_class(model: YOLO, image_path: Path, conf: float, iou: float) -> tuple[int | None, float]:
    """Return (class_id, confidence) of highest-conf detection, or (None, 0)."""
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        verbose=False,
        imgsz=640,
    )
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return None, 0.0
    confs = r0.boxes.conf.cpu().numpy()
    clss = r0.boxes.cls.cpu().numpy().astype(int)
    idx = int(np.argmax(confs))
    return int(clss[idx]), float(confs[idx])


def predict_class_scores(
    model: YOLO, image_path: Path, conf: float, iou: float
) -> tuple[float, float]:
    """Return (best_benign_conf, best_malignant_conf)."""
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        verbose=False,
        imgsz=640,
    )
    r0 = results[0]
    best_ben = 0.0
    best_mal = 0.0
    if r0.boxes is None or len(r0.boxes) == 0:
        return best_ben, best_mal
    confs = r0.boxes.conf.cpu().numpy()
    clss = r0.boxes.cls.cpu().numpy().astype(int)
    for c, score in zip(clss, confs):
        if int(c) == 0:
            best_ben = max(best_ben, float(score))
        elif int(c) == 1:
            best_mal = max(best_mal, float(score))
    return best_ben, best_mal


def decide_asymmetric(
    best_ben: float, best_mal: float, mal_thr: float, ben_thr: float
) -> tuple[int | None, float]:
    """Malignant-first asymmetric decision. Returns (class_id|None, decision_conf)."""
    if best_mal >= mal_thr:
        return 1, best_mal
    if best_ben >= ben_thr:
        return 0, best_ben
    return None, 0.0


def resolve_image_path(stem: str, image_source: str, split: str = "test") -> Path | None:
    if image_source == "raw":
        root = PROJECT_ROOT / "data" / "raw" / "images"
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = root / f"{stem}{ext}"
            if candidate.is_file():
                return candidate
        return None
    img_dir = PROCESSED / "images" / split
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = img_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def evaluate_model(
    name: str,
    weights: Path,
    gt: dict[str, int],
    conf: float,
    iou: float,
    max_images: int | None,
    decision: str = "top1",
    mal_thr: float | None = None,
    ben_thr: float | None = None,
    image_source: str = "processed",
    split: str = "test",
) -> dict:
    model = YOLO(str(weights))
    stems = sorted(gt.keys())
    if max_images is not None:
        stems = stems[:max_images]

    confusion = {
        "benign->benign": 0,
        "benign->malignant": 0,
        "benign->miss": 0,
        "malignant->malignant": 0,
        "malignant->benign": 0,
        "malignant->miss": 0,
    }
    conf_correct: list[float] = []
    conf_confused: list[float] = []
    failures: dict[str, list[dict]] = defaultdict(list)

    use_asymmetric = decision == "asymmetric"
    if use_asymmetric and (mal_thr is None or ben_thr is None):
        raise ValueError("asymmetric decision requires mal_thr and ben_thr")

    for stem in stems:
        true_cls = gt[stem]
        true_name = CLASS_NAMES[true_cls]
        img_path = resolve_image_path(stem, image_source, split=split)
        if img_path is None:
            continue

        if use_asymmetric:
            best_ben, best_mal = predict_class_scores(model, img_path, conf, iou)
            pred_cls, pred_conf = decide_asymmetric(best_ben, best_mal, mal_thr, ben_thr)  # type: ignore[arg-type]
        else:
            pred_cls, pred_conf = predict_top_class(model, img_path, conf, iou)

        if pred_cls is None:
            key = f"{true_name}->miss"
            confusion[key] += 1
            failures[key].append({"image": stem, "conf": 0.0})
            continue

        pred_name = CLASS_NAMES.get(pred_cls, str(pred_cls))
        key = f"{true_name}->{pred_name}"
        if key not in confusion:
            confusion[key] = 0
        confusion[key] += 1

        if pred_cls == true_cls:
            conf_correct.append(pred_conf)
        else:
            conf_confused.append(pred_conf)
            failures[key].append({"image": stem, "conf": pred_conf})

    n_benign = sum(1 for s in stems if gt[s] == 0)
    n_mal = sum(1 for s in stems if gt[s] == 1)
    total = n_benign + n_mal

    benign_acc = confusion["benign->benign"] / n_benign if n_benign else 0.0
    mal_acc = confusion["malignant->malignant"] / n_mal if n_mal else 0.0
    overall_acc = (confusion["benign->benign"] + confusion["malignant->malignant"]) / total if total else 0.0

    # Image-level precision/recall for malignant (clinically critical)
    tp = confusion["malignant->malignant"]
    fp = confusion["benign->malignant"]
    fn = confusion["malignant->benign"] + confusion["malignant->miss"]
    mal_precision = tp / (tp + fp) if (tp + fp) else 0.0
    mal_recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "model": name,
        "weights": str(weights),
        "decision": decision,
        "image_source": image_source,
        "mal_thr": mal_thr,
        "ben_thr": ben_thr,
        "n_images": total,
        "n_benign": n_benign,
        "n_malignant": n_mal,
        "confusion": confusion,
        "overall_class_accuracy": overall_acc,
        "benign_accuracy": benign_acc,
        "malignant_accuracy": mal_acc,
        "malignant_precision_image": mal_precision,
        "malignant_recall_image": mal_recall,
        "mean_conf_correct": float(np.mean(conf_correct)) if conf_correct else None,
        "mean_conf_confused": float(np.mean(conf_confused)) if conf_confused else None,
        "failure_examples": {
            k: v[:12] for k, v in failures.items() if k.endswith("->benign") or k.endswith("->malignant") or k.endswith("->miss")
        },
    }


def val_test_gap(compare: dict, model: str) -> dict | None:
    entry = compare.get(model)
    if not entry or not entry.get("val") or not entry.get("test"):
        return None
    v, t = entry["val"], entry["test"]
    return {
        "map50_val": v["map50"],
        "map50_test": t["map50"],
        "map50_drop": v["map50"] - t["map50"],
        "malignant_recall_val": v.get("malignant_recall"),
        "malignant_recall_test": t.get("malignant_recall"),
        "malignant_recall_drop": (v.get("malignant_recall") or 0) - (t.get("malignant_recall") or 0),
        "benign_recall_val": v.get("benign_recall"),
        "benign_recall_test": t.get("benign_recall"),
    }


def defects_for(result: dict, gap: dict | None) -> list[str]:
    defects = []
    conf = result["confusion"]
    n_mal = result["n_malignant"]
    n_ben = result["n_benign"]

    mal_as_benign = conf["malignant->benign"] / n_mal if n_mal else 0
    ben_as_mal = conf["benign->malignant"] / n_ben if n_ben else 0
    mal_miss = conf["malignant->miss"] / n_mal if n_mal else 0

    if mal_as_benign >= 0.20:
        defects.append(
            f"High malignant→benign confusion ({mal_as_benign:.1%}): model often calls cancer benign."
        )
    elif mal_as_benign >= 0.10:
        defects.append(
            f"Moderate malignant→benign confusion ({mal_as_benign:.1%})."
        )

    if ben_as_mal >= 0.15:
        defects.append(
            f"High benign→malignant false alarms ({ben_as_mal:.1%})."
        )
    elif ben_as_mal >= 0.08:
        defects.append(
            f"Moderate benign→malignant false alarms ({ben_as_mal:.1%})."
        )

    if mal_miss >= 0.10:
        defects.append(f"Misses malignant lesions entirely ({mal_miss:.1%}).")

    if result["malignant_accuracy"] + 0.15 < result["benign_accuracy"]:
        defects.append(
            "Class imbalance bias: much stronger on benign than malignant "
            f"(benign_acc={result['benign_accuracy']:.1%}, malignant_acc={result['malignant_accuracy']:.1%})."
        )

    if gap and gap["map50_drop"] >= 0.05:
        defects.append(
            f"Generalization gap val→test mAP50 drop={gap['map50_drop']:.3f}."
        )
    if gap and gap["malignant_recall_drop"] >= 0.05:
        defects.append(
            f"Malignant recall drops on test by {gap['malignant_recall_drop']:.3f}."
        )

    mc = result.get("mean_conf_confused")
    cc = result.get("mean_conf_correct")
    if mc is not None and cc is not None and mc >= cc - 0.05:
        defects.append(
            f"Confused predictions are nearly as confident as correct ones "
            f"(correct={cc:.2f}, confused={mc:.2f})."
        )

    if not defects:
        defects.append("No major discrimination defects under the configured thresholds.")
    return defects


def generalization_gate(result: dict, gap: dict | None) -> tuple[bool, list[str]]:
    """Check image-level metrics against acceptance thresholds."""
    conf = result["confusion"]
    n_mal = max(1, result["n_malignant"])
    n_ben = max(1, result["n_benign"])
    mal_to_ben = conf["malignant->benign"] / n_mal
    ben_to_mal = conf["benign->malignant"] / n_ben
    reasons = []

    if result["malignant_accuracy"] < GATE["malignant_accuracy_min"]:
        reasons.append(
            f"malignant_accuracy {result['malignant_accuracy']:.3f} < {GATE['malignant_accuracy_min']}"
        )
    if mal_to_ben > GATE["malignant_to_benign_max"]:
        reasons.append(
            f"malignant→benign {mal_to_ben:.3f} > {GATE['malignant_to_benign_max']}"
        )
    if ben_to_mal > GATE["benign_to_malignant_max"]:
        reasons.append(
            f"benign→malignant {ben_to_mal:.3f} > {GATE['benign_to_malignant_max']}"
        )
    if result["overall_class_accuracy"] < GATE["overall_class_accuracy_min"]:
        reasons.append(
            f"overall_class_accuracy {result['overall_class_accuracy']:.3f} < {GATE['overall_class_accuracy_min']}"
        )
    if gap and gap.get("malignant_recall_drop") is not None:
        if gap["malignant_recall_drop"] > GATE["malignant_recall_drop_max"]:
            reasons.append(
                f"malignant_recall_drop {gap['malignant_recall_drop']:.3f} > {GATE['malignant_recall_drop_max']}"
            )

    return (len(reasons) == 0), reasons


def save_failure_mosaic(model_name: str, result: dict, seed: int = 42) -> Path | None:
    rng = random.Random(seed)
    samples = []
    for key in ("malignant->benign", "benign->malignant", "malignant->miss"):
        items = result.get("failure_examples", {}).get(key, [])
        chosen = rng.sample(items, k=min(4, len(items))) if items else []
        for item in chosen:
            samples.append((key, item["image"], item.get("conf", 0.0)))
    if not samples:
        return None

    cols = 4
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    img_dir = PROCESSED / "images" / "test"
    for ax, (key, stem, conf) in zip(axes, samples):
        path = next((img_dir / f"{stem}{ext}" for ext in (".jpg", ".jpeg", ".png") if (img_dir / f"{stem}{ext}").is_file()), None)
        if path is None:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{key}\n{stem} conf={conf:.2f}", fontsize=8)
        ax.axis("off")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{model_name}_failures.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalization / class-separation test")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for quick smoke tests")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model keys to evaluate (default: all existing weights)",
    )
    parser.add_argument(
        "--require-gate",
        action="store_true",
        help="Exit non-zero if any evaluated model fails the strict generalization gate",
    )
    parser.add_argument(
        "--compare-key",
        type=str,
        default=None,
        help="Override compare_summary key used for val→test gap (default: model name)",
    )
    parser.add_argument(
        "--decision",
        type=str,
        default="top1",
        choices=["top1", "asymmetric"],
        help="Image-level decision policy",
    )
    parser.add_argument(
        "--decision-config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "decision_asymmetric.yaml"),
        help="YAML with mal_thr/ben_thr for asymmetric decision",
    )
    parser.add_argument(
        "--image-source",
        type=str,
        default="processed",
        choices=["processed", "raw"],
        help="Use processed split images or raw HAM images",
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()

    if not DATA_YAML.is_file():
        raise SystemExit(f"Missing {DATA_YAML}")

    gt = load_gt_labels(args.split)
    if not gt:
        raise SystemExit(f"No {args.split} labels found. Run prepare_yolo_dataset.py first.")

    mal_thr = None
    ben_thr = None
    detect_conf = args.conf
    if args.decision == "asymmetric":
        cfg_path = Path(args.decision_config)
        if not cfg_path.is_file():
            raise SystemExit(f"Missing decision config: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            dcfg = yaml.safe_load(f)
        mal_thr = float(dcfg["mal_thr"])
        ben_thr = float(dcfg["ben_thr"])
        detect_conf = float(dcfg.get("detect_conf", args.conf))

    compare = {}
    if COMPARE_PATH.is_file():
        compare = json.loads(COMPARE_PATH.read_text(encoding="utf-8"))

    selected = args.models or list(MODELS.keys())
    unknown = [m for m in selected if m not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}. Choose from {list(MODELS)}")

    report = {
        "conf_threshold": detect_conf,
        "iou_threshold": args.iou,
        "decision": args.decision,
        "image_source": args.image_source,
        "split": args.split,
        "mal_thr": mal_thr,
        "ben_thr": ben_thr,
        "test_images": len(gt),
        "gate": GATE,
        "models": {},
    }

    print("=== Generalization test (held-out test set) ===")
    print(
        f"split={args.split} images={len(gt)} conf={detect_conf} iou={args.iou} "
        f"decision={args.decision} image_source={args.image_source}"
    )
    if args.decision == "asymmetric":
        print(f"asymmetric thr: mal>={mal_thr} ben>={ben_thr}")
    print(f"gate={GATE}\n")

    any_fail = False
    for name in selected:
        weights = MODELS[name]
        if not weights.is_file():
            print(f"[skip] {name}: missing {weights}")
            continue
        print(f">>> Evaluating {name} ...")
        result = evaluate_model(
            name,
            weights,
            gt,
            detect_conf,
            args.iou,
            args.max_images,
            decision=args.decision,
            mal_thr=mal_thr,
            ben_thr=ben_thr,
            image_source=args.image_source,
            split=args.split,
        )
        gap_key = args.compare_key or name
        gap = val_test_gap(compare, gap_key)
        result["val_test_gap"] = gap
        result["defects"] = defects_for(result, gap)
        gate_ok, gate_reasons = generalization_gate(result, gap)
        result["gate_pass"] = gate_ok
        result["gate_reasons"] = gate_reasons
        mosaic = save_failure_mosaic(name, result)
        result["failure_mosaic"] = str(mosaic) if mosaic else None
        report["models"][name] = result

        print(f"  overall_class_acc={result['overall_class_accuracy']:.3f}")
        print(f"  benign_acc={result['benign_accuracy']:.3f}  malignant_acc={result['malignant_accuracy']:.3f}")
        print(f"  malignant_P={result['malignant_precision_image']:.3f}  malignant_R={result['malignant_recall_image']:.3f}")
        print(f"  confusion={result['confusion']}")
        print(f"  GATE: {'PASS' if gate_ok else 'FAIL'}")
        for r in gate_reasons:
            print(f"  - gate: {r}")
        for d in result["defects"]:
            print(f"  - {d}")
        print()
        if not gate_ok:
            any_fail = True

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "generalization_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")
    if args.require_gate and any_fail:
        raise SystemExit("FAIL generalization gate")


if __name__ == "__main__":
    main()
