"""
Calibrate asymmetric image-level decision thresholds on val (raw images),
then evaluate on held-out test. Never uses test labels for threshold search.

Decision rule:
  - If best malignant conf >= mal_thr  -> malignant
  - Elif best benign conf >= ben_thr   -> benign
  - Else                               -> miss (counts against accuracy)

Usage:
  python scripts/calibrate_asymmetric_decision.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

from audit_failures import find_raw_image
from test_generalization import CLASS_NAMES, GATE, PROJECT_ROOT, generalization_gate, load_gt_labels

DEFAULT_WEIGHTS = (
    PROJECT_ROOT / "runs" / "detect" / "yolov5su_balanced" / "weights" / "best.pt"
)
OUT_YAML = PROJECT_ROOT / "configs" / "decision_asymmetric.yaml"
OUT_JSON = PROJECT_ROOT / "runs" / "error_audit" / "asymmetric_calibration.json"
GATE_STATUS = PROJECT_ROOT / "runs" / "generalization" / "gate_status.json"


def collect_scores(
    model: YOLO,
    gt: dict[str, int],
    detect_conf: float,
    iou: float,
) -> list[dict]:
    rows: list[dict] = []
    for stem, true_cls in sorted(gt.items()):
        img_path = find_raw_image(stem)
        if img_path is None:
            continue
        results = model.predict(
            source=str(img_path),
            conf=detect_conf,
            iou=iou,
            verbose=False,
            imgsz=640,
        )
        r0 = results[0]
        best_ben = 0.0
        best_mal = 0.0
        if r0.boxes is not None and len(r0.boxes) > 0:
            confs = r0.boxes.conf.cpu().numpy()
            clss = r0.boxes.cls.cpu().numpy().astype(int)
            for c, conf in zip(clss, confs):
                if int(c) == 0:
                    best_ben = max(best_ben, float(conf))
                elif int(c) == 1:
                    best_mal = max(best_mal, float(conf))
        rows.append(
            {
                "image": stem,
                "true_cls": true_cls,
                "best_benign": best_ben,
                "best_malignant": best_mal,
            }
        )
    return rows


def decide(best_ben: float, best_mal: float, mal_thr: float, ben_thr: float) -> int | None:
    if best_mal >= mal_thr:
        return 1
    if best_ben >= ben_thr:
        return 0
    return None


def evaluate_rows(rows: list[dict], mal_thr: float, ben_thr: float) -> dict:
    confusion = {
        "benign->benign": 0,
        "benign->malignant": 0,
        "benign->miss": 0,
        "malignant->malignant": 0,
        "malignant->benign": 0,
        "malignant->miss": 0,
    }
    for row in rows:
        true_name = CLASS_NAMES[row["true_cls"]]
        pred = decide(row["best_benign"], row["best_malignant"], mal_thr, ben_thr)
        if pred is None:
            pred_name = "miss"
        else:
            pred_name = CLASS_NAMES[pred]
        confusion[f"{true_name}->{pred_name}"] += 1

    n_ben = sum(1 for r in rows if r["true_cls"] == 0)
    n_mal = sum(1 for r in rows if r["true_cls"] == 1)
    total = n_ben + n_mal
    benign_acc = confusion["benign->benign"] / n_ben if n_ben else 0.0
    mal_acc = confusion["malignant->malignant"] / n_mal if n_mal else 0.0
    overall = (
        (confusion["benign->benign"] + confusion["malignant->malignant"]) / total
        if total
        else 0.0
    )
    mal_to_ben = (
        (confusion["malignant->benign"] + confusion["malignant->miss"]) / n_mal
        if n_mal
        else 0.0
    )
    # Keep gate-compatible malignant->benign as wrong class only (not miss) AND
    # also report miss-inclusive fail rate.
    mal_to_ben_strict = confusion["malignant->benign"] / n_mal if n_mal else 0.0
    ben_to_mal = confusion["benign->malignant"] / n_ben if n_ben else 0.0
    return {
        "n_images": total,
        "n_benign": n_ben,
        "n_malignant": n_mal,
        "confusion": confusion,
        "benign_accuracy": benign_acc,
        "malignant_accuracy": mal_acc,
        "overall_class_accuracy": overall,
        "malignant_to_benign": mal_to_ben_strict,
        "malignant_fail_rate": mal_to_ben,
        "benign_to_malignant": ben_to_mal,
        "mal_thr": mal_thr,
        "ben_thr": ben_thr,
    }


def score_candidate(m: dict) -> tuple:
    """
    Prefer meeting gate constraints on val, then maximize malignant accuracy,
    then overall, then minimize false alarms.
    """
    gate_penalties = 0
    if m["malignant_accuracy"] < GATE["malignant_accuracy_min"]:
        gate_penalties += 1
    if m["malignant_to_benign"] > GATE["malignant_to_benign_max"]:
        gate_penalties += 1
    if m["benign_to_malignant"] > GATE["benign_to_malignant_max"]:
        gate_penalties += 1
    if m["overall_class_accuracy"] < GATE["overall_class_accuracy_min"]:
        gate_penalties += 1

    # Soft distance to unmet targets (lower better when used as secondary)
    soft = 0.0
    soft += max(0.0, GATE["malignant_accuracy_min"] - m["malignant_accuracy"])
    soft += max(0.0, m["malignant_to_benign"] - GATE["malignant_to_benign_max"])
    soft += max(0.0, m["benign_to_malignant"] - GATE["benign_to_malignant_max"])
    soft += max(0.0, GATE["overall_class_accuracy_min"] - m["overall_class_accuracy"])

    # Sort key: fewer gate failures first, then smaller soft distance,
    # then higher mal_acc, higher overall, lower ben->mal
    return (
        gate_penalties,
        soft,
        -m["malignant_accuracy"],
        -m["overall_class_accuracy"],
        m["benign_to_malignant"],
    )


def grid_search(rows: list[dict], mal_grid: list[float], ben_grid: list[float]) -> dict:
    best = None
    best_key = None
    all_ok = []
    for mal_thr in mal_grid:
        for ben_thr in ben_grid:
            metrics = evaluate_rows(rows, mal_thr, ben_thr)
            key = score_candidate(metrics)
            if best is None or key < best_key:
                best = metrics
                best_key = key
            # Track candidates that pass all hard constraints on val
            if key[0] == 0:
                all_ok.append(metrics)
    return {"best": best, "passing_on_val": all_ok, "n_evaluated": len(mal_grid) * len(ben_grid)}


def to_gate_result(metrics: dict) -> dict:
    """Shape expected by generalization_gate()."""
    return {
        "malignant_accuracy": metrics["malignant_accuracy"],
        "overall_class_accuracy": metrics["overall_class_accuracy"],
        "n_benign": metrics["n_benign"],
        "n_malignant": metrics["n_malignant"],
        "confusion": metrics["confusion"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate asymmetric decision on val")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--detect-conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"Missing weights: {weights}")

    model = YOLO(str(weights))
    print("Collecting val scores (raw images)...")
    val_rows = collect_scores(model, load_gt_labels("val"), args.detect_conf, args.iou)
    print(f"val images with raw files: {len(val_rows)}")

    mal_grid = [round(x, 2) for x in np.arange(0.15, 0.56, 0.05)]
    ben_grid = [round(x, 2) for x in np.arange(0.25, 0.81, 0.05)]
    search = grid_search(val_rows, mal_grid, ben_grid)
    best = search["best"]
    assert best is not None

    print(
        f"Best on val: mal_thr={best['mal_thr']} ben_thr={best['ben_thr']} "
        f"mal_acc={best['malignant_accuracy']:.3f} "
        f"mal->ben={best['malignant_to_benign']:.3f} "
        f"ben->mal={best['benign_to_malignant']:.3f} "
        f"overall={best['overall_class_accuracy']:.3f}"
    )
    print(f"passing_on_val={len(search['passing_on_val'])}")

    print("Evaluating on held-out test...")
    test_rows = collect_scores(model, load_gt_labels("test"), args.detect_conf, args.iou)
    test_metrics = evaluate_rows(test_rows, best["mal_thr"], best["ben_thr"])
    # Baseline top-1 style for comparison (mal if best_mal>=best_ben and >=0.25)
    baseline_top = evaluate_rows(test_rows, mal_thr=0.25, ben_thr=0.25)
    # Force classic: pick max of the two if either >=0.25
    classic_rows_metrics = []
    confusion_classic = {
        "benign->benign": 0,
        "benign->malignant": 0,
        "benign->miss": 0,
        "malignant->malignant": 0,
        "malignant->benign": 0,
        "malignant->miss": 0,
    }
    for row in test_rows:
        true_name = CLASS_NAMES[row["true_cls"]]
        bb, bm = row["best_benign"], row["best_malignant"]
        if max(bb, bm) < 0.25:
            pred_name = "miss"
        elif bm >= bb:
            pred_name = "malignant"
        else:
            pred_name = "benign"
        confusion_classic[f"{true_name}->{pred_name}"] += 1
    n_ben = sum(1 for r in test_rows if r["true_cls"] == 0)
    n_mal = sum(1 for r in test_rows if r["true_cls"] == 1)
    classic = {
        "confusion": confusion_classic,
        "malignant_accuracy": confusion_classic["malignant->malignant"] / n_mal,
        "overall_class_accuracy": (
            confusion_classic["benign->benign"] + confusion_classic["malignant->malignant"]
        )
        / (n_ben + n_mal),
        "malignant_to_benign": confusion_classic["malignant->benign"] / n_mal,
        "benign_to_malignant": confusion_classic["benign->malignant"] / n_ben,
        "n_benign": n_ben,
        "n_malignant": n_mal,
    }

    gate_ok, gate_reasons = generalization_gate(to_gate_result(test_metrics), gap=None)
    classic_ok, classic_reasons = generalization_gate(to_gate_result(classic), gap=None)

    def to_builtin(obj):
        if isinstance(obj, dict):
            return {k: to_builtin(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_builtin(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    cfg = {
        "decision": "asymmetric",
        "weights": str(weights),
        "image_source": "data/raw/images",
        "detect_conf": float(args.detect_conf),
        "iou": float(args.iou),
        "mal_thr": float(best["mal_thr"]),
        "ben_thr": float(best["ben_thr"]),
        "rule": (
            "If best malignant confidence >= mal_thr -> malignant; "
            "else if best benign confidence >= ben_thr -> benign; else miss."
        ),
        "calibrated_on": "val",
        "val_metrics": to_builtin(best),
        "test_metrics": to_builtin(test_metrics),
    }

    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    OUT_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    report = {
        "config_path": str(OUT_YAML),
        "val_best": best,
        "test_asymmetric": test_metrics,
        "test_classic_top1": classic,
        "gate_asymmetric": {"pass": gate_ok, "reasons": gate_reasons},
        "gate_classic": {"pass": classic_ok, "reasons": classic_reasons},
        "n_passing_on_val": len(search["passing_on_val"]),
        "search_size": search["n_evaluated"],
        "baseline_asymmetric_equal_thr_0_25": baseline_top,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    GATE_STATUS.parent.mkdir(parents=True, exist_ok=True)
    GATE_STATUS.write_text(
        json.dumps(
            {
                "model": "yolov5su_balanced",
                "decision": "asymmetric",
                "mal_thr": best["mal_thr"],
                "ben_thr": best["ben_thr"],
                "gate_pass": gate_ok,
                "gate_reasons": gate_reasons,
                "test_metrics": {
                    "overall_class_accuracy": test_metrics["overall_class_accuracy"],
                    "malignant_accuracy": test_metrics["malignant_accuracy"],
                    "malignant_to_benign": test_metrics["malignant_to_benign"],
                    "benign_to_malignant": test_metrics["benign_to_malignant"],
                    "confusion": test_metrics["confusion"],
                },
                "classic_gate_pass": classic_ok,
                "classic_gate_reasons": classic_reasons,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {OUT_YAML}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {GATE_STATUS}")
    print(
        f"TEST asymmetric: overall={test_metrics['overall_class_accuracy']:.3f} "
        f"mal_acc={test_metrics['malignant_accuracy']:.3f} "
        f"mal->ben={test_metrics['malignant_to_benign']:.3f} "
        f"ben->mal={test_metrics['benign_to_malignant']:.3f} "
        f"GATE={'PASS' if gate_ok else 'FAIL'}"
    )
    for r in gate_reasons:
        print(f"  - {r}")
    print(
        f"TEST classic: overall={classic['overall_class_accuracy']:.3f} "
        f"mal_acc={classic['malignant_accuracy']:.3f} "
        f"mal->ben={classic['malignant_to_benign']:.3f} "
        f"ben->mal={classic['benign_to_malignant']:.3f} "
        f"GATE={'PASS' if classic_ok else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
