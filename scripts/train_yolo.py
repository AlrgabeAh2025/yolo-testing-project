"""
Train YOLO models (v8 / v5 / v3 via Ultralytics) on the shared HAM10000 binary dataset.

Usage (from project root):
  python scripts/train_yolo.py --model yolov8s
  python scripts/train_yolo.py --model yolov5su
  python scripts/train_yolo.py --model yolov3u
  python scripts/train_yolo.py --model yolov8s --eval-only runs/detect/yolov8s/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def _install_safe_torch_save() -> None:
    """Avoid zipfile 'closed file' crashes on large Windows checkpoints."""
    original = torch.save

    def safe_save(obj, f, *args, **kwargs):
        kwargs.setdefault("_use_new_zipfile_serialization", False)
        target = os.fspath(f) if not hasattr(f, "write") else None
        if target is None:
            return original(obj, f, *args, **kwargs)
        tmp = f"{target}.tmp"
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                original(obj, tmp, *args, **kwargs)
                os.replace(tmp, target)
                return
            except Exception as exc:  # noqa: BLE001 - retry transient IO failures
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
        raise last_err  # type: ignore[misc]

    torch.save = safe_save  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "configs" / "data.yaml"
DEFAULT_TRAIN_YAML = PROJECT_ROOT / "configs" / "train.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
COMPARE_PATH = PROJECT_ROOT / "runs" / "compare_summary.json"

MODEL_WEIGHTS = {
    "yolov8s": "yolov8s.pt",
    "yolov5su": "yolov5su.pt",
    # Full yolov3u (~104M params / ~1GB ckpt) OOMs or MemoryErrors on RTX 3060 12GB
    # when saving checkpoints; yolov3-tinyu is the practical YOLOv3 for this box.
    "yolov3-tinyu": "yolov3-tinyu.pt",
    "yolov3u": "yolov3u.pt",
}


def load_train_cfg(config_path: Path | None = None) -> dict:
    path = config_path or DEFAULT_TRAIN_YAML
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def metrics_to_dict(metrics) -> dict:
    """Extract key detection metrics including per-class precision/recall."""
    box = metrics.box
    names = metrics.names if hasattr(metrics, "names") else {0: "benign", 1: "malignant"}
    per_class = {}
    for i, name in names.items():
        try:
            per_class[name] = {
                "precision": float(box.p[i]) if box.p is not None else None,
                "recall": float(box.r[i]) if box.r is not None else None,
                "map50": float(box.ap50[i]) if box.ap50 is not None else None,
                "map50_95": float(box.ap[i]) if box.ap is not None else None,
            }
        except Exception:
            per_class[name] = {
                "precision": None,
                "recall": None,
                "map50": None,
                "map50_95": None,
            }

    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class": per_class,
        "malignant_recall": per_class.get("malignant", {}).get("recall"),
        "benign_recall": per_class.get("benign", {}).get("recall"),
    }


def evaluate(
    model: YOLO,
    split: str,
    run_name: str,
    train_cfg: dict,
    data_yaml: Path | None = None,
) -> dict:
    metrics = model.val(
        data=str(data_yaml or DATA_YAML),
        split=split,
        imgsz=train_cfg.get("imgsz", 640),
        batch=train_cfg.get("batch", 16),
        device=train_cfg.get("device", 0),
        workers=train_cfg.get("workers", 4),
        plots=True,
        project=str(RUNS_DIR),
        name=f"{run_name}_{split}_eval",
        exist_ok=True,
    )
    result = metrics_to_dict(metrics)
    print(f"\n=== {run_name} [{split}] ===")
    print(json.dumps(result, indent=2))
    return result


def success_gate(val_metrics: dict) -> tuple[bool, str]:
    map50 = val_metrics.get("map50")
    mal_rec = val_metrics.get("malignant_recall")
    if map50 is None:
        return False, "missing map50"
    if map50 < 0.25:
        return False, f"map50 too low ({map50:.4f})"
    if mal_rec is None:
        return False, "missing malignant recall"
    if mal_rec < 0.05:
        return False, f"malignant recall collapsed ({mal_rec:.4f})"
    return True, "ok"


def update_compare_summary(run_name: str, payload: dict) -> None:
    COMPARE_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    if COMPARE_PATH.is_file():
        summary = json.loads(COMPARE_PATH.read_text(encoding="utf-8"))
    summary[run_name] = payload
    COMPARE_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Updated {COMPARE_PATH}")


def train_one(
    model_key: str,
    train_cfg: dict,
    run_name: str,
    weights_override: str | None = None,
    data_yaml: Path | None = None,
) -> Path:
    if model_key not in MODEL_WEIGHTS:
        raise SystemExit(f"Unknown model '{model_key}'. Choose from {list(MODEL_WEIGHTS)}")
    data_path = data_yaml or DATA_YAML
    if not data_path.is_file():
        raise SystemExit(f"Missing {data_path}. Run prepare_yolo_dataset.py first.")

    weights = weights_override or MODEL_WEIGHTS[model_key]
    model = YOLO(weights)

    train_kwargs = {
        "data": str(data_path),
        "project": str(RUNS_DIR),
        "name": run_name,
        "exist_ok": True,
    }
    # Pass all shared hyperparams from train config yaml
    for key, value in train_cfg.items():
        train_kwargs[key] = value

    print(f"\n>>> Training {model_key} ({weights}) run_name={run_name}")
    model.train(**train_kwargs)

    best = RUNS_DIR / run_name / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"Training finished but best.pt missing: {best}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/eval YOLO on HAM10000 binary set")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_WEIGHTS.keys()),
        help="Model key to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_TRAIN_YAML),
        help="Path to training hyperparameters yaml",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run directory name under runs/detect (default: model key)",
    )
    parser.add_argument(
        "--eval-only",
        type=str,
        default=None,
        help="Path to weights for evaluation only (skip training)",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test-split evaluation",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override batch size from train config (useful for larger models)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from runs/detect/<name>/weights/last.pt",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override dataloader workers (use 0 to avoid host RAM crashes)",
    )
    parser.add_argument(
        "--mixup",
        type=float,
        default=None,
        help="Override mixup probability",
    )
    parser.add_argument(
        "--copy-paste",
        type=float,
        default=None,
        dest="copy_paste",
        help="Override copy_paste probability",
    )
    parser.add_argument(
        "--mosaic",
        type=float,
        default=None,
        help="Override mosaic probability (0 disables mosaic)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs from train config",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional starting weights path (fine-tune instead of COCO pretrained)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional dataset yaml override (default: configs/data.yaml)",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"Missing train config: {config_path}")
    train_cfg = load_train_cfg(config_path)
    run_name = args.name or args.model
    data_yaml = Path(args.data) if args.data else DATA_YAML
    print(f"Using train config: {config_path}")
    print(f"Using data yaml: {data_yaml}")
    if args.batch is not None:
        train_cfg["batch"] = args.batch
        print(f"Using override batch={args.batch}")
    if args.workers is not None:
        train_cfg["workers"] = args.workers
        print(f"Using override workers={args.workers}")
    if args.mixup is not None:
        train_cfg["mixup"] = args.mixup
        print(f"Using override mixup={args.mixup}")
    if args.copy_paste is not None:
        train_cfg["copy_paste"] = args.copy_paste
        print(f"Using override copy_paste={args.copy_paste}")
    if args.mosaic is not None:
        train_cfg["mosaic"] = args.mosaic
        if args.mosaic <= 0:
            train_cfg["close_mosaic"] = 0
        print(f"Using override mosaic={args.mosaic}")
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
        print(f"Using override epochs={args.epochs}")

    if args.eval_only:
        best = Path(args.eval_only)
        if not best.is_file():
            raise SystemExit(f"Weights not found: {best}")
        model = YOLO(str(best))
    elif args.resume:
        last = RUNS_DIR / run_name / "weights" / "last.pt"
        if not last.is_file():
            raise SystemExit(f"Cannot resume; missing {last}")
        print(f"\n>>> Resuming {run_name} from {last}")
        model = YOLO(str(last))
        # Keep memory-safe overrides even under resume (Ultralytics restores most args).
        resume_kwargs = {"resume": True}
        for key in ("workers", "mixup", "copy_paste", "batch", "mosaic", "close_mosaic"):
            if key in train_cfg:
                resume_kwargs[key] = train_cfg[key]
        # Extra CLI/runtime overrides already applied into train_cfg above.
        model.train(**resume_kwargs)
        best = RUNS_DIR / run_name / "weights" / "best.pt"
        if not best.is_file():
            raise SystemExit(f"Resume finished but best.pt missing: {best}")
        model = YOLO(str(best))
    else:
        best = train_one(
            args.model,
            train_cfg,
            run_name,
            weights_override=args.weights,
            data_yaml=data_yaml,
        )
        model = YOLO(str(best))

    val_metrics = evaluate(model, "val", run_name, train_cfg, data_yaml=data_yaml)
    ok, reason = success_gate(val_metrics)
    test_metrics = None
    if not args.skip_test:
        test_metrics = evaluate(model, "test", run_name, train_cfg, data_yaml=data_yaml)

    payload = {
        "model": args.model,
        "run_name": run_name,
        "train_config": str(config_path),
        "weights": str(best),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "success": ok,
        "gate_reason": reason,
        "balanced": "balanced" in run_name.lower() or "balanced" in config_path.name.lower(),
        "val": val_metrics,
        "test": test_metrics,
    }
    metrics_path = RUNS_DIR / run_name / "eval_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    update_compare_summary(run_name, payload)

    if not ok:
        raise SystemExit(f"FAIL gate for {run_name}: {reason}")
    print(f"\nPASS gate for {run_name}: {reason}")
    print(f"best weights: {best}")


if __name__ == "__main__":
    _install_safe_torch_save()
    main()
