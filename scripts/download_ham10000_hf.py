"""
Download HAM10000 images+masks from Hugging Face GenSegDataset mirror,
then merge with Dataverse metadata already in data/raw/metadata/.

Faster CDN fallback when Harvard Dataverse is slow.

Usage:
  python scripts/download_ham10000_hf.py
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DOWNLOAD_DIR = RAW_DIR / "_download"
IMAGES_DIR = RAW_DIR / "images"
MASKS_DIR = RAW_DIR / "masks"
META_PATH = RAW_DIR / "metadata" / "HAM10000_metadata.csv"

HF_URL = "https://huggingface.co/datasets/MaybeRichard/GenSegDataset/resolve/main/ham10000.tar"
TAR_NAME = "ham10000.tar"


def find_aria2() -> str | None:
    local = (
        PROJECT_ROOT
        / "tools"
        / "aria2"
        / "aria2-1.37.0-win-64bit-build1"
        / "aria2c.exe"
    )
    if local.is_file():
        return str(local)
    return shutil.which("aria2c")


def download_tar() -> Path:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = DOWNLOAD_DIR / TAR_NAME
    ctrl = Path(str(dest) + ".aria2")
    if dest.exists() and dest.stat().st_size > 100_000_000 and not ctrl.exists():
        print(f"Using existing {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
        return dest

    aria = find_aria2()
    if not aria:
        raise SystemExit("aria2c not found. Place it under tools/aria2/ or install aria2.")

    print(f"Downloading {HF_URL}")
    cmd = [
        aria,
        "-c",
        "-x",
        "16",
        "-s",
        "16",
        "-k",
        "1M",
        "--file-allocation=none",
        "--max-tries=0",
        "--retry-wait=3",
        "--user-agent=Mozilla/5.0",
        "-d",
        str(DOWNLOAD_DIR),
        "-o",
        TAR_NAME,
        HF_URL,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0 or not dest.exists():
        raise SystemExit("Hugging Face download failed")
    return dest


def extract_and_collect(tar_path: Path) -> tuple[int, int]:
    extract_root = DOWNLOAD_DIR / "ham10000_hf"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {tar_path.name} ...")
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(extract_root)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    n_img = 0
    n_mask = 0
    for img in extract_root.rglob("*"):
        if not img.is_file():
            continue
        parts = {p.lower() for p in img.parts}
        name = img.name
        # Prefer files under images/ or masks/
        if "masks" in parts or "mask" in parts:
            # Normalize to ISIC_xxxx_segmentation.png when possible
            stem = img.stem
            if stem.endswith("_segmentation"):
                out_name = f"{stem}.png" if img.suffix.lower() == ".png" else name
            else:
                out_name = f"{stem}_segmentation.png"
            dst = MASKS_DIR / out_name
            if not dst.exists():
                shutil.copy2(img, dst)
            n_mask += 1
        elif "images" in parts or img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            # Skip if this path looks like a mask
            if "mask" in img.name.lower() or "seg" in img.name.lower():
                continue
            dst = IMAGES_DIR / name
            # Convert png->keep; prepare script accepts jpg/png
            if not dst.exists():
                shutil.copy2(img, dst)
            n_img += 1
    return n_img, n_mask


def main() -> None:
    if not META_PATH.is_file():
        raise SystemExit(
            f"Missing {META_PATH}. Download metadata first:\n"
            "  python scripts/download_ham10000.py --source dataverse\n"
            "(metadata+masks are small; images can come from this HF mirror)"
        )

    tar_path = download_tar()
    n_img, n_mask = extract_and_collect(tar_path)
    print("\n=== HF mirror summary ===")
    print(f"Images : {n_img} -> {IMAGES_DIR}")
    print(f"Masks  : {n_mask} -> {MASKS_DIR}")
    print(f"Metadata: {META_PATH}")
    if n_img < 1000:
        raise SystemExit("Too few images extracted from HF archive")
    print("OK: raw images ready. Run: python scripts/prepare_yolo_dataset.py")


if __name__ == "__main__":
    main()
