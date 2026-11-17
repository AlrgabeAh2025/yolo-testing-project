"""
Download HAM10000 images, metadata, and lesion segmentation masks.

Sources (tried in order for --source auto):
  1) Harvard Dataverse via curl (no API key) — default
  2) Kaggle API — if %USERPROFILE%\\.kaggle\\kaggle.json exists

Usage (from project root):
  python scripts/download_ham10000.py
  python scripts/download_ham10000.py --source dataverse
  python scripts/download_ham10000.py --source kaggle
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMAGES_DIR = RAW_DIR / "images"
MASKS_DIR = RAW_DIR / "masks"
META_DIR = RAW_DIR / "metadata"
DOWNLOAD_DIR = RAW_DIR / "_download"

HAM_KAGGLE = "kmader/skin-cancer-mnist-ham10000"
MASKS_KAGGLE = "tschandl/ham10000-lesion-segmentations"

# Harvard Dataverse DOI 10.7910/DVN/DBW86T
DATAVERSE_FILES = {
    "HAM10000_images_part_1.zip": 3172585,
    "HAM10000_images_part_2.zip": 3172584,
    "HAM10000_metadata.tab": 4338392,
    "HAM10000_segmentations_lesion_tschandl.zip": 3838943,
}


def _kaggle_creds_path() -> Path:
    return Path.home() / ".kaggle" / "kaggle.json"


def _find_aria2() -> str | None:
    local = (
        PROJECT_ROOT
        / "tools"
        / "aria2"
        / "aria2-1.37.0-win-64bit-build1"
        / "aria2c.exe"
    )
    if local.is_file():
        return str(local)
    return shutil.which("aria2c") or shutil.which("aria2c.exe")


def download_file(url: str, dest: Path) -> None:
    """Download with aria2 (multi-connection) when available, else curl."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Skip only if a valid non-empty zip/file already exists and is not a control file
    ctrl = Path(str(dest) + ".aria2")
    if dest.exists() and dest.stat().st_size > 0 and not ctrl.exists():
        # For zips, quick integrity probe
        if dest.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(dest, "r") as zf:
                    if zf.testzip() is None:
                        print(
                            f"Skip existing: {dest.name} "
                            f"({dest.stat().st_size / 1024 / 1024:.1f} MB)"
                        )
                        return
            except zipfile.BadZipFile:
                print(f"Incomplete/corrupt zip, resuming: {dest.name}")
        else:
            print(f"Skip existing: {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
            return

    print(f"Downloading {dest.name} ...")
    aria = _find_aria2()
    if aria:
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
            f"--user-agent=Mozilla/5.0 (compatible; yolo-testing-project/1.0)",
            "-d",
            str(dest.parent),
            "-o",
            dest.name,
            url,
        ]
    else:
        curl = shutil.which("curl") or shutil.which("curl.exe")
        if not curl:
            raise RuntimeError("Neither aria2c nor curl found on PATH")
        cmd = [
            curl,
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "3",
            "-A",
            "Mozilla/5.0 (compatible; yolo-testing-project/1.0; research)",
            "-C",
            "-",
            "-o",
            str(dest),
            url,
        ]

    result = subprocess.run(cmd)
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"Failed to download {url} -> {dest}")
    if dest.suffix.lower() == ".zip":
        with zipfile.ZipFile(dest, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f"Corrupt zip member after download: {bad}")


def unzip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def collect_images(search_roots: list[Path]) -> int:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    seen: set[str] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for src in root.rglob("*"):
            if not src.is_file():
                continue
            if src.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lower = str(src).lower()
            if "segment" in lower or "mask" in lower:
                continue
            if src.name in seen:
                continue
            dst = IMAGES_DIR / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            seen.add(src.name)
            count += 1
    return count


def collect_masks(search_roots: list[Path]) -> int:
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for root in search_roots:
        if not root.exists():
            continue
        for src in root.rglob("*.png"):
            dst = MASKS_DIR / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            count += 1
    return count


def normalize_metadata(src: Path) -> Path:
    META_DIR.mkdir(parents=True, exist_ok=True)
    dst = META_DIR / "HAM10000_metadata.csv"

    # Detect delimiter from first line
    sample = src.read_text(encoding="utf-8", errors="replace").splitlines()[:5]
    delim = "\t" if sample and sample[0].count("\t") >= sample[0].count(",") else ","

    with src.open("r", encoding="utf-8", errors="replace", newline="") as fin:
        reader = csv.DictReader(fin, delimiter=delim)
        if not reader.fieldnames:
            raise RuntimeError(f"Could not read header from {src}")
        fieldnames = [name.strip().strip('"') for name in reader.fieldnames]
        rows = []
        for row in reader:
            cleaned = {
                (k or "").strip().strip('"'): (v or "").strip().strip('"')
                for k, v in row.items()
            }
            # Remap keys if quotes remained in fieldnames
            remapped = {}
            for fk, fv in cleaned.items():
                key = fk.strip().strip('"')
                remapped[key] = fv
            rows.append({name: remapped.get(name, "") for name in fieldnames})

    with dst.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return dst


def download_from_dataverse(skip_images: bool = False) -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Smaller files first so we fail fast if network is broken
    ordered = [
        "HAM10000_metadata.tab",
        "HAM10000_segmentations_lesion_tschandl.zip",
    ]
    if not skip_images:
        ordered.extend(
            [
                "HAM10000_images_part_1.zip",
                "HAM10000_images_part_2.zip",
            ]
        )
    for filename in ordered:
        file_id = DATAVERSE_FILES[filename]
        url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
        download_file(url, DOWNLOAD_DIR / filename)

    masks = DOWNLOAD_DIR / "HAM10000_segmentations"
    unzip(DOWNLOAD_DIR / "HAM10000_segmentations_lesion_tschandl.zip", masks)
    n_masks = collect_masks([masks])
    meta = normalize_metadata(DOWNLOAD_DIR / "HAM10000_metadata.tab")

    if skip_images:
        n_images = len(list(IMAGES_DIR.glob("*"))) if IMAGES_DIR.exists() else 0
        print("Skipped Dataverse image zips (use scripts/download_ham10000_hf.py for images).")
        print(f"Existing images: {n_images}")
        print(f"Masks copied  : {n_masks} -> {MASKS_DIR}")
        print(f"Metadata      : {meta}")
        return

    part1 = DOWNLOAD_DIR / "HAM10000_images_part_1"
    part2 = DOWNLOAD_DIR / "HAM10000_images_part_2"
    unzip(DOWNLOAD_DIR / "HAM10000_images_part_1.zip", part1)
    unzip(DOWNLOAD_DIR / "HAM10000_images_part_2.zip", part2)
    n_images = collect_images([part1, part2])
    _print_summary(n_images, n_masks, meta)


def run_kaggle_download(dataset: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(dest),
        "--unzip",
    ]
    print(f"Downloading {dataset} -> {dest}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        if result.stdout.strip():
            print(result.stdout.strip())
        return

    cmd_no_unzip = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(dest),
    ]
    result2 = subprocess.run(cmd_no_unzip, capture_output=True, text=True)
    if result2.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed for {dataset}.\n"
            f"stderr:\n{result.stderr}\n{result2.stderr}\n"
            "Accept dataset terms in the browser, or use --source dataverse."
        )
    for zpath in dest.glob("*.zip"):
        unzip(zpath, dest)
        zpath.unlink(missing_ok=True)


def download_from_kaggle() -> None:
    if not _kaggle_creds_path().is_file():
        raise SystemExit(
            f"Missing Kaggle credentials at {_kaggle_creds_path()}\n"
            "Use: python scripts/download_ham10000.py --source dataverse"
        )
    ham_dl = DOWNLOAD_DIR / "kaggle_ham"
    mask_dl = DOWNLOAD_DIR / "kaggle_masks"
    run_kaggle_download(HAM_KAGGLE, ham_dl)
    run_kaggle_download(MASKS_KAGGLE, mask_dl)

    n_images = collect_images([ham_dl])
    n_masks = collect_masks([mask_dl])
    meta_candidates = list(ham_dl.rglob("HAM10000_metadata.csv")) + list(
        ham_dl.rglob("*metadata*.csv")
    )
    if not meta_candidates:
        raise SystemExit("Could not find HAM10000 metadata CSV in Kaggle download.")
    meta = normalize_metadata(meta_candidates[0])
    _print_summary(n_images, n_masks, meta)


def _print_summary(n_images: int, n_masks: int, meta: Path) -> None:
    print("\n=== Download summary ===")
    print(f"Images copied : {n_images} -> {IMAGES_DIR}")
    print(f"Masks copied  : {n_masks} -> {MASKS_DIR}")
    print(f"Metadata      : {meta}")
    if n_images < 1000 or n_masks < 1000:
        raise SystemExit("Download incomplete — expected ~10k images and masks.")
    print("OK: raw data is ready under data/raw/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HAM10000 for YOLO prep")
    parser.add_argument(
        "--source",
        choices=("auto", "dataverse", "kaggle"),
        default="auto",
        help="Download source (default: auto)",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Dataverse only: download metadata+masks (use HF mirror for images)",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    source = args.source
    if source == "auto":
        # Prefer Dataverse: no login, includes official masks
        source = "dataverse"
        print(f"Auto-selected source: {source}")

    if source == "dataverse":
        download_from_dataverse(skip_images=args.skip_images)
    else:
        download_from_kaggle()


if __name__ == "__main__":
    main()
