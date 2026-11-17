"""
Download HAM10000 JPEG images from the public ISIC Archive API.

Use when Dataverse/HF bulk archives are too slow. Masks + metadata should
already exist under data/raw/.

Usage:
  python scripts/download_images_isic.py              # all images
  python scripts/download_images_isic.py --limit 500  # smoke subset
  python scripts/download_images_isic.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
META = PROJECT_ROOT / "data" / "raw" / "metadata" / "HAM10000_metadata.csv"
IMAGES = PROJECT_ROOT / "data" / "raw" / "images"
UA = "yolo-testing-project/1.0 (research; HAM10000 prep)"


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def download_one(image_id: str, retries: int = 4) -> tuple[str, str]:
    dest = IMAGES / f"{image_id}.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        return image_id, "exists"
    last_err = ""
    for attempt in range(retries):
        try:
            meta = fetch_json(f"https://api.isic-archive.com/api/v2/images/{image_id}/")
            url = meta["files"]["full"]["url"]
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            tmp = dest.with_suffix(".partial")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return image_id, "ok"
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            import time

            time.sleep(0.5 * (2**attempt) + random.random())
    return image_id, f"err:{last_err}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Optional max images")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="When --limit set, sample stratified by dx",
    )
    args = parser.parse_args()

    if not META.is_file():
        raise SystemExit(f"Missing metadata: {META}")
    IMAGES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(META).drop_duplicates("image_id")
    if args.limit and args.limit < len(df):
        if args.stratified and "dx" in df.columns:
            parts = []
            rng = random.Random(args.seed)
            # approximate stratified sample
            groups = list(df.groupby("dx"))
            per = max(1, args.limit // max(1, len(groups)))
            for _, g in groups:
                take = min(len(g), per)
                parts.append(g.sample(n=take, random_state=args.seed))
            df = pd.concat(parts).drop_duplicates("image_id")
            if len(df) > args.limit:
                df = df.sample(n=args.limit, random_state=args.seed)
            # fill remainder randomly if needed
            if len(df) < args.limit:
                rest = pd.read_csv(META).drop_duplicates("image_id")
                rest = rest[~rest["image_id"].isin(df["image_id"])]
                need = args.limit - len(df)
                df = pd.concat([df, rest.sample(n=min(need, len(rest)), random_state=args.seed)])
        else:
            df = df.sample(n=args.limit, random_state=args.seed)

    ids = df["image_id"].astype(str).tolist()
    print(f"Downloading {len(ids)} images to {IMAGES} with {args.workers} workers")

    ok = exists = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, i): i for i in ids}
        for fut in tqdm(as_completed(futs), total=len(futs)):
            _, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "exists":
                exists += 1
            else:
                err += 1
                tqdm.write(str(status))

    print(f"done ok={ok} exists={exists} err={err}")


if __name__ == "__main__":
    main()
