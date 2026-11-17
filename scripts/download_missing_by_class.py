"""Download missing images for underrepresented binary classes (malignant first)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from download_images_isic import IMAGES, META, download_one
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

MALIGNANT = {"mel", "bcc", "akiec"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-malignant", type=int, default=400)
    parser.add_argument("--target-benign", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    df = pd.read_csv(META).drop_duplicates("image_id")
    df["dx"] = df["dx"].astype(str).str.lower()
    df["is_mal"] = df["dx"].isin(MALIGNANT)

    have = {p.stem for p in IMAGES.glob("*.jpg")}
    mal = df[df["is_mal"]].copy()
    ben = df[~df["is_mal"]].copy()

    have_mal = sum(1 for i in mal["image_id"] if i in have)
    have_ben = sum(1 for i in ben["image_id"] if i in have)
    print(f"have malignant={have_mal} benign={have_ben}")

    need_mal = max(0, args.target_malignant - have_mal)
    need_ben = max(0, args.target_benign - have_ben)
    todo = []
    todo += [i for i in mal["image_id"] if i not in have][:need_mal]
    todo += [i for i in ben["image_id"] if i not in have][:need_ben]
    print(f"queue={len(todo)} (mal_need={need_mal}, ben_need={need_ben})")

    ok = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(download_one, i) for i in todo]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            _, status = fut.result()
            if status in {"ok", "exists"}:
                ok += 1
            else:
                err += 1
    print(f"done ok={ok} err={err}")


if __name__ == "__main__":
    main()
