"""Download POPE (pinned commit) and only the COCO val2014 images POPE references.

Safe to re-run: files already present are skipped. Every image is checked by
decoding it with PIL, and a partial download never leaves a broken file behind.

    python scripts/download_data.py            # POPE json + ~500 images (~80 MB)
    python scripts/download_data.py --pope-only
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlmsyc.data import POPE_FILE, POPE_SPLITS, load_all_pope  # noqa: E402
from vlmsyc.paths import data_dir, load_yaml  # noqa: E402

HEADERS = {"User-Agent": "vlmsyc-data-downloader"}


def fetch(url: str, dest: Path, check_image: bool = False) -> str:
    """Download url to dest via a temporary file. Returns 'skipped' or 'downloaded'."""
    if dest.exists() and dest.stat().st_size > 0:
        return "skipped"
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
                tmp.write_bytes(r.read())
            if check_image:
                from PIL import Image

                with Image.open(tmp) as im:
                    im.load()  # fails on truncated or non-image data
            tmp.rename(dest)
            return "downloaded"
        except Exception as err:
            last_error = err
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Failed to download {url}: {last_error!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--pope-only", action="store_true", help="skip the COCO images")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    pope_dir = data_dir() / cfg["pope_dir"]
    image_dir = data_dir() / cfg["image_dir"]
    pope_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    for split in POPE_SPLITS:
        url = cfg["pope_url"].format(commit=cfg["pope_commit"], split=split)
        status = fetch(url, pope_dir / POPE_FILE.format(split=split))
        print(f"POPE {split}: {status}")

    # Loading also validates every row of every file.
    questions = load_all_pope(pope_dir, image_dir)
    files = sorted({q.image_file for qs in questions.values() for q in qs})
    print(f"POPE: {sum(map(len, questions.values()))} questions, {len(files)} distinct images")
    if args.pope_only:
        return

    def get(file: str) -> str:
        return fetch(cfg["coco_url"].format(file=file), image_dir / file, check_image=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        statuses = list(pool.map(get, files))
    print(
        f"COCO images: {statuses.count('downloaded')} downloaded, "
        f"{statuses.count('skipped')} already present, in {image_dir}"
    )


if __name__ == "__main__":
    main()
