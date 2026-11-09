"""Build the image-level dev/test split and write the summary CSV (Phase 1).

    python scripts/build_splits.py

Writes DATA_DIR/splits.json and RESULTS_DIR/splits_summary.csv. The split file
is committed, so this normally runs once. Re-running with the same config gives
an identical file (a unit test checks this).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlmsyc.data import load_all_pope  # noqa: E402
from vlmsyc.paths import data_dir, load_yaml, results_dir  # noqa: E402
from vlmsyc.splits import build_splits, save_splits, summary_rows, write_summary_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    questions = load_all_pope(data_dir() / cfg["pope_dir"], data_dir() / cfg["image_dir"])
    splits = build_splits(questions, seed=cfg["seed"], dev_fraction=cfg["dev_fraction"], n_images=cfg["n_images"])

    splits_path = data_dir() / cfg["splits_file"]
    save_splits(splits, splits_path)
    rows = summary_rows(splits, questions)
    summary_path = results_dir() / "splits_summary.csv"
    write_summary_csv(rows, summary_path)

    print(f"wrote {splits_path}\nwrote {summary_path}")
    for row in rows:
        print(row)
    if rows[-1]["n_images"] != 0:
        raise SystemExit("ERROR: an image is in both dev and test")


if __name__ == "__main__":
    main()
