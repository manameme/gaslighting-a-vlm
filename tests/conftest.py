"""Shared test helpers: tiny synthetic POPE files with the real format.

The fake data copies the structure we found in the real files: JSON Lines with
question_id/image/text/label, the same images in every split, 3 yes + 3 no
questions per image per split, yes questions identical across splits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlmsyc.data import POPE_FILE, POPE_SPLITS

OBJECTS = ["dog", "apple", "car", "umbrella", "chair", "oven", "cup", "bus", "kite", "bed"]


def write_fake_pope(pope_dir: Path, n_images: int = 20) -> None:
    """Write three POPE-format files with n_images images each."""
    pope_dir.mkdir(parents=True, exist_ok=True)
    for s_idx, split in enumerate(POPE_SPLITS):
        lines, qid = [], 0
        for i in range(n_images):
            image = f"COCO_val2014_{100000 + i:012d}.jpg"
            yes_objs = [OBJECTS[(i + k) % 10] for k in range(3)]
            no_objs = [OBJECTS[(i + 3 + s_idx + k) % 10] + f"{s_idx}" for k in range(3)]
            for obj, label in [(o, "yes") for o in yes_objs] + [(o, "no") for o in no_objs]:
                qid += 1
                art = "an" if obj[0] in "aeiou" else "a"
                lines.append(
                    json.dumps(
                        {"question_id": qid, "image": image, "text": f"Is there {art} {obj} in the image?", "label": label}
                    )
                )
        (pope_dir / POPE_FILE.format(split=split)).write_text("\n".join(lines) + "\n")


@pytest.fixture
def fake_pope(tmp_path: Path) -> tuple[Path, Path]:
    """(pope_dir, image_dir) with fake POPE files; images are not created."""
    pope_dir, image_dir = tmp_path / "pope", tmp_path / "images"
    write_fake_pope(pope_dir)
    return pope_dir, image_dir
