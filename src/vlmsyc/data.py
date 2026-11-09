"""Load POPE questions (COCO subset) and their images.

POPE format, checked on the files at commit 109985c (see SPEC.md, "Changes"):
each file is JSON Lines, one object per line with exactly these keys:
    {"question_id": 1, "image": "COCO_val2014_000000310196.jpg",
     "text": "Is there a snowboard in the image?", "label": "yes"}
There is no object field, so we parse it from the question text.
question_id restarts at 1 in every file, so we build a unique id
uid = "<pope_split>:<question_id>".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

POPE_SPLITS = ("random", "popular", "adversarial")
POPE_FILE = "coco_pope_{split}.json"
POPE_KEYS = {"question_id", "image", "text", "label"}

# Every POPE question has this exact form; we rely on it to get the object.
_QUESTION_RE = re.compile(r"^Is there (?:a|an) (.+) in the image\?$")
_IMAGE_RE = re.compile(r"^COCO_val2014_(\d{12})\.jpg$")


@dataclass(frozen=True)
class Question:
    """One POPE yes/no question about one image."""

    uid: str  # "<pope_split>:<question_id>", unique across the three files
    question_id: int  # POPE's own id (only unique within one file)
    pope_split: str  # random, popular or adversarial
    image_id: int  # COCO image id, used for the dev/test split and clustering
    image_file: str  # e.g. COCO_val2014_000000310196.jpg
    image_path: str  # where the image is on disk
    object: str  # the object asked about, e.g. "snowboard"
    text: str  # the original POPE question text
    label: str  # ground truth, "yes" or "no"


def parse_object(text: str) -> str:
    """Return the object in 'Is there a/an <object> in the image?'."""
    match = _QUESTION_RE.match(text)
    if not match:
        raise ValueError(f"Unexpected POPE question format: {text!r}")
    return match.group(1)


def image_id_from_filename(filename: str) -> int:
    """'COCO_val2014_000000310196.jpg' -> 310196."""
    match = _IMAGE_RE.match(filename)
    if not match:
        raise ValueError(f"Unexpected COCO filename: {filename!r}")
    return int(match.group(1))


def load_pope(split_name: str, pope_dir: str | Path, image_dir: str | Path) -> list[Question]:
    """Load one POPE file and check every row, failing loudly on anything unexpected."""
    if split_name not in POPE_SPLITS:
        raise ValueError(f"Unknown POPE split {split_name!r}; expected one of {POPE_SPLITS}")
    path = Path(pope_dir) / POPE_FILE.format(split=split_name)
    questions = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != POPE_KEYS:
                raise ValueError(f"{path}:{line_no}: keys {sorted(row)} != {sorted(POPE_KEYS)}")
            if row["label"] not in ("yes", "no"):
                raise ValueError(f"{path}:{line_no}: label {row['label']!r}")
            questions.append(
                Question(
                    uid=f"{split_name}:{row['question_id']}",
                    question_id=int(row["question_id"]),
                    pope_split=split_name,
                    image_id=image_id_from_filename(row["image"]),
                    image_file=row["image"],
                    image_path=str(Path(image_dir) / row["image"]),
                    object=parse_object(row["text"]),
                    text=row["text"],
                    label=row["label"],
                )
            )
    uids = [q.uid for q in questions]
    if len(uids) != len(set(uids)):
        raise ValueError(f"{path}: duplicate question_id")
    return questions


def load_all_pope(pope_dir: str | Path, image_dir: str | Path) -> dict[str, list[Question]]:
    """All three POPE splits, keyed by split name."""
    return {s: load_pope(s, pope_dir, image_dir) for s in POPE_SPLITS}


def load_image(path: str | Path, max_side: int | None = None):
    """Open an image as RGB, shrinking it so its longer side is at most max_side.

    Shrinking keeps the number of image tokens (and so memory and time) bounded
    on a 16 GB T4. The aspect ratio is kept.
    """
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
    return image
