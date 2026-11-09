"""Image-level dev/test split, the subsample, and the test-split lock.

Design (SPEC.md "Changes"): all three POPE files use the same 500 images, so
splitting by image id automatically stratifies by POPE split. Each image gives
exactly one question per POPE split, with yes/no balanced within every
(partition, POPE split) cell. An image never gets the same yes question in two
POPE splits (the yes questions are identical across files), so no question is
computed twice.

Test discipline (CLAUDE.md rule 3): load_questions(partition="test") raises
unless ALLOW_TEST=1 is set, and only scripts/run_final.py --final sets it.
"""

from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from .data import POPE_SPLITS, Question, load_all_pope

PARTITIONS = ("dev", "test")


class TestSplitLockedError(RuntimeError):
    """Raised when code other than run_final.py --final tries to read the test split."""

    __test__ = False  # tells pytest this is not a test class despite the name


def build_splits(
    questions_by_split: dict[str, list[Question]],
    seed: int = 0,
    dev_fraction: float = 0.4,
    n_images: int | None = None,
) -> dict:
    """Assign images to dev/test and pick one question per image per POPE split.

    Every random choice comes from one random.Random(seed), and inputs are
    sorted first, so the result is the same on every machine.
    """
    rng = random.Random(seed)

    # 1. Images. Our design needs every POPE split to use the same images.
    image_sets = {s: {q.image_id for q in qs} for s, qs in questions_by_split.items()}
    images = sorted(image_sets[POPE_SPLITS[0]])
    if any(image_sets[s] != set(images) for s in POPE_SPLITS):
        raise ValueError("POPE splits do not share one image set; the subsample design assumes they do")

    rng.shuffle(images)
    if n_images is not None:
        images = images[:n_images]
    n_dev = round(len(images) * dev_fraction)
    partition_images = {"dev": sorted(images[:n_dev]), "test": sorted(images[n_dev:])}
    assert not set(partition_images["dev"]) & set(partition_images["test"])

    # 2. Questions of each (split, image, label), sorted for determinism.
    pool: dict[tuple[str, int, str], list[Question]] = defaultdict(list)
    for split in POPE_SPLITS:
        for q in sorted(questions_by_split[split], key=lambda q: q.question_id):
            pool[(split, q.image_id, q.label)].append(q)

    # Each image's yes questions in a random order; splits that need a yes
    # question take the next one, so no yes question is repeated.
    yes_order: dict[int, list[str]] = {}
    for image in sorted(images):
        texts = sorted(q.text for q in pool[(POPE_SPLITS[0], image, "yes")])
        rng.shuffle(texts)
        yes_order[image] = texts
    yes_used: dict[int, int] = defaultdict(int)

    chosen: dict[str, list[str]] = {}
    for partition in PARTITIONS:
        uids = []
        for split in POPE_SPLITS:
            order = list(partition_images[partition])
            rng.shuffle(order)
            half = len(order) // 2
            for i, image in enumerate(order):
                if i < half:  # yes question
                    text = yes_order[image][yes_used[image]]
                    yes_used[image] += 1
                    match = [q for q in pool[(split, image, "yes")] if q.text == text]
                    assert len(match) == 1, (split, image, text)
                    uids.append(match[0].uid)
                else:  # no question, from this split's own negatives
                    uids.append(rng.choice(pool[(split, image, "no")]).uid)
        # Shuffle once so any prefix (pilot, partial runs) mixes splits and labels.
        rng.shuffle(uids)
        chosen[partition] = uids

    return {
        "seed": seed,
        "dev_fraction": dev_fraction,
        "n_images": len(images),
        "images": partition_images,
        "questions": chosen,
    }


def save_splits(splits: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=1)
        f.write("\n")


def summary_rows(splits: dict, questions_by_split: dict[str, list[Question]]) -> list[dict]:
    """Counts per (partition, POPE split): questions, yes, no, images. Used for the CSV."""
    by_uid = {q.uid: q for qs in questions_by_split.values() for q in qs}
    rows = []
    for partition in PARTITIONS:
        for split in POPE_SPLITS:
            qs = [by_uid[u] for u in splits["questions"][partition] if by_uid[u].pope_split == split]
            rows.append(
                {
                    "partition": partition,
                    "pope_split": split,
                    "n_questions": len(qs),
                    "n_yes": sum(q.label == "yes" for q in qs),
                    "n_no": sum(q.label == "no" for q in qs),
                    "n_images": len({q.image_id for q in qs}),
                }
            )
    overlap = set(splits["images"]["dev"]) & set(splits["images"]["test"])
    rows.append(
        {
            "partition": "check",
            "pope_split": "images_in_both_dev_and_test",
            "n_questions": "",
            "n_yes": "",
            "n_no": "",
            "n_images": len(overlap),
        }
    )
    return rows


def write_summary_csv(rows: list[dict], path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_questions(
    partition: str,
    splits_path: str | Path,
    pope_dir: str | Path,
    image_dir: str | Path,
) -> list[Question]:
    """Questions of one partition, in the fixed order stored in splits.json.

    The test partition is locked: it needs the environment variable
    ALLOW_TEST=1, which only scripts/run_final.py --final sets.
    """
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    if partition == "test" and os.environ.get("ALLOW_TEST") != "1":
        raise TestSplitLockedError(
            "The test split is locked. Only scripts/run_final.py --final may read it (CLAUDE.md rule 3)."
        )
    with open(splits_path) as f:
        splits = json.load(f)
    by_uid = {q.uid: q for qs in load_all_pope(pope_dir, image_dir).values() for q in qs}
    return [by_uid[u] for u in splits["questions"][partition]]
