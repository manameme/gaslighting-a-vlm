import json
import re
from collections import Counter
from pathlib import Path

import pytest

from vlmsyc.data import POPE_SPLITS, load_all_pope
from vlmsyc.paths import REPO_ROOT, data_dir, load_yaml
from vlmsyc.splits import TestSplitLockedError, build_splits, load_questions, save_splits, summary_rows


@pytest.fixture
def built(fake_pope):
    questions = load_all_pope(*fake_pope)
    return questions, build_splits(questions, seed=0, dev_fraction=0.4)


def test_no_image_in_both_partitions(built):
    _, splits = built
    assert not set(splits["images"]["dev"]) & set(splits["images"]["test"])
    assert len(splits["images"]["dev"]) == 8 and len(splits["images"]["test"]) == 12


def test_one_question_per_image_per_split_balanced(built):
    questions, splits = built
    by_uid = {q.uid: q for qs in questions.values() for q in qs}
    for partition in ("dev", "test"):
        qs = [by_uid[u] for u in splits["questions"][partition]]
        for split in POPE_SPLITS:
            in_split = [q for q in qs if q.pope_split == split]
            images = [q.image_id for q in in_split]
            assert sorted(images) == sorted(splits["images"][partition])  # each image exactly once
            labels = Counter(q.label for q in in_split)
            assert labels["yes"] == labels["no"]


def test_no_question_text_repeated_for_an_image(built):
    """The yes questions are identical across POPE files, so a repeat would waste compute."""
    questions, splits = built
    by_uid = {q.uid: q for qs in questions.values() for q in qs}
    pairs = [(by_uid[u].image_id, by_uid[u].text) for p in ("dev", "test") for u in splits["questions"][p]]
    assert len(pairs) == len(set(pairs))


def test_deterministic(fake_pope):
    questions = load_all_pope(*fake_pope)
    assert build_splits(questions, seed=0) == build_splits(questions, seed=0)
    assert build_splits(questions, seed=0) != build_splits(questions, seed=1)


def test_n_images_subsample(fake_pope):
    splits = build_splits(load_all_pope(*fake_pope), n_images=10)
    assert len(splits["images"]["dev"]) + len(splits["images"]["test"]) == 10


def test_summary_overlap_check_row(built):
    questions, splits = built
    rows = summary_rows(splits, questions)
    assert rows[-1]["pope_split"] == "images_in_both_dev_and_test" and rows[-1]["n_images"] == 0


# ---- Test-split lock (CLAUDE.md rule 3) ----


def test_test_partition_locked(built, fake_pope, tmp_path, monkeypatch):
    _, splits = built
    save_splits(splits, tmp_path / "splits.json")
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    with pytest.raises(TestSplitLockedError):
        load_questions("test", tmp_path / "splits.json", *fake_pope)
    assert len(load_questions("dev", tmp_path / "splits.json", *fake_pope)) == 8 * 3


def test_test_partition_opens_only_with_allow_test(built, fake_pope, tmp_path, monkeypatch):
    _, splits = built
    save_splits(splits, tmp_path / "splits.json")
    monkeypatch.setenv("ALLOW_TEST", "1")
    assert len(load_questions("test", tmp_path / "splits.json", *fake_pope)) == 12 * 3


def test_only_run_final_may_unlock_test_split():
    """No script, notebook or library file other than run_final.py may set ALLOW_TEST
    or ask for the test partition."""
    allowed = {"run_final.py", "splits.py"}
    offenders = []
    for path in list((REPO_ROOT / "scripts").iterdir()) + list((REPO_ROOT / "src" / "vlmsyc").glob("*.py")):
        if path.name in allowed or path.suffix not in (".py", ".ipynb"):
            continue
        text = path.read_text()
        if "ALLOW_TEST" in text or re.search(r"""partition\s*=\s*\\?["']test""", text) or re.search(
            r"""load_questions\(\s*\\?["']test""", text
        ):
            offenders.append(path.name)
    assert offenders == []


# ---- The committed split file ----


def test_committed_splits_match_rebuild():
    """Rebuilding from the real POPE files must give the committed splits.json.
    Skipped when the POPE files are not downloaded."""
    cfg = load_yaml("configs/data.yaml")
    pope_dir = data_dir() / cfg["pope_dir"]
    committed = Path(data_dir() / cfg["splits_file"])
    if not (pope_dir / "coco_pope_random.json").exists() or not committed.exists():
        pytest.skip("POPE data not downloaded")
    questions = load_all_pope(pope_dir, data_dir() / cfg["image_dir"])
    rebuilt = build_splits(questions, seed=cfg["seed"], dev_fraction=cfg["dev_fraction"], n_images=cfg["n_images"])
    assert rebuilt == json.loads(committed.read_text())
