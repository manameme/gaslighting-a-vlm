import json
import subprocess
import sys

import pytest

from vlmsyc.data import load_all_pope
from vlmsyc.metrics import flip_summary
from vlmsyc.models import MockVLM
from vlmsyc.paths import REPO_ROOT, load_yaml
from vlmsyc.pipeline import (
    BatchRunner,
    append_jsonl,
    output_path,
    parse_shard,
    read_rows,
    run_question,
    run_questions,
    run_to_file,
    select_shard,
)
from vlmsyc.prompts import pushbacks_from_config

CFG = load_yaml("configs/pushback.yaml")
PUSHBACKS = pushbacks_from_config(CFG)  # 12
NO_IMAGE = lambda q: None  # noqa: E731  (the mock ignores images)


@pytest.fixture
def questions(fake_pope):
    return load_all_pope(*fake_pope)["random"][:10]


def run(model, qs, path, **kw):
    args = dict(partition="dev", resume=True, batch_size=4, image_loader=NO_IMAGE)
    args.update(kw)
    return run_to_file(model, qs, PUSHBACKS, CFG, path, **args)


def test_rows_fields_and_logic(questions):
    rows = run_questions(MockVLM(), questions[:2], PUSHBACKS, CFG, partition="dev", image_loader=NO_IMAGE)
    assert len(rows) == 2 * 12
    for r in rows:
        assert r["a1"] == ("yes" if r["p1_yes"] > 0.5 else "no")
        assert r["flip"] == (r["a1"] != r["a2"])
        assert r["a1_correct"] == (r["a1"] == r["label"])
        assert 0 < r["yesno_mass_t1"] <= 1
    # Turn 1 is shared: one a1 per question across its 12 pushbacks.
    assert len({(r["uid"], r["p1_yes"]) for r in rows}) == 2


def test_run_question_matches_batch(questions):
    q, pb = questions[0], PUSHBACKS[3]
    single = run_question(MockVLM(), q, pb, CFG, partition="dev", image_loader=NO_IMAGE)
    batch = run_questions(MockVLM(), [q], PUSHBACKS, CFG, partition="dev", image_loader=NO_IMAGE)
    assert single == batch[3]


def test_resume_skips_done_and_finishes(questions, tmp_path):
    path = tmp_path / "out.jsonl"
    s1 = run(MockVLM(), questions, path, max_new_questions=3)
    assert s1["questions_run"] == 3 and s1["remaining"] == 7
    s2 = run(MockVLM(), questions, path)
    assert s2["questions_run"] == 7 and s2["remaining"] == 0
    s3 = run(MockVLM(), questions, path)
    assert s3["questions_run"] == 0 and s3["rows_written"] == 0
    rows = read_rows(path)
    assert len(rows) == 10 * 12 and len({json.dumps(r, sort_keys=True) for r in rows}) == 120


def test_resumed_file_equals_one_shot(questions, tmp_path):
    run(MockVLM(), questions, tmp_path / "a.jsonl", max_new_questions=4)
    run(MockVLM(), questions, tmp_path / "a.jsonl")
    run(MockVLM(), questions, tmp_path / "b.jsonl")
    key = lambda r: (r["uid"], r["pushback_type"], r["paraphrase_id"])  # noqa: E731

    def content(path):  # timing differs between runs, everything else must match
        return sorted(({k: v for k, v in r.items() if k != "batch_sec_per_question"} for r in read_rows(path)), key=key)

    assert content(tmp_path / "a.jsonl") == content(tmp_path / "b.jsonl")
    assert all(r["batch_sec_per_question"] >= 0 for r in read_rows(tmp_path / "a.jsonl"))


def test_partial_question_and_torn_line_recovered(questions, tmp_path):
    """A session killed mid-write leaves part of a question and a cut-off line."""
    path = tmp_path / "out.jsonl"
    full = run_questions(MockVLM(), questions[:1], PUSHBACKS, CFG, partition="dev", image_loader=NO_IMAGE)
    append_jsonl(path, full[:5])
    with open(path, "a") as f:
        f.write('{"uid": "random:1", "mod')  # torn line
    run(MockVLM(), questions[:1], path)
    rows = read_rows(path)
    assert len(rows) == 12 and len({(r["pushback_type"], r["paraphrase_id"]) for r in rows}) == 12


def test_refuses_to_append_without_resume(questions, tmp_path):
    path = tmp_path / "out.jsonl"
    run(MockVLM(), questions[:1], path)
    with pytest.raises(FileExistsError):
        run(MockVLM(), questions[:1], path, resume=False)


def test_meta_line_written_and_skipped(questions, tmp_path):
    path = tmp_path / "out.jsonl"
    run(MockVLM(), questions[:1], path, meta={"versions": {"x": 1}})
    lines = path.read_text().splitlines()
    assert "_meta" in json.loads(lines[0]) and len(read_rows(path)) == 12


def test_oom_halves_batch_and_keeps_going(questions, tmp_path):
    model = MockVLM(oom_above=3)
    s = run(model, questions[:2], tmp_path / "out.jsonl", batch_size=8)
    assert s["remaining"] == 0 and len(read_rows(tmp_path / "out.jsonl")) == 24
    assert max(model.batch_sizes_seen[-3:]) <= 3


def test_oom_at_batch_size_one_raises():
    runner = BatchRunner(MockVLM(oom_above=0), batch_size=4)
    with pytest.raises(RuntimeError):
        runner([[{"role": "user", "content": [{"type": "text", "text": "x"}]}]], [None])


def test_shards_disjoint_and_complete(questions):
    shards = [select_shard(questions, k, 3) for k in range(3)]
    uids = [q.uid for s in shards for q in s]
    assert sorted(uids) == sorted(q.uid for q in questions) and len(uids) == len(set(uids))


def test_parse_shard():
    assert parse_shard("2/5") == (2, 5)
    for bad in ["5/5", "-1/2", "1", "a/b", "0/0"]:
        with pytest.raises(ValueError):
            parse_shard(bad)


def test_merge_shards_script(questions, tmp_path):
    """Run 3 shards, run one twice (duplicates), and merge with the real script."""
    for k in range(3):
        path = output_path(tmp_path, "baseline", "dev", "mock", k, 3)
        run(MockVLM(name="mock"), select_shard(questions, k, 3), path)
    dup = output_path(tmp_path, "baseline", "dev", "mock", 0, 3)
    append_jsonl(dup, read_rows(dup)[:2])
    out = subprocess.run(
        [sys.executable, "scripts/merge_shards.py", "--model", "mock", "--expect-questions", "10", "--expect-pushbacks", "12"],
        cwd=REPO_ROOT, env={"RESULTS_DIR": str(tmp_path), "PATH": ""}, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert '"duplicates": 2' in out.stdout and '"conflicts": 0' in out.stdout
    assert len(read_rows(tmp_path / "baseline_dev_mock.jsonl")) == 120


def test_flip_summary_hand_computed():
    rows = [
        {"uid": "a", "a1_correct": True, "flip": True, "a2_correct": False, "yesno_mass_t1": 0.9, "yesno_mass_t2": 0.5},
        {"uid": "a", "a1_correct": True, "flip": False, "a2_correct": True, "yesno_mass_t1": 0.9, "yesno_mass_t2": 0.7},
        {"uid": "b", "a1_correct": False, "flip": True, "a2_correct": True, "yesno_mass_t1": 0.5, "yesno_mass_t2": 0.3},
    ]
    s = flip_summary(rows)
    assert s["n_questions"] == 2 and s["turn1_acc"] == 0.5
    assert s["flip_rate"] == 2 / 3 and s["flip_rate_a1_correct"] == 0.5 and s["flip_rate_a1_wrong"] == 1.0
    assert s["min_mass_t2"] == 0.3


def test_resume_refuses_different_suffix(questions, tmp_path):
    """Row keys do not include the prompt, so a file must not mix two suffix settings."""
    path = tmp_path / "out.jsonl"
    run(MockVLM(), questions[:1], path, meta={"pushback_suffix": ""})
    with pytest.raises(ValueError):
        run(MockVLM(), questions[:2], path, meta={"pushback_suffix": "Answer yes or no."})
    run(MockVLM(), questions[:2], path, meta={"pushback_suffix": ""})  # same setting resumes fine
