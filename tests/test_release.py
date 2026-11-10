"""Guards of the one-time final run (scripts/run_final.py --final)."""

import os
import subprocess
import sys

import pytest

from vlmsyc.paths import REPO_ROOT
from vlmsyc.pipeline import append_jsonl
from vlmsyc.release import ReleaseError, check_output_file, check_release


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "results").mkdir()
    (tmp_path / "a.txt").write_text("x")
    git(tmp_path, "add", "a.txt")
    git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_release_needs_tag_on_head(repo):
    with pytest.raises(ReleaseError, match="pre-final"):
        check_release(repo)
    git(repo, "tag", "pre-final")
    assert len(check_release(repo)) == 40


def test_release_tag_must_be_on_head_not_older_commit(repo):
    git(repo, "tag", "pre-final")
    (repo / "a.txt").write_text("y")
    git(repo, "commit", "-qam", "later")
    with pytest.raises(ReleaseError):
        check_release(repo)


def test_release_needs_clean_tree_but_tolerates_own_outputs(repo):
    git(repo, "tag", "pre-final")
    (repo / "results" / "baseline_test_qwen2vl_2b.jsonl").write_text("{}\n")   # own output (resume)
    check_release(repo)
    (repo / "results" / "notes.txt").write_text("x")                           # anything else
    with pytest.raises(ReleaseError, match="not clean"):
        check_release(repo)


def test_release_rejects_modified_tracked_file(repo):
    git(repo, "tag", "pre-final")
    (repo / "a.txt").write_text("changed")
    with pytest.raises(ReleaseError):
        check_release(repo)


def test_output_file_never_overwritten_and_resume_needs_matching_header(tmp_path):
    path = tmp_path / "baseline_test_m.jsonl"
    hdr = {"git_commit": "abc", "partition": "test", "condition": "base"}
    check_output_file(path, hdr, resume=False)                     # new file: fine
    append_jsonl(path, [{"_meta": hdr}, {"uid": "x"}])
    with pytest.raises(FileExistsError):
        check_output_file(path, hdr, resume=False)                 # never overwrite
    check_output_file(path, hdr, resume=True)                      # same run: resume ok
    with pytest.raises(ValueError):
        check_output_file(path, {**hdr, "git_commit": "def"}, resume=True)
    bare = tmp_path / "bare.jsonl"
    append_jsonl(bare, [{"uid": "x"}])
    with pytest.raises(ValueError, match="no header"):
        check_output_file(bare, hdr, resume=True)


def test_run_final_refuses_without_tag():
    """In the working repo HEAD is not tagged pre-final, so --final must stop before any model or data."""
    tags = subprocess.run(["git", "tag", "--points-at", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout
    if "pre-final" in tags.split():
        pytest.skip("HEAD is tagged pre-final")
    out = subprocess.run([sys.executable, "scripts/run_final.py", "--final"], cwd=REPO_ROOT,
                         capture_output=True, text=True, env={**os.environ, "ALLOW_TEST": ""})
    assert out.returncode != 0 and "ReleaseError" in out.stderr


def test_analyse_final_test_needs_final_flag():
    out = subprocess.run([sys.executable, "scripts/analyse_final.py", "--partition", "test"], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    assert out.returncode != 0 and "--final" in (out.stdout + out.stderr)


@pytest.mark.skipif(not (REPO_ROOT / "data" / "coco_val2014").exists(), reason="dev images not downloaded")
def test_dry_run_dev_with_mock_end_to_end(tmp_path):
    """The whole final pipeline on 4 dev questions with the mock model; then the never-overwrite rule."""
    cmd = [sys.executable, "scripts/run_final.py", "--dry-run-dev", "--mock", "--limit", "4",
           "--out-dir", str(tmp_path), "--n-boot", "20"]
    first = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr[-2000:]
    for name in ("baseline", "turn1_degraded", "grounding", "relook", "signals"):
        assert (tmp_path / f"{name}_dev_qwen2vl_2b.jsonl").exists()
    assert (tmp_path / "final_dryrun_dev" / "h2.csv").exists() and (tmp_path / "figures" / "eir_ecr.png").exists()
    again = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert again.returncode != 0 and "never overwrites" in again.stderr
    resumed = subprocess.run(cmd + ["--resume"], cwd=REPO_ROOT, capture_output=True, text=True)
    assert resumed.returncode == 0, resumed.stderr[-2000:]
    assert "0 of 4 questions to run" in resumed.stdout
