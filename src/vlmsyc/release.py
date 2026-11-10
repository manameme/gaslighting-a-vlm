"""Guards for the one-time final run on the test split (Phase 7).

scripts/run_final.py --final calls these before it touches the test split:
  check_release()        HEAD carries the tag 'pre-final' and the work tree is clean
                         (the only files allowed to be new are the final run's own outputs,
                         so a resumed run is still accepted);
  check_output_file()    never overwrite an existing output; resume only when every
                         header line of the existing file matches this run;
  sha256_file()          fingerprint of frozen_gates.json, stored in every output.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from .paths import REPO_ROOT
from .pipeline import read_metas, read_rows

RELEASE_TAG = "pre-final"

# New (untracked) files a final run itself creates; anything else makes the tree dirty.
FINAL_OUTPUT_PATTERN = re.compile(
    r"^(results/[^/]*_test_[^/]*\.jsonl|results/final_test/.+|figures/final_test/.+)$"
)


class ReleaseError(RuntimeError):
    """The repository is not in the frozen state required for the final run."""

    __test__ = False


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if out.returncode != 0:
        raise ReleaseError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def check_release(cwd: Path = REPO_ROOT, tag: str = RELEASE_TAG) -> str:
    """Return the HEAD commit if HEAD is tagged `tag` and the tree is clean; raise otherwise."""
    if not (Path(cwd) / ".git").exists():
        raise ReleaseError("The final run needs a git checkout (copy the .git folder too) to check the tag.")
    head = _git("rev-parse", "HEAD", cwd=cwd).strip()
    tags = _git("tag", "--points-at", "HEAD", cwd=cwd).split()
    if tag not in tags:
        raise ReleaseError(f"HEAD {head[:8]} does not carry the tag '{tag}' (tags here: {tags or 'none'}).")
    problems = []
    for line in _git("status", "--porcelain", "--untracked-files=all", cwd=cwd).splitlines():
        status, path = line[:2], line[3:]
        if status == "??" and FINAL_OUTPUT_PATTERN.match(path):
            continue  # this run's own outputs (a resumed run)
        problems.append(line)
    if problems:
        raise ReleaseError("The work tree is not clean:\n  " + "\n  ".join(problems))
    return head


def check_output_file(path: str | Path, expected: dict, resume: bool) -> None:
    """Refuse to overwrite; allow appending only if every header matches `expected`.

    expected holds the header fields that must be identical, e.g. the git
    commit, partition, model and condition. A file without a header is refused.
    """
    path = Path(path)
    if not path.exists() or (not read_rows(path) and not read_metas(path)):
        return
    if not resume:
        raise FileExistsError(f"{path} exists. The final run never overwrites; use --resume to continue it.")
    metas = read_metas(path)
    if not metas:
        raise ValueError(f"{path} has no header line, so it cannot be checked; refusing to resume.")
    for meta in metas:
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"{path}: header {key}={meta.get(key)!r}, this run has {value!r}; refusing to resume.")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
