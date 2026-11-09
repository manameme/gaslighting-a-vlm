"""Where files live, and run provenance (git commit, package versions).

Paths come from environment variables so the same code runs on a laptop, on
Kaggle and on Colab without hard-coded absolute paths:

- DATA_DIR    : POPE json, COCO images, splits.json. Default: <repo>/data
- RESULTS_DIR : every output file. Default: <repo>/results. Point it at a
                Google Drive folder on Colab, or a synced folder on Kaggle.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Folder for input data (env DATA_DIR, default <repo>/data)."""
    return Path(os.environ.get("DATA_DIR", REPO_ROOT / "data"))


def results_dir() -> Path:
    """Folder for all outputs (env RESULTS_DIR, default <repo>/results). Created if missing."""
    path = Path(os.environ.get("RESULTS_DIR", REPO_ROOT / "results"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def repo_path(relative: str | Path) -> Path:
    """Resolve a repo-relative path such as 'configs/data.yaml'."""
    return REPO_ROOT / relative


def load_yaml(path: str | Path) -> dict:
    """Load a yaml file; relative paths are taken from the repo root."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        return yaml.safe_load(f)


def git_commit() -> str:
    """Commit of the code that produced a result, stored in every output row.

    Order: env VLMSYC_GIT_COMMIT, then `git rev-parse HEAD`. A '-dirty' suffix
    means there were uncommitted changes, so the commit alone does not fully
    describe the code that ran.
    """
    if os.environ.get("VLMSYC_GIT_COMMIT"):
        return os.environ["VLMSYC_GIT_COMMIT"]
    try:
        run = lambda *args: subprocess.run(  # noqa: E731
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        commit = run("rev-parse", "HEAD")
        dirty = run("status", "--porcelain", "--untracked-files=no")
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def package_versions() -> dict:
    """Versions of the packages that affect results, for result file headers."""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ["numpy", "torch", "transformers", "accelerate", "bitsandbytes", "pillow"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions
