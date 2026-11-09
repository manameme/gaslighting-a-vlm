"""Merge shard files into one JSONL and check it.

    python scripts/merge_shards.py --model qwen2vl_2b
    python scripts/merge_shards.py --model qwen2vl_2b --expect-questions 600 --expect-pushbacks 12

Reads RESULTS_DIR/baseline_dev_<model>_shard*of*.jsonl and writes
RESULTS_DIR/baseline_dev_<model>.jsonl. Reports, rather than hides:
  - shards that are missing, or files from different n;
  - duplicate rows (the same key twice; the first copy is kept);
  - conflicting duplicates (same key, different answers);
  - questions with fewer rows than expected.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlmsyc.paths import git_commit, results_dir  # noqa: E402
from vlmsyc.pipeline import read_rows, row_key  # noqa: E402

COMPARE = ("a1", "a2", "p1_yes", "p2_yes")


def merge(paths: list[Path]) -> tuple[list[dict], dict]:
    """Rows from all files, first copy of each key kept, plus a report."""
    merged, seen = [], {}
    report = {"files": {}, "duplicates": 0, "conflicts": 0}
    for path in paths:
        rows = read_rows(path)
        report["files"][path.name] = len(rows)
        for row in rows:
            key = row_key(row)
            if key in seen:
                report["duplicates"] += 1
                if any(seen[key][f] != row[f] for f in COMPARE):
                    report["conflicts"] += 1
                continue
            seen[key] = row
            merged.append(row)
    return merged, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", default="baseline")
    parser.add_argument("--partition", default="dev", choices=["dev"])
    parser.add_argument("--expect-questions", type=int)
    parser.add_argument("--expect-pushbacks", type=int)
    args = parser.parse_args()

    stem = f"{args.kind}_{args.partition}_{args.model}"
    pattern = re.compile(rf"^{re.escape(stem)}_shard(\d+)of(\d+)\.jsonl$")
    found = sorted(p for p in results_dir().iterdir() if pattern.match(p.name))
    if not found:
        raise SystemExit(f"No files matching {stem}_shard*of*.jsonl in {results_dir()}")

    ns = {int(pattern.match(p.name).group(2)) for p in found}
    ks = {int(pattern.match(p.name).group(1)) for p in found}
    problems = []
    if len(ns) > 1:
        problems.append(f"files from different shard counts n={sorted(ns)}")
    else:
        missing = set(range(next(iter(ns)))) - ks
        if missing:
            problems.append(f"missing shards {sorted(missing)} of n={next(iter(ns))}")

    rows, report = merge(found)
    per_question = Counter(r["uid"] for r in rows)
    if args.expect_pushbacks:
        short = [u for u, c in per_question.items() if c < args.expect_pushbacks]
        if short:
            problems.append(f"{len(short)} questions have fewer than {args.expect_pushbacks} rows")
    if args.expect_questions and len(per_question) != args.expect_questions:
        problems.append(f"{len(per_question)} questions, expected {args.expect_questions}")
    if report["conflicts"]:
        problems.append(f"{report['conflicts']} duplicate rows disagree")

    out = results_dir() / f"{stem}.jsonl"
    with open(out, "w") as f:
        f.write(json.dumps({"_meta": {"merged_from": report["files"], "merge_git_commit": git_commit()}}) + "\n")
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(json.dumps(report, indent=1))
    print(f"{len(rows)} rows, {len(per_question)} questions -> {out}")
    print("PROBLEMS:\n  " + "\n  ".join(problems) if problems else "No problems found.")


if __name__ == "__main__":
    main()
