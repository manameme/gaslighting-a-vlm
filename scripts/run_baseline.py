"""Baseline two-turn run on the dev split (Phase 4).

    python scripts/run_baseline.py --model-config configs/models/qwen2vl_2b.yaml --shard 0/3 --resume

Runs every question of the shard x 4 pushback types x 3 paraphrases and
appends rows to RESULTS_DIR/baseline_dev_<model>_shard<k>of<n>.jsonl after
each batch. Re-run the same command with --resume after a session ends; it
skips finished rows. Merge shards with scripts/merge_shards.py.
Prints 'REMAINING=<n>' at the end (0 means the shard is finished).
Dev split only: the test split is read only by run_final.py --final.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlmsyc.models import load_model  # noqa: E402
from vlmsyc.paths import load_yaml, results_dir  # noqa: E402
from vlmsyc.pipeline import output_path, parse_shard, prepare_dev_run, run_to_file, select_shard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--experiment", default="configs/experiment.yaml")
    parser.add_argument("--partition", default="dev", choices=["dev"], help="only dev is allowed here")
    parser.add_argument("--shard", default="0/1", help="k/n: run the k-th of n disjoint shards (from 0)")
    parser.add_argument("--resume", action="store_true", help="skip rows already in the output file")
    parser.add_argument("--limit", type=int, help="only the first N questions of the shard (debugging)")
    parser.add_argument("--max-new", type=int, help="stop after N new questions in this call")
    parser.add_argument("--batch-size", type=int, help="override the model config")
    args = parser.parse_args()

    k, n = parse_shard(args.shard)
    model_cfg = load_yaml(args.model_config)
    setup = prepare_dev_run(args.experiment, model_cfg, stage="baseline")
    questions = select_shard(setup["questions"], k, n)[: args.limit]

    model = load_model(model_cfg)
    out = output_path(results_dir(), "baseline", args.partition, model.name, k, n)
    print(f"shard {k}/{n}: {len(questions)} questions x {len(setup['pushbacks'])} pushbacks -> {out}")
    stats = run_to_file(
        model, questions, setup["pushbacks"], setup["prompt_cfg"], out,
        partition=args.partition, resume=args.resume, batch_size=args.batch_size or model_cfg["batch_size"],
        max_new_questions=args.max_new, image_loader=setup["image_loader"], seed=setup["seed"],
        git_commit=setup["git_commit"], meta={**setup["meta"], "shard": f"{k}/{n}", "model": model.describe()},
        log_every=5,
    )
    print(f"run stats: {stats}")
    print(f"REMAINING={stats['remaining']}")


if __name__ == "__main__":
    main()
