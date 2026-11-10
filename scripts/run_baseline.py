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
from vlmsyc.prompts import n_images, turn2_messages  # noqa: E402


def check_condition_rendered(model, setup, q) -> None:
    """Stop if the chat template silently drops the condition's text or image.

    Some chat templates ignore a system message; then the grounding run would
    quietly be a copy of the baseline. Prints the rendered turn-2 prompt once.
    """
    cfg = setup["prompt_cfg"]
    msgs = turn2_messages(q, "yes", setup["pushbacks"][0], cfg)
    text = model.render_prompt(msgs)
    print("--- rendered turn-2 prompt (first question, first pushback) ---")
    print(text)
    for key in ("system", "relook"):
        if cfg.get(key) and cfg[key] not in text:
            raise RuntimeError(f"The {key} text is missing from the rendered prompt; the template drops it.")
    if cfg.get("relook") and n_images(msgs) != 2:
        raise RuntimeError("The re-look prompt should contain the image twice.")


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
    parser.add_argument("--condition", default="base",
                        help="base, or a Phase 6 condition from configs/pushback.yaml (grounding, relook, nosuffix)")
    args = parser.parse_args()

    k, n = parse_shard(args.shard)
    model_cfg = load_yaml(args.model_config)
    setup = prepare_dev_run(args.experiment, model_cfg, stage="baseline", condition=args.condition)
    questions = select_shard(setup["questions"], k, n)[: args.limit]

    model = load_model(model_cfg)
    check_condition_rendered(model, setup, questions[0])
    kind = "baseline" if args.condition == "base" else args.condition
    out = output_path(results_dir(), kind, args.partition, model.name, k, n)
    print(f"condition {args.condition}, shard {k}/{n}: {len(questions)} questions x "
          f"{len(setup['pushbacks'])} pushbacks -> {out}")
    stats = run_to_file(
        model, questions, setup["pushbacks"], setup["prompt_cfg"], out, condition=args.condition,
        partition=args.partition, resume=args.resume, batch_size=args.batch_size or model_cfg["batch_size"],
        max_new_questions=args.max_new, image_loader=setup["image_loader"], seed=setup["seed"],
        git_commit=setup["git_commit"], meta={**setup["meta"], "shard": f"{k}/{n}", "model": model.describe()},
        log_every=5,
    )
    print(f"run stats: {stats}")
    print(f"REMAINING={stats['remaining']}")


if __name__ == "__main__":
    main()
