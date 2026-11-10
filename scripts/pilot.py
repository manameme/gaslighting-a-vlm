"""Pilot (Phase 3) and hand check (Phase 2) on the first N dev questions.

    python scripts/pilot.py --model-config configs/models/qwen2vl_2b.yaml --resume

1. Runs N dev questions (default 100) x 4 pushback types x paraphrase 0.
   Rows go to RESULTS_DIR/pilot_dev_<model>.jsonl, written after each batch.
2. Prints the first few examples in full: the exact chat-template text,
   p(yes), p(no), turn-1 and turn-2 answers (the Phase 2 hand check).
3. Writes this model's line of RESULTS_DIR/pilot.csv: flip rates, accuracy,
   yes/no mass and seconds per question.
Dev split only.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlmsyc.metrics import flip_summary  # noqa: E402
from vlmsyc.models import load_model  # noqa: E402
from vlmsyc.paths import load_yaml, results_dir  # noqa: E402
from vlmsyc.pipeline import prepare_dev_run, read_rows, run_to_file  # noqa: E402
from vlmsyc.prompts import turn1_messages, turn2_messages  # noqa: E402


def show_examples(model, setup, rows, n_show):
    """Print the exact prompts and the numbers for the first n_show questions."""
    by_uid = {}
    for r in rows:
        by_uid.setdefault(r["uid"], []).append(r)
    for q in setup["questions"][:n_show]:
        q_rows = by_uid.get(q.uid)
        if not q_rows:
            continue
        a1 = q_rows[0]["a1"]
        print("=" * 80)
        print(f"{q.uid}  image {q.image_file}  object '{q.object}'  label {q.label}")
        print("--- turn-1 prompt (exact chat-template text) ---")
        print(model.render_prompt(turn1_messages(q, setup["prompt_cfg"])))
        print("--- turn-2 prompt for the first pushback ---")
        print(model.render_prompt(turn2_messages(q, a1, setup["pushbacks"][0], setup["prompt_cfg"])))
        r = q_rows[0]
        print(f"turn 1: p(yes)={r['p1_yes']:.3f} p(no)={1 - r['p1_yes']:.3f} -> a1={a1} "
              f"(correct={r['a1_correct']}, yes/no mass={r['yesno_mass_t1']:.3f})")
        for r in q_rows:
            print(f"  {r['pushback_type']:<14} p2(yes)={r['p2_yes']:.3f} p2(no)={1 - r['p2_yes']:.3f} "
                  f"-> a2={r['a2']} flip={r['flip']} mass={r['yesno_mass_t2']:.3f}  | {r['pushback_text']}")


def update_pilot_csv(path: Path, row: dict) -> None:
    """Replace this (model, variant) line in pilot.csv, keeping the other lines."""
    rows = []
    if path.exists():
        with open(path) as f:
            rows = [
                r for r in csv.DictReader(f)
                if (r["model"], r.get("variant", "")) != (row["model"], row["variant"])
            ]
    rows.append(row)
    fields = list(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in fields} for r in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--experiment", default="configs/experiment.yaml")
    parser.add_argument("--n", type=int, help="number of dev questions (default from experiment.yaml)")
    parser.add_argument("--batch-size", type=int, help="override the model config")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pushback-suffix", help='override the config, e.g. "" for no suffix')
    parser.add_argument("--tag", default="", help="variant name: goes into the file name and the variant column")
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)
    setup = prepare_dev_run(args.experiment, model_cfg, stage="pilot")
    if args.pushback_suffix is not None:
        setup["prompt_cfg"]["pushback_suffix"] = args.pushback_suffix
    # File naming kept from the first pilot: no tag = no suffix, "instr" = suffix.
    # A suffix without a tag would share a file (and resume keys) with no-suffix rows.
    if setup["prompt_cfg"].get("pushback_suffix") and not args.tag:
        parser.error('a pushback suffix is active, so give --tag (e.g. --tag instr), '
                     'or pass --pushback-suffix "" for the no-suffix variant')
    setup["meta"]["pushback_suffix"] = setup["prompt_cfg"].get("pushback_suffix", "")
    pilot_cfg = setup["exp"]["pilot"]
    setup["questions"] = setup["questions"][: args.n or pilot_cfg["n_questions"]]

    model = load_model(model_cfg)
    out = results_dir() / f"pilot_dev_{model.name}{'_' + args.tag if args.tag else ''}.jsonl"
    stats = run_to_file(
        model, setup["questions"], setup["pushbacks"], setup["prompt_cfg"], out,
        partition="dev", resume=args.resume, batch_size=args.batch_size or model_cfg["batch_size"],
        image_loader=setup["image_loader"], seed=setup["seed"], git_commit=setup["git_commit"],
        meta={**setup["meta"], "model": model.describe()},
    )
    print(f"run stats: {stats}")

    wanted = {q.uid for q in setup["questions"]}
    rows = [r for r in read_rows(out) if r["uid"] in wanted]
    show_examples(model, setup, rows, pilot_cfg["show"])

    summary = {"model": model.name, "variant": args.tag, **flip_summary(rows)}
    summary["pushback_suffix"] = setup["prompt_cfg"].get("pushback_suffix", "")
    # Mean over questions of the per-batch timing stored on each row.
    # Includes image loading and the first (slower, warm-up) batch.
    per_q = {r["uid"]: r["batch_sec_per_question"] for r in rows if "batch_sec_per_question" in r}
    summary["sec_per_question"] = sum(per_q.values()) / len(per_q) if per_q else None
    summary["prompts_per_question"] = 1 + len(setup["pushbacks"])
    summary["git_commit"] = setup["git_commit"]
    update_pilot_csv(results_dir() / "pilot.csv", summary)
    print("=" * 80)
    for k, v in summary.items():
        print(f"{k:>22}: {v}")


if __name__ == "__main__":
    main()
