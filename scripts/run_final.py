"""Phase 7: the ONE run on the test split, then the frozen-gate analysis.

    python scripts/run_final.py --final                       # the real run (TC1, SLURM)
    python scripts/run_final.py --final --resume              # continue it after a timeout
    python scripts/run_final.py --dry-run-dev                 # same steps on dev (no tag needed)
    python scripts/run_final.py --dry-run-dev --mock --limit 6 --out-dir /tmp/x   # CPU orchestration test

Steps, per model (the model is loaded once, and only if a step has work left):
  1. base condition, 900 test questions x 12 pushbacks  -> baseline_<p>_<model>.jsonl
  2. blurred / blank turn-1 passes                       -> turn1_degraded_<p>_<model>.jsonl
  3. grounding and re-look conditions                    -> grounding_<p>_<model>.jsonl, relook_<p>_<model>.jsonl
  4. join signals (CPU)                                  -> signals_<p>_<model>.jsonl
Then scripts/analyse_final.py applies results/frozen_gates.json unchanged.

Guards for --final (scripts/analyse_final.py repeats the first):
  - HEAD must carry the git tag 'pre-final' and the work tree must be clean
    (only this run's own new output files are tolerated, so a resume works);
  - the frozen gates are read from the committed results/frozen_gates.json, never refit,
    and their sha256 is written into every output header;
  - an existing output file is never overwritten; with --resume, every header line
    in it must match this run (commit, partition, model, condition, frozen-gate hash);
  - only after these checks is the test partition unlocked for this process.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlmsyc.paths import REPO_ROOT, data_dir, git_commit, load_yaml, results_dir  # noqa: E402
from vlmsyc.pipeline import (  # noqa: E402
    append_jsonl, pending_questions, prepare_run, read_rows, run_degraded_passes, run_to_file,
)
from vlmsyc.release import check_output_file, check_release, sha256_file  # noqa: E402
from vlmsyc.signals import DEGRADATIONS, signal_rows  # noqa: E402

MODELS = ["qwen2vl_2b", "llava15_7b_fp16"]
CONDITIONS = ["base", "grounding", "relook"]   # no-suffix stays dev-only (analysis_plan.md, Amendment A5)
FROZEN = REPO_ROOT / "results" / "frozen_gates.json"
EXPERIMENT = "configs/experiment.yaml"


def file_for(out_dir: Path, kind: str, partition: str, model: str) -> Path:
    return out_dir / f"{kind}_{partition}_{model}.jsonl"


def run_model(model_cfg: dict, partition: str, out_dir: Path, header: dict, resume: bool, limit: int | None,
              strict_headers: bool) -> None:
    """Steps 1-4 for one model. The model is loaded lazily, only if some step has work left."""
    name = model_cfg["name"]
    sig_cfg = load_yaml(EXPERIMENT)["signals"]
    model = None

    def get_model():
        nonlocal model
        if model is None:
            from vlmsyc.models import load_model
            model = load_model(model_cfg)
        return model

    def expected(condition: str, stage: str) -> dict:
        # --final: every header field must match this run (commit, tag, frozen-gate hash,
        # partition, condition, stage). The dev dry run reuses dev files written by older
        # commits with older header formats, so it only requires that a header exists.
        if not strict_headers:
            return {}
        return {**header, "partition": partition, "condition": condition, "stage": stage}

    base_setup = None
    for cond in CONDITIONS:
        setup = prepare_run(EXPERIMENT, model_cfg, "baseline", cond, partition)
        if limit:
            setup["questions"] = setup["questions"][:limit]
        base_setup = base_setup or setup
        kind = "baseline" if cond == "base" else cond
        path = file_for(out_dir, kind, partition, name)
        check_output_file(path, expected(cond, "baseline"), resume)
        todo = pending_questions(path, setup["questions"], setup["pushbacks"], name, cond)
        print(f"[{name}] {cond}: {len(todo)} of {len(setup['questions'])} questions to run -> {path}", flush=True)
        if todo:
            m = get_model()
            run_to_file(m, setup["questions"], setup["pushbacks"], setup["prompt_cfg"], path, condition=cond,
                        partition=partition, resume=True, batch_size=model_cfg["batch_size"],
                        image_loader=setup["image_loader"], seed=setup["seed"], git_commit=header["git_commit"],
                        meta={**setup["meta"], **header, "model": m.describe()}, log_every=5)
        if cond == "base":
            # Step 2 right after the base run (needs the same questions; turn 1 only).
            dpath = file_for(out_dir, "turn1_degraded", partition, name)
            check_output_file(dpath, expected("base", "signals"), resume)
            done = {(r["uid"], r["degradation"]) for r in read_rows(dpath)}
            left = [q for q in setup["questions"] for dg in DEGRADATIONS if (q.uid, dg) not in done]
            print(f"[{name}] degraded passes: {len(left)} to run -> {dpath}", flush=True)
            if left:
                m = get_model()
                run_degraded_passes(m, setup["questions"], setup["prompt_cfg"], sig_cfg, dpath, partition=partition,
                                    batch_size=model_cfg["batch_size"], image_loader=setup["image_loader"],
                                    seed=setup["seed"], git_commit=header["git_commit"],
                                    meta={**setup["meta"], **header, "stage": "signals", "signals_cfg": sig_cfg})

    # Step 4: join (CPU). Written once; an existing complete file is kept, never rewritten.
    spath = file_for(out_dir, "signals", partition, name)
    base_rows = read_rows(file_for(out_dir, "baseline", partition, name))
    degraded = {(r["uid"], r["degradation"]): r["p1_yes"] for r in read_rows(file_for(out_dir, "turn1_degraded", partition, name))}
    expect_rows = len(base_setup["questions"]) * len(base_setup["pushbacks"])
    if len(base_rows) != expect_rows or len(degraded) != len(base_setup["questions"]) * len(DEGRADATIONS):
        raise RuntimeError(f"[{name}] incomplete before join: {len(base_rows)}/{expect_rows} rows, "
                           f"{len(degraded)} degraded passes. Resume the run.")
    if spath.exists():
        check_output_file(spath, expected("base", "signals_join"), resume=True)
        if len(read_rows(spath)) != expect_rows:
            raise RuntimeError(f"{spath} exists but is incomplete; inspect it by hand (never overwritten).")
        print(f"[{name}] signals already joined: {spath}")
    else:
        rows = signal_rows(base_rows, degraded)
        append_jsonl(spath, [{"_meta": {**header, "partition": partition, "condition": "base",
                                        "stage": "signals_join", "signals_cfg": sig_cfg,
                                        "joined": time.strftime("%Y-%m-%d %H:%M:%S")}}])
        append_jsonl(spath, rows)
        print(f"[{name}] joined {len(rows)} signal rows -> {spath}")
    del model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--final", action="store_true", help="the one real run on the test split")
    mode.add_argument("--dry-run-dev", action="store_true", help="same steps on dev; no tag needed")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--mock", action="store_true", help="dry run only: use the CPU mock model")
    parser.add_argument("--limit", type=int, help="dry run only: first N questions")
    parser.add_argument("--out-dir", type=Path, help="dry run only (default RESULTS_DIR)")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--n-boot", type=int, help="dry run only: fewer bootstrap resamples")
    args = parser.parse_args()

    if args.final:
        if args.mock or args.limit or args.out_dir or args.n_boot:
            sys.exit("--mock/--limit/--out-dir/--n-boot are for the dry run only.")
        commit = check_release()                       # tag pre-final on HEAD + clean tree
        if not FROZEN.exists():
            sys.exit("results/frozen_gates.json is missing.")
        # The split used must be the committed one, even if DATA_DIR points elsewhere.
        used, committed = data_dir() / "splits.json", REPO_ROOT / "data" / "splits.json"
        if sha256_file(used) != sha256_file(committed):
            sys.exit(f"{used} differs from the committed data/splits.json; refusing.")
        partition, out_dir = "test", results_dir()
        header = {"git_commit": commit, "release_tag": "pre-final", "frozen_gates_sha256": sha256_file(FROZEN)}
        # Every guard passed: unlock the test partition for this process only.
        os.environ["ALLOW_TEST"] = "1"
        strict = True
    else:
        partition = "dev"
        out_dir = args.out_dir or results_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        header = {"git_commit": git_commit(), "release_tag": "none (dry run)",
                  "frozen_gates_sha256": sha256_file(FROZEN)}
        strict = False

    print(f"run_final: partition={partition} out={out_dir} commit={header['git_commit']}", flush=True)
    for name in args.models:
        cfg = load_yaml("configs/models/mock.yaml") if args.mock else load_yaml(f"configs/models/{name}.yaml")
        if args.mock:
            cfg = {**cfg, "name": name}  # mock model under the real name, so file names match
        run_model(cfg, partition, out_dir, header, args.resume, args.limit if args.dry_run_dev else None, strict)

    if args.skip_analysis:
        return
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import analyse_final

    tag = "final_test" if args.final else "final_dryrun_dev"
    fig_dir = REPO_ROOT / "figures" / tag if args.out_dir is None else out_dir / "figures"
    analyse_final.run(partition, args.final, out_dir, out_dir / tag, fig_dir, args.models, FROZEN,
                      None if args.final else args.n_boot)
    if args.final:
        print("Final run complete. Copy results/ and figures/final_test/ back and commit them. Do not re-run.")


if __name__ == "__main__":
    main()
