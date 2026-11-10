"""Signals on the dev split (Phase 5, first half).

    python scripts/run_signals.py --model-config configs/models/qwen2vl_2b.yaml --resume
    python scripts/run_signals.py --model-config configs/models/qwen2vl_2b.yaml --join-only   # CPU only

Step 1 (GPU): the turn-1 question again, with the image blurred and with a
  blank (mean-colour) image. One forward pass per question and degradation,
  appended to RESULTS_DIR/turn1_degraded_dev_<model>.jsonl after every batch.
  --resume skips (question, degradation) pairs already done.
Step 2 (CPU): join with RESULTS_DIR/baseline_dev_<model>.jsonl (the merged
  baseline) and write
    RESULTS_DIR/signals_dev_<model>.jsonl   one row per question x pushback
    RESULTS_DIR/signals_auroc_dev.csv       AUROC of each signal, with 95% CIs
                                            from a bootstrap clustered by image
  Two AUROC targets:
    flip           over all rows: does the signal predict that the model caves?
    good_revision  only among revisions (a2 != a1): does it predict that the
                   revision is correct? (SPEC section 8)
Dev split only. No thresholds are fitted here.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlmsyc.paths import git_commit as get_git_commit  # noqa: E402
from vlmsyc.paths import load_yaml, package_versions, results_dir  # noqa: E402
from vlmsyc.pipeline import BatchRunner, append_jsonl, check_meta_consistent, read_rows  # noqa: E402
from vlmsyc.prompts import turn1_messages  # noqa: E402
from vlmsyc.signals import DEGRADATIONS, degrade, signal_rows  # noqa: E402
from vlmsyc.stats import auroc, cluster_bootstrap_ci  # noqa: E402

# (column, score transform). A higher score should mean "more likely positive";
# AUROC below 0.5 means the signal points the other way.
SIGNALS = {
    "conf1": lambda r: r["conf1"],
    "reliance_blur": lambda r: r["reliance_blur"],
    "reliance_blank": lambda r: r["reliance_blank"],
    "shift_to_claim": lambda r: r["shift_to_claim"],
    "abs_shift": lambda r: abs(r["shift"]),
}
SUBSETS = ("all", "random", "popular", "adversarial")


def run_degraded(model, setup, model_cfg, sig_cfg, out_path, resume, batch_size):
    """Step 1: turn-1 p(yes) on degraded images, resumable."""
    if out_path.exists() and not resume and read_rows(out_path):
        raise FileExistsError(f"{out_path} already has rows. Use --resume.")
    meta = {**setup["meta"], "stage": "signals", "signals_cfg": sig_cfg, "model": model.describe()}
    check_meta_consistent(out_path, meta)
    done = {(r["uid"], r["degradation"]) for r in read_rows(out_path)}
    todo = [(q, d) for q in setup["questions"] for d in DEGRADATIONS if (q.uid, d) not in done]
    print(f"degraded turn-1 passes: {len(todo)} to do, {len(done)} already done -> {out_path}", flush=True)
    if not todo:
        return
    append_jsonl(out_path, [{"_meta": {**meta, "started": time.strftime("%Y-%m-%d %H:%M:%S")}}])
    runner = BatchRunner(model, batch_size)
    start = time.time()
    for b in range(0, len(todo), batch_size):
        batch = todo[b : b + batch_size]
        images = [degrade(setup["image_loader"](q), d, sig_cfg) for q, d in batch]
        results = runner([turn1_messages(q, setup["prompt_cfg"]) for q, _ in batch], images)
        append_jsonl(out_path, [
            {
                "uid": q.uid, "image_id": q.image_id, "pope_split": q.pope_split, "partition": "dev",
                "model": model.name, "degradation": d, "p1_yes": r.p_yes, "yesno_mass": r.mass,
                "logp_yes": r.logp_yes, "logp_no": r.logp_no,
                "seed": setup["seed"], "git_commit": setup["git_commit"],
            }
            for (q, d), r in zip(batch, results)
        ])
        if (b // batch_size) % 20 == 0:
            n = b + len(batch)
            print(f"  {n}/{len(todo)}  {(time.time() - start) / n:.3f} s/pass", flush=True)


def auroc_table(model_name: str, rows: list[dict], n_boot: int, seed: int) -> list[dict]:
    out = []
    for subset in SUBSETS:
        sub = rows if subset == "all" else [r for r in rows if r["pope_split"] == subset]
        for target in ("flip", "good_revision"):
            pool = sub if target == "flip" else [r for r in sub if r["flip"]]
            y = np.array([bool(r[target]) for r in pool])
            clusters = np.array([r["image_id"] for r in pool])
            for name, fn in SIGNALS.items():
                keep = [i for i, r in enumerate(pool) if fn(r) is not None]
                s = np.array([fn(pool[i]) for i in keep], dtype=float)
                yy, cc = y[keep], clusters[keep]
                value = auroc(s, yy)
                lo, hi = cluster_bootstrap_ci(lambda idx: auroc(s[idx], yy[idx]), cc, n_boot=n_boot, seed=seed)
                out.append({
                    "model": model_name, "pope_split": subset, "target": target, "signal": name,
                    "n_rows": len(keep), "n_positive": int(yy.sum()), "n_images": len(set(cc.tolist())),
                    "auroc": value, "ci_low": lo, "ci_high": hi, "n_boot": n_boot,
                })
    return out


def write_auroc_csv(path: Path, model_name: str, new_rows: list[dict]) -> None:
    """Replace this model's lines in the CSV, keep the other models'."""
    keep = []
    if path.exists():
        with open(path) as f:
            keep = [r for r in csv.DictReader(f) if r["model"] != model_name]
    fields = list(new_rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep + new_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--experiment", default="configs/experiment.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--join-only", action="store_true", help="skip the GPU step (CPU only)")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-boot", type=int, help="bootstrap resamples (default from experiment.yaml)")
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)
    exp = load_yaml(args.experiment)
    sig_cfg = exp["signals"]
    name = model_cfg["name"]
    out_dir = results_dir()
    degraded_path = out_dir / f"turn1_degraded_dev_{name}.jsonl"

    if not args.join_only:
        from vlmsyc.models import load_model
        from vlmsyc.pipeline import prepare_dev_run

        setup = prepare_dev_run(args.experiment, model_cfg, stage="baseline")
        model = load_model(model_cfg)
        run_degraded(model, setup, model_cfg, sig_cfg, degraded_path, args.resume,
                     args.batch_size or model_cfg["batch_size"])

    baseline_path = out_dir / f"baseline_dev_{name}.jsonl"
    baseline = read_rows(baseline_path)
    if not baseline:
        sys.exit(f"{baseline_path} is missing or empty: run the baseline and merge_shards.py first.")
    degraded = {(r["uid"], r["degradation"]): r["p1_yes"] for r in read_rows(degraded_path)}
    missing = {r["uid"] for r in baseline} - {u for u, _ in degraded}
    if missing:
        print(f"WARNING: {len(missing)} questions have no degraded pass yet; their reliance is empty.")

    rows = signal_rows(baseline, degraded)
    sig_path = out_dir / f"signals_dev_{name}.jsonl"
    sig_path.unlink(missing_ok=True)  # rebuilt from scratch every time (CPU only, cheap)
    append_jsonl(sig_path, [{"_meta": {"stage": "signals_join", "git_commit": get_git_commit(),
                                       "signals_cfg": sig_cfg, "versions": package_versions(),
                                       "baseline_file": baseline_path.name, "degraded_file": degraded_path.name}}])
    append_jsonl(sig_path, rows)
    print(f"wrote {len(rows)} rows -> {sig_path}")

    table = auroc_table(name, rows, args.n_boot or sig_cfg["n_boot"], exp["seed"])
    csv_path = out_dir / "signals_auroc_dev.csv"
    write_auroc_csv(csv_path, name, table)
    print(f"AUROC table -> {csv_path}")
    for t in table:
        if t["pope_split"] == "all":
            ci = f"[{t['ci_low']:.3f}, {t['ci_high']:.3f}]" if t["ci_low"] is not None else ""
            if t["auroc"] is None:
                continue
            print(f"  {t['target']:<14} {t['signal']:<15} n={t['n_rows']:<5} AUROC={t['auroc']:.3f} {ci}")


if __name__ == "__main__":
    main()
