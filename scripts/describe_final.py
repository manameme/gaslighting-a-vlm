"""Descriptive analyses of saved outputs (results/analysis_plan.md, Amendment C). CPU only.

    python scripts/describe_final.py --partition dev --check-dev   # must equal the existing dev tables
    python scripts/describe_final.py --partition test              # only after job B's analysis exists

Reads only saved outputs (baseline_<p>_<model>.jsonl, signals_<p>_<model>.jsonl); runs no model.
Writes to results/describe_<p>/ (never overwritten for test):
  flip_summary.csv            C1: metrics.flip_summary overall, by pushback type, by POPE split
  flip_auroc_by_a1.csv        C2 (primary): conf1 / blur / blank AUROC for flips within a1 = yes and a1 = no
  flip_auroc_pooled_context.csv  C2 (context only): the same pooled over answers (run_signals.auroc_table)
  calibration.csv, reliability.csv   C3: turn-1 calibration per POPE split
  h1.csv                      C4: the H1 verdict per model
The dev code is reused unchanged (metrics.flip_summary, run_signals.auroc_table,
fit_gates_dev.ci / ece / reliability_bins); the by-answer and calibration loops below
mirror fit_gates_dev.py line by line, and --check-dev proves the outputs are identical.
shift_to_claim's AUROC for flips is never reported (circular).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_gates_dev import SPLITS, ci, ece, reliability_bins  # noqa: E402  (unchanged dev functions)
from run_signals import auroc_table  # noqa: E402  (unchanged dev function)
from vlmsyc.metrics import flip_summary  # noqa: E402
from vlmsyc.paths import load_yaml, results_dir  # noqa: E402
from vlmsyc.pipeline import read_rows  # noqa: E402
from vlmsyc.stats import auroc  # noqa: E402

MODELS = ["qwen2vl_2b", "llava15_7b_fp16"]
FLIP_SIGNALS = ["conf1", "reliance_blur", "reliance_blank"]   # no shift_to_claim (circular for flips)
SIGNALS = ["conf1", "reliance_blur", "reliance_blank", "shift_to_claim"]


def load(path: Path, partition: str) -> tuple[dict, list[dict]]:
    """Same arrays as fit_gates_dev.load, for any partition's saved signals file."""
    rows = read_rows(path)
    if not rows:
        sys.exit(f"missing or empty: {path}")
    if any(r["partition"] != partition for r in rows):
        sys.exit(f"{path} contains rows from another partition; refusing.")
    d = {
        "uid": np.array([r["uid"] for r in rows]),
        "image": np.array([r["image_id"] for r in rows]),
        "split": np.array([r["pope_split"] for r in rows]),
        "a1": np.array([r["a1"] for r in rows]),
        "pushback_type": np.array([r["pushback_type"] for r in rows]),
        "a1c": np.array([r["a1_correct"] for r in rows], bool),
        "flip": np.array([r["flip"] for r in rows], bool),
        "n": len(rows),
    }
    d["good"] = d["flip"] & ~d["a1c"]
    for s in SIGNALS:
        d[s] = np.array([r[s] for r in rows], float)
    return d, rows


def flip_by_a1(model: str, d: dict, n_boot: int, seed: int) -> list[dict]:
    """C2 primary. Mirrors fit_gates_dev.py '# ---- by turn-1 answer ----' (flip target only)."""
    out = []
    for a1 in ("yes", "no"):
        sel = d["a1"] == a1
        for target, sigs in (("flip", FLIP_SIGNALS),):
            rows_sel = sel
            y = d["flip"]
            for s in sigs:
                val = d[s]
                stat = (lambda v, yy: lambda i: auroc(v[i], yy[i]))(val, y)
                idx = np.flatnonzero(rows_sel)
                a = auroc(val[idx], y[idx])
                lo, hi = ci(stat, d, n_boot, seed, rows=rows_sel)
                m_lo, m_hi = ci(lambda i: float(val[i].mean()), d, n_boot, seed, rows=rows_sel)
                out.append({
                    "model": model, "a1": a1, "target": target, "signal": s, "orientation": "raw signal",
                    "n_rows": int(rows_sel.sum()), "n_positive": int(y[rows_sel].sum()),
                    "auroc": a, "ci_low": lo, "ci_high": hi,
                    "mean_signal": float(val[idx].mean()), "mean_ci_low": m_lo, "mean_ci_high": m_hi,
                })
    return out


def calibration(model: str, d: dict, n_boot: int, seed: int) -> tuple[list[dict], list[dict]]:
    """C3. Mirrors fit_gates_dev.py '# ---- H3: calibration ... ----' (calibration part)."""
    calib, rel = [], []
    first = {}
    for i, u in enumerate(d["uid"]):
        first.setdefault(u, i)
    q = np.array(sorted(first.values()))   # one row per question
    for sp in SPLITS:
        qs = q[d["split"][q] == sp]
        conf, cor = d["conf1"][qs], d["a1c"][qs]
        stats = {
            "turn1_accuracy": lambda i: float(d["a1c"][i].mean()),
            "ece_10_equal_count_bins": lambda i: ece(d["conf1"][i], d["a1c"][i]),
            "mean_conf1_when_wrong": lambda i: float(d["conf1"][i][~d["a1c"][i]].mean()) if (~d["a1c"][i]).any() else None,
            "share_wrong_with_conf1_over_0.9": lambda i: float((d["conf1"][i][~d["a1c"][i]] > 0.9).mean()) if (~d["a1c"][i]).any() else None,
        }
        mask = np.zeros(d["n"], bool)
        mask[qs] = True
        row = {"model": model, "pope_split": sp, "n_questions": len(qs), "n_wrong": int((~cor).sum())}
        for k, fn in stats.items():
            row[k] = fn(qs)
            row[k + "_ci_low"], row[k + "_ci_high"] = ci(fn, d, n_boot, seed, rows=mask)
        calib.append(row)
        for b in reliability_bins(conf, cor):
            rel.append({"model": model, "pope_split": sp, **b})
    return calib, rel


def flip_summaries(model: str, rows: list[dict]) -> list[dict]:
    """C1 with the unchanged metrics.flip_summary."""
    out = [{"model": model, "group": "all", "value": "all", **flip_summary(rows)}]
    for key in ("pushback_type", "pope_split"):
        for v in sorted({r[key] for r in rows}):
            out.append({"model": model, "group": key, "value": v,
                        **flip_summary([r for r in rows if r[key] == v])})
    return out


def h1_verdict(model: str, by_a1: list[dict]) -> dict:
    """C4: conf1 flip AUROC CI entirely below 0.5 within both answers / one / neither."""
    below = {}
    for a1 in ("yes", "no"):
        r = next(x for x in by_a1 if x["model"] == model and x["a1"] == a1 and x["signal"] == "conf1")
        below[a1] = r["ci_high"] is not None and r["ci_high"] < 0.5
    n = sum(below.values())
    verdict = "supported" if n == 2 else ("mixed" if n == 1 else "not supported")
    return {"model": model, "ci_below_0.5_within_a1_yes": below["yes"], "ci_below_0.5_within_a1_no": below["no"],
            "h1_verdict": verdict}


def write_csv(path: Path, rows: list[dict], overwrite_ok: bool) -> None:
    if path.exists() and not overwrite_ok:
        raise FileExistsError(f"{path} exists; never overwritten.")
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def check_dev(out_dir: Path) -> bool:
    """Compare the dev outputs with the existing dev tables, cell by cell (as written to CSV)."""
    def read(p):
        with open(p) as f:
            return list(csv.DictReader(f))

    def same(name, ours, theirs, cols):
        a = [{c: r[c] for c in cols} for r in ours]
        b = [{c: r[c] for c in cols} for r in theirs]
        ok = a == b
        print(f"  {name:<44} {len(a):>3} rows vs {len(b):>3}: {'IDENTICAL' if ok else 'DIFFERENT'}")
        return ok

    R = results_dir()
    ok = True
    by_a1 = read(out_dir / "flip_auroc_by_a1.csv")
    ok &= same("flip_auroc_by_a1 vs signals_by_a1_dev (flip rows)", by_a1,
               [r for r in read(R / "signals_by_a1_dev.csv") if r["target"] == "flip"], list(by_a1[0]))
    pooled = read(out_dir / "flip_auroc_pooled_context.csv")
    dev_flip = [r for r in read(R / "signals_auroc_dev.csv") if r["target"] == "flip"]
    ok &= same("flip_auroc_pooled vs signals_auroc_dev (conf1/blur/blank flip rows)", pooled,
               [r for r in dev_flip if r["signal"] in FLIP_SIGNALS], list(pooled[0]))
    circular = [r for r in dev_flip if r["signal"] not in FLIP_SIGNALS]
    if circular:  # reported, not hidden: these rows should not be in the dev table at all
        print(f"  NOTE: results/signals_auroc_dev.csv also holds {len(circular)} circular flip rows "
              f"({sorted({r['signal'] for r in circular})}); they are not compared and must not be reported.")
    cal = read(out_dir / "calibration.csv")
    ok &= same("calibration vs h3_calibration_dev", cal, read(R / "h3_calibration_dev.csv"), list(cal[0]))
    rel = read(out_dir / "reliability.csv")
    ok &= same("reliability vs reliability_dev", rel, read(R / "reliability_dev.csv"), list(rel[0]))
    print("  flip_summary: no earlier dev table exists to compare with (see overview section 5.2)")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--partition", required=True, choices=["dev", "test"])
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--check-dev", action="store_true", help="dev only: compare with the existing dev tables")
    args = parser.parse_args()
    exp = load_yaml("configs/experiment.yaml")
    n_boot, seed = exp["signals"]["n_boot"], exp["seed"]
    R = results_dir()
    p = args.partition
    out_dir = R / f"describe_{p}"
    overwrite_ok = p == "dev"
    if p != "dev":
        if not (R / "final_test" / "run_info.json").exists():
            sys.exit("results/final_test/run_info.json not found: wait until job B's analysis has finished.")
        if out_dir.exists():
            sys.exit(f"{out_dir} exists; never overwritten.")
    out_dir.mkdir(parents=True, exist_ok=True)

    flips, by_a1, pooled, calib, rel, h1 = [], [], [], [], [], []
    for m in args.models:
        d, rows = load(R / f"signals_{p}_{m}.jsonl", p)
        base = read_rows(R / f"baseline_{p}_{m}.jsonl")
        flips += flip_summaries(m, base)
        by_a1 += flip_by_a1(m, d, n_boot, seed)
        pooled += [r for r in auroc_table(m, rows, n_boot, seed) if r["target"] == "flip"]
        c, r_ = calibration(m, d, n_boot, seed)
        calib += c
        rel += r_
        h1.append(h1_verdict(m, by_a1))
        print(f"{m}: H1 (Amendment C4) = {h1[-1]['h1_verdict']}")
    for name, rows_ in [("flip_summary.csv", flips), ("flip_auroc_by_a1.csv", by_a1),
                        ("flip_auroc_pooled_context.csv", pooled), ("calibration.csv", calib),
                        ("reliability.csv", rel), ("h1.csv", h1)]:
        write_csv(out_dir / name, rows_, overwrite_ok)
    if args.check_dev:
        if p != "dev":
            sys.exit("--check-dev is for the dev partition")
        print("dev-equality check:")
        if not check_dev(out_dir):
            sys.exit("FAILED: outputs differ from the existing dev tables")
        print("PASSED: identical to the existing dev tables")


if __name__ == "__main__":
    main()
