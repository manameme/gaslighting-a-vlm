"""Phase 5 gate analysis on the dev split, following results/analysis_plan.md. CPU only.

    python scripts/fit_gates_dev.py
    python scripts/fit_gates_dev.py --models qwen2vl_2b --n-boot 200     # quicker check

Reads RESULTS_DIR/signals_dev_<model>.jsonl and writes to RESULTS_DIR:
  gates_dev.csv               every gate, out-of-fold (OOF): AUROC, EIR-ECR area, and EIR/ECR/net
                              at the threshold tuned on the training folds (empirical pi)
  h2_dev.csv                  learned gate B (conf1+shift+reliance) minus A (conf1+shift), paired bootstrap
  sensitivity_pi_dev.csv      net accuracy for pi = P(user right) in {0.1, 0.3, 0.5}, thresholds re-tuned per pi
  signals_by_a1_dev.csv       signal AUROCs and means split by turn-1 answer (yes / no)
  h3_calibration_dev.csv      turn-1 calibration per POPE split
  h3_gates_by_split_dev.csv   OOF gate AUROC per POPE split
  h3_dev.csv                  the H3 decision rule per model
  gate_curves_dev.csv         OOF EIR-ECR curve points (and in-sample learned curves, marked exploratory)
  random_gate_dev.csv         random-gate bands
  reliability_dev.csv         reliability-diagram bins per POPE split
  frozen_gates.json           thresholds and learned-gate coefficients refit on all of dev, for run_final.py
All confidence intervals: 95%, bootstrap clustered by image (paired where a difference is shown).
Dev split only; the test split is never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlmsyc.gates import (  # noqa: E402
    LEARNED_GATES, SINGLE_SIGNAL_GATES, best_threshold, feature_matrix, curve_area, describe_learned_gate, eir_ecr,
    fit_learned_gate, image_folds, net_change, oracle_accept, random_gate_band, sweep,
)
from vlmsyc.paths import git_commit, load_yaml, package_versions, results_dir  # noqa: E402
from vlmsyc.pipeline import read_rows  # noqa: E402
from vlmsyc.stats import auroc, cluster_bootstrap_ci  # noqa: E402

MODELS = ["qwen2vl_2b", "llava15_7b_fp16"]
PIS = [0.1, 0.3, 0.5]
K_FOLDS = 5
SPLITS = ["random", "popular", "adversarial"]
SIGNALS = ["conf1", "reliance_blur", "reliance_blank", "shift_to_claim"]
GATES = list(SINGLE_SIGNAL_GATES) + list(LEARNED_GATES)


# ---------- loading ----------


def load(model: str) -> dict:
    rows = read_rows(results_dir() / f"signals_dev_{model}.jsonl")
    if not rows:
        sys.exit(f"No signals for {model}: run scripts/run_signals.py first.")
    if any(r["partition"] != "dev" for r in rows):
        sys.exit("signals file contains non-dev rows; refusing.")
    for s in SIGNALS:
        if any(r[s] is None for r in rows):
            sys.exit(f"{model}: some rows have no {s}; finish run_signals.py first.")
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
    return d


def features(d: dict, names: list[str]) -> np.ndarray:
    return feature_matrix(d, names)  # expands pushback_type into one-hot columns


# ---------- cross-validation ----------


def cv_gate(d: dict, folds: np.ndarray, gate: str, pis: list, seed: int):
    """OOF acceptance scores and OOF accept decisions (one per pi; None = empirical pi)."""
    n = d["n"]
    oof = np.full(n, np.nan)
    decisions = {pi: np.zeros(n, bool) for pi in pis}
    for f in range(K_FOLDS):
        tr, te = folds != f, folds == f
        if gate in SINGLE_SIGNAL_GATES:
            score = -d[SINGLE_SIGNAL_GATES[gate]]          # accept when the signal is low
            s_tr, s_te = score[tr], score[te]
        else:
            X = features(d, LEARNED_GATES[gate])
            fit_rows = tr & d["flip"]
            model = fit_learned_gate(X[fit_rows], d["good"][fit_rows], seed=seed)
            s_tr, s_te = model.predict_proba(X[tr])[:, 1], model.predict_proba(X[te])[:, 1]
        oof[te] = s_te
        for pi in pis:
            tau = best_threshold(s_tr, d["a1c"][tr], d["flip"][tr], pi=pi)
            decisions[pi][te] = s_te > tau
    return oof, decisions


def in_sample_learned(d: dict, gate: str, seed: int) -> np.ndarray:
    """Learned-gate scores fit and evaluated on all of dev (EXPLORATORY, optimistic)."""
    X = features(d, LEARNED_GATES[gate])
    model = fit_learned_gate(X[d["flip"]], d["good"][d["flip"]], seed=seed)
    return model.predict_proba(X)[:, 1]


# ---------- metric helpers ----------


def auroc_good(score, d, idx=None):
    idx = np.arange(d["n"]) if idx is None else idx
    m = d["flip"][idx]
    return auroc(score[idx][m], d["good"][idx][m])


def area(score, d, idx=None):
    idx = np.arange(d["n"]) if idx is None else idx
    _, eir, ecr = sweep(score[idx], d["a1c"][idx], d["flip"][idx])
    return curve_area(eir, ecr)


def ci(stat, d, n_boot, seed, rows=None):
    """Image-clustered bootstrap CI of stat(idx); idx indexes all rows (or `rows`)."""
    base = np.arange(d["n"]) if rows is None else np.flatnonzero(rows)
    return cluster_bootstrap_ci(lambda i: stat(base[i]), d["image"][base], n_boot=n_boot, seed=seed)


def fmt_decision(lo, hi, positive="effect", negative="opposite effect"):
    if lo is None:
        return "undefined"
    if lo > 0:
        return positive
    if hi < 0:
        return negative
    return "no clear effect (CI includes 0)"


# ---------- per-model analysis ----------


def analyse(model: str, n_boot: int, seed: int) -> dict:
    d = load(model)
    folds = image_folds(d["image"], K_FOLDS, seed)
    pis = [None] + PIS
    pi_emp = float((~d["a1c"]).mean())
    out = {"gates": [], "sens": [], "curves": [], "h2": [], "by_a1": [], "calib": [], "h3_split": [],
           "h3": [], "rel": [], "random": [], "frozen": {}}
    print(f"\n=== {model}: {d['n']} rows, {d['flip'].sum()} flips, {d['good'].sum()} good revisions, "
          f"empirical pi = {pi_emp:.3f} ===", flush=True)

    oof, dec = {}, {}
    for g in GATES:
        oof[g], dec[g] = cv_gate(d, folds, g, pis, seed)

    # Reference gates.
    refs = {
        "always_accept": np.ones(d["n"], bool),
        "never_accept": np.zeros(d["n"], bool),
        "oracle": oracle_accept(d["a1c"], d["flip"]),
    }
    eir_max, ecr_max = eir_ecr(d["a1c"], d["flip"], refs["always_accept"])

    def net_stat(accept, pi):
        return lambda idx: net_change(*eir_ecr(d["a1c"][idx], d["flip"][idx], accept[idx]), pi)

    # ---- gates_dev.csv ----
    for name, acc in refs.items():
        eir, ecr = eir_ecr(d["a1c"], d["flip"], acc)
        lo, hi = ci(net_stat(acc, pi_emp), d, n_boot, seed)
        out["gates"].append({
            "model": model, "gate": name, "evaluation": "reference", "auroc_good_revision": None,
            "auroc_ci_low": None, "auroc_ci_high": None, "area": None, "area_normalised": None,
            "area_ci_low": None, "area_ci_high": None,
            "accept_rate_among_flips": float(acc[d["flip"]].mean()), "eir": eir, "ecr": ecr,
            "pi_empirical": pi_emp, "net_change": net_change(eir, ecr, pi_emp),
            "net_ci_low": lo, "net_ci_high": hi,
        })
    for g in GATES:
        s = oof[g]
        a = auroc_good(s, d)
        a_lo, a_hi = ci(lambda i: auroc_good(s, d, i), d, n_boot, seed)
        ar, arn = area(s, d)
        ar_lo, ar_hi = ci(lambda i: area(s, d, i)[0], d, n_boot, seed)
        acc = dec[g][None]
        eir, ecr = eir_ecr(d["a1c"], d["flip"], acc)
        lo, hi = ci(net_stat(acc, pi_emp), d, n_boot, seed)
        out["gates"].append({
            "model": model, "gate": g, "evaluation": "out-of-fold (5-fold grouped by image)",
            "auroc_good_revision": a, "auroc_ci_low": a_lo, "auroc_ci_high": a_hi,
            "area": ar, "area_normalised": arn, "area_ci_low": ar_lo, "area_ci_high": ar_hi,
            "accept_rate_among_flips": float(acc[d["flip"]].mean()), "eir": eir, "ecr": ecr,
            "pi_empirical": pi_emp, "net_change": net_change(eir, ecr, pi_emp),
            "net_ci_low": lo, "net_ci_high": hi,
        })
        print(f"  {g:<32} OOF AUROC {a:.3f} [{a_lo:.3f}, {a_hi:.3f}]  EIR {eir:.3f} ECR {ecr:.3f} "
              f"net {net_change(eir, ecr, pi_emp):+.3f}", flush=True)
        th, ce, cc = sweep(s, d["a1c"], d["flip"])
        out["curves"] += [{"model": model, "gate": g, "evaluation": "OOF", "eir": e, "ecr": c}
                          for e, c in zip(ce, cc)]
    for g in LEARNED_GATES:  # exploratory in-sample curves
        s = in_sample_learned(d, g, seed)
        th, ce, cc = sweep(s, d["a1c"], d["flip"])
        out["curves"] += [{"model": model, "gate": g, "evaluation": "in-sample (EXPLORATORY)", "eir": e, "ecr": c}
                          for e, c in zip(ce, cc)]
        out["gates"].append({
            "model": model, "gate": g, "evaluation": "in-sample (EXPLORATORY)",
            "auroc_good_revision": auroc_good(s, d), "area": area(s, d)[0], "area_normalised": area(s, d)[1],
        })
    out["random"] = [{"model": model, **r} for r in
                     random_gate_band(d["a1c"], d["flip"], np.linspace(0, 1, 21), n_seeds=100, seed=seed)]
    out["oracle_ceiling"] = {"eir_max": eir_max, "ecr_max": ecr_max}

    # ---- H2: B minus A, paired ----
    A, B = "learned_A_conf_shift", "learned_B_conf_shift_reliance"
    for metric, fn in [
        ("auroc_good_revision", lambda s, i: auroc_good(s, d, i)),
        ("area", lambda s, i: area(s, d, i)[0]),
        ("area_normalised", lambda s, i: area(s, d, i)[1]),
    ]:
        va, vb = fn(oof[A], None), fn(oof[B], None)
        lo, hi = ci(lambda i: fn(oof[B], i) - fn(oof[A], i), d, n_boot, seed)
        out["h2"].append({
            "model": model, "metric": metric, "gate_A_conf_shift": va, "gate_B_plus_reliance": vb,
            "difference_B_minus_A": vb - va, "ci_low": lo, "ci_high": hi, "n_boot": n_boot,
            "decision": fmt_decision(lo, hi, "reliance adds information", "adding reliance hurts"),
        })

    # ---- sensitivity to pi ----
    for pi in PIS:
        for name, acc in refs.items():
            eir, ecr = eir_ecr(d["a1c"], d["flip"], acc)
            out["sens"].append({"model": model, "gate": name, "pi": pi, "eir": eir, "ecr": ecr,
                                "accept_rate_among_flips": float(acc[d["flip"]].mean()),
                                "final_accuracy": (1 - pi) * (1 - eir) + pi * ecr,
                                "net_change": net_change(eir, ecr, pi), "thresholds": "n/a"})
        for g in GATES:
            acc = dec[g][pi]
            eir, ecr = eir_ecr(d["a1c"], d["flip"], acc)
            lo, hi = ci(net_stat(acc, pi), d, n_boot, seed)
            out["sens"].append({"model": model, "gate": g, "pi": pi, "eir": eir, "ecr": ecr,
                                "accept_rate_among_flips": float(acc[d["flip"]].mean()),
                                "final_accuracy": (1 - pi) * (1 - eir) + pi * ecr,
                                "net_change": net_change(eir, ecr, pi), "net_ci_low": lo, "net_ci_high": hi,
                                "thresholds": "re-tuned on training folds for this pi (OOF)"})

    # ---- by turn-1 answer ----
    for a1 in ("yes", "no"):
        sel = d["a1"] == a1
        for target, sigs in (("flip", ["conf1", "reliance_blur", "reliance_blank"]),
                             ("good_revision", SIGNALS)):
            rows_sel = sel if target == "flip" else sel & d["flip"]
            y = d["flip"] if target == "flip" else d["good"]
            for s in sigs:
                val = d[s]
                stat = (lambda v, yy: lambda i: auroc(v[i], yy[i]))(val, y)
                idx = np.flatnonzero(rows_sel)
                a = auroc(val[idx], y[idx])
                lo, hi = ci(stat, d, n_boot, seed, rows=rows_sel)
                m_lo, m_hi = ci(lambda i: float(val[i].mean()), d, n_boot, seed, rows=rows_sel)
                out["by_a1"].append({
                    "model": model, "a1": a1, "target": target, "signal": s, "orientation": "raw signal",
                    "n_rows": int(rows_sel.sum()), "n_positive": int(y[rows_sel].sum()),
                    "auroc": a, "ci_low": lo, "ci_high": hi,
                    "mean_signal": float(val[idx].mean()), "mean_ci_low": m_lo, "mean_ci_high": m_hi,
                })

    # ---- H3: calibration and gates per POPE split ----
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
        out["calib"].append(row)
        for b in reliability_bins(conf, cor):
            out["rel"].append({"model": model, "pope_split": sp, **b})
        sel = d["split"] == sp
        for g in GATES:
            idx = np.flatnonzero(sel)
            a = auroc_good(oof[g], d, idx)
            lo, hi = ci(lambda i: auroc_good(oof[g], d, i), d, n_boot, seed, rows=sel)
            out["h3_split"].append({"model": model, "pope_split": sp, "gate": g,
                                    "n_flips": int(d["flip"][sel].sum()), "n_good": int(d["good"][sel].sum()),
                                    "auroc_good_revision_oof": a, "ci_low": lo, "ci_high": hi})

    adv, rnd = d["split"] == "adversarial", d["split"] == "random"

    def split_auc(g, i, which):
        return auroc_good(oof[g], d, i[which[i]])

    conf_drop = lambda i: split_auc("conf_gate", i, adv) - split_auc("conf_gate", i, rnd)  # noqa: E731
    blur_drop = lambda i: split_auc("reliance_blur_gate", i, adv) - split_auc("reliance_blur_gate", i, rnd)  # noqa: E731
    all_idx = np.arange(d["n"])
    c_val, c_lo, c_hi = conf_drop(all_idx), *ci(conf_drop, d, n_boot, seed)
    did = lambda i: blur_drop(i) - conf_drop(i)  # noqa: E731
    dd_val, dd_lo, dd_hi = did(all_idx), *ci(did, d, n_boot, seed)
    cond1 = c_hi is not None and c_hi < 0
    cond2 = dd_lo is not None and dd_lo > 0
    out["h3"].append({
        "model": model,
        "conf_gate_auroc_adv_minus_random": c_val, "ci_low_1": c_lo, "ci_high_1": c_hi,
        "condition_1_conf_degrades_on_adversarial": cond1,
        "did_blur_minus_conf": dd_val, "ci_low_2": dd_lo, "ci_high_2": dd_hi,
        "condition_2_reliance_holds_up_better": cond2,
        "h3_supported": cond1 and cond2,
        "verdict": "supported" if (cond1 and cond2) else "NOT supported (negative / inconclusive result)",
    })

    # ---- frozen gates: refit on all of dev ----
    frozen = {}
    for g in GATES:
        if g in SINGLE_SIGNAL_GATES:
            score = -d[SINGLE_SIGNAL_GATES[g]]
            entry = {"type": "single_signal", "signal": SINGLE_SIGNAL_GATES[g],
                     "rule": f"accept a flip iff -{SINGLE_SIGNAL_GATES[g]} > threshold "
                             f"(i.e. {SINGLE_SIGNAL_GATES[g]} < -threshold)"}
        else:
            X = features(d, LEARNED_GATES[g])
            m = fit_learned_gate(X[d["flip"]], d["good"][d["flip"]], seed=seed)
            score = m.predict_proba(X)[:, 1]
            entry = {"type": "learned_logistic", **describe_learned_gate(m, LEARNED_GATES[g]),
                     "rule": "accept a flip iff P(good revision) > threshold"}
        entry["thresholds"] = {"empirical_pi": best_threshold(score, d["a1c"], d["flip"], None)}
        entry["thresholds"].update({f"pi_{pi}": best_threshold(score, d["a1c"], d["flip"], pi) for pi in PIS})
        frozen[g] = entry
    out["frozen"] = {"empirical_pi_dev": pi_emp, "n_rows": d["n"], "n_flips": int(d["flip"].sum()), "gates": frozen}
    return out


def ece(conf, correct, n_bins: int = 10):
    """Expected calibration error with equal-count bins (questions sorted by confidence)."""
    if len(conf) == 0:
        return None
    order = np.argsort(conf, kind="mergesort")
    total = 0.0
    for b in np.array_split(order, min(n_bins, len(conf))):
        total += len(b) / len(conf) * abs(correct[b].mean() - conf[b].mean())
    return float(total)


def reliability_bins(conf, correct, n_bins: int = 10):
    order = np.argsort(conf, kind="mergesort")
    return [{"bin": k, "n": len(b), "mean_conf1": float(conf[b].mean()), "accuracy": float(correct[b].mean())}
            for k, b in enumerate(np.array_split(order, n_bins))]


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--experiment", default="configs/experiment.yaml")
    parser.add_argument("--n-boot", type=int)
    args = parser.parse_args()
    exp = load_yaml(args.experiment)
    n_boot = args.n_boot or exp["signals"]["n_boot"]
    seed = exp["seed"]
    # Read the commit before writing anything, or our own new files mark the tree dirty.
    commit = git_commit()

    results = [analyse(m, n_boot, seed) for m in args.models]
    out = results_dir()
    for key, name in [("gates", "gates_dev.csv"), ("h2", "h2_dev.csv"), ("sens", "sensitivity_pi_dev.csv"),
                      ("by_a1", "signals_by_a1_dev.csv"), ("calib", "h3_calibration_dev.csv"),
                      ("h3_split", "h3_gates_by_split_dev.csv"), ("h3", "h3_dev.csv"),
                      ("curves", "gate_curves_dev.csv"), ("random", "random_gate_dev.csv"),
                      ("rel", "reliability_dev.csv")]:
        write_csv(out / name, [r for res in results for r in res[key]])

    # H2 overall verdict (analysis_plan.md section 5).
    primary = [r for res in results for r in res["h2"] if r["metric"] == "auroc_good_revision"]
    n_pos = sum(r["decision"] == "reliance adds information" for r in primary)
    overall = "supported" if n_pos == len(primary) else ("mixed" if n_pos else "not supported (negative result)")
    print(f"\nH2 overall (primary metric = OOF AUROC difference): {overall}")
    for r in primary:
        print(f"  {r['model']}: B-A = {r['difference_B_minus_A']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] -> {r['decision']}")
    for res in results:
        for r in res["h3"]:
            print(f"H3 {r['model']}: {r['verdict']}")

    frozen = {
        "_note": "Fit on the dev split only (scripts/fit_gates_dev.py). Apply unchanged to test in run_final.py --final.",
        "git_commit": commit, "seed": seed, "k_folds": K_FOLDS, "n_boot": n_boot,
        "versions": package_versions(), "analysis_plan": "results/analysis_plan.md",
        "h2_overall": overall,
        "models": {res_model: res["frozen"] for res_model, res in zip(args.models, results)},
        "oracle_ceiling": {res_model: res["oracle_ceiling"] for res_model, res in zip(args.models, results)},
    }
    (out / "frozen_gates.json").write_text(json.dumps(frozen, indent=2))
    print(f"wrote {out / 'frozen_gates.json'}")


if __name__ == "__main__":
    main()
