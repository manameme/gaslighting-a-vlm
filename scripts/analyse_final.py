"""Final analysis: apply the FROZEN gates (results/frozen_gates.json) to a split. CPU only.

    python scripts/analyse_final.py --partition dev        # dry run (in-sample, NOT a result)
    python scripts/analyse_final.py --partition test --final   # called by run_final.py --final

Follows results/analysis_plan.md, Amendment A. Nothing is fitted or tuned here:
learned gates use the saved scaler and coefficients, thresholds are the saved ones.

Inputs (from --in-dir, default RESULTS_DIR):
  signals_<partition>_<model>.jsonl, baseline_<partition>_<model>.jsonl,
  grounding_<partition>_<model>.jsonl and relook_<partition>_<model>.jsonl (if present).
Outputs (to --out-dir; never overwritten for test):
  gates.csv        headline + secondary gates at the native-pi threshold and at pi 0.1/0.3/0.5:
                   EIR, ECR, net change (= vs never-accept) and net minus matched random, with CIs
  references.csv   always-accept, never-accept, oracle at each pi
  auroc.csv        threshold-free AUROC (good revision) and EIR-ECR area per gate, with CIs
  h2.csv           frozen gate B minus frozen gate A (paired bootstrap) + decision (Amendment A4)
  h3.csv           per-POPE-split AUROC of the headline gates + the section 8 rule (secondary)
  mitigations.csv  grounding / re-look with every revision accepted, paired to base (secondary)
  by_a1.csv        signal AUROCs split by turn-1 answer (descriptive)
Figures (to --fig-dir): eir_ecr.png, sensitivity_pi.png.
Every CSV row carries a 'status' column saying whether it is a dry run or the final result.
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
    HEADLINE_GATES, LEARNED_GATES, SINGLE_SIGNAL_GATES, apply_frozen_learned_gate, ci_side, curve_area, eir_ecr,
    overall_h2, overall_h3,
    feature_matrix, net_change, oracle_accept, random_gate_band, sweep,
)
from vlmsyc.paths import REPO_ROOT, load_yaml, results_dir  # noqa: E402
from vlmsyc.pipeline import read_rows  # noqa: E402
from vlmsyc.release import check_release, sha256_file  # noqa: E402
from vlmsyc.stats import auroc, cluster_bootstrap_ci  # noqa: E402

PIS = [0.1, 0.3, 0.5]
SPLITS = ["random", "popular", "adversarial"]
MITIGATIONS = ["grounding", "relook"]   # no-suffix stays dev-only (Amendment A5)
SIGNALS = ["conf1", "reliance_blur", "reliance_blank", "shift_to_claim"]


def load_signals(path: Path) -> dict:
    rows = read_rows(path)
    if not rows:
        sys.exit(f"missing or empty: {path}")
    d = {k: np.array([r[k] for r in rows]) for k in ("uid", "image_id", "pope_split", "a1", "pushback_type")}
    d["a1c"] = np.array([r["a1_correct"] for r in rows], bool)
    d["flip"] = np.array([r["flip"] for r in rows], bool)
    d["good"] = d["flip"] & ~d["a1c"]
    for s in SIGNALS:
        d[s] = np.array([r[s] for r in rows], float)
    d["n"] = len(rows)
    d["partition"] = {r["partition"] for r in rows}
    return d


def gate_score(d: dict, name: str, entry: dict) -> np.ndarray:
    """Acceptance score from the frozen definition (higher = accept)."""
    if entry["type"] == "single_signal":
        return -d[entry["signal"]]
    return apply_frozen_learned_gate(entry, feature_matrix(d, entry["features"]))


class Boot:
    """Image-clustered bootstrap with one fixed seed, so every statistic uses the same
    resamples (paired comparisons are paired automatically)."""

    def __init__(self, d, n_boot, seed):
        self.d, self.n_boot, self.seed = d, n_boot, seed

    def ci(self, stat, rows=None):
        base = np.arange(self.d["n"]) if rows is None else np.flatnonzero(rows)
        return cluster_bootstrap_ci(lambda i: stat(base[i]), self.d["image_id"][base],
                                    n_boot=self.n_boot, seed=self.seed)


def analyse_model(model: str, d: dict, frozen: dict, in_dir: Path, partition: str, n_boot: int, seed: int,
                  status: str) -> dict:
    gates = frozen["models"][model]["gates"]
    boot = Boot(d, n_boot, seed)
    pi_native = float((~d["a1c"]).mean())
    all_idx = np.arange(d["n"])
    eir_max, ecr_max = eir_ecr(d["a1c"], d["flip"], np.ones(d["n"], bool))
    out = {k: [] for k in ("gates", "refs", "auroc", "h2", "h3", "mit", "by_a1", "curves", "points")}
    base_row = {"status": status, "model": model, "partition": partition}

    def ee(acc, idx):
        return eir_ecr(d["a1c"][idx], d["flip"][idx], acc[idx])

    scores = {g: gate_score(d, g, gates[g]) for g in gates}
    settings = [("native_pi", "empirical_pi", pi_native)] + [(f"pi_{p}", f"pi_{p}", p) for p in PIS]

    # ---- references ----
    refs = {"always_accept": np.ones(d["n"], bool), "never_accept": np.zeros(d["n"], bool),
            "oracle": oracle_accept(d["a1c"], d["flip"])}
    for label, _, pi in settings:
        for name, acc in refs.items():
            eir, ecr = ee(acc, all_idx)
            out["refs"].append({**base_row, "reference": name, "operating_point": label, "pi": pi,
                                "eir": eir, "ecr": ecr, "net_change": net_change(eir, ecr, pi),
                                "accept_rate_among_flips": float(acc[d["flip"]].mean())})

    # ---- gates at frozen thresholds ----
    for g, entry in gates.items():
        s = scores[g]
        for label, key, pi in settings:
            tau = entry["thresholds"][key]
            acc = s > tau
            eir, ecr = ee(acc, all_idx)
            rate = float(acc[d["flip"]].mean())

            def net_stat(idx, acc=acc, pi=pi):
                return net_change(*ee(acc, idx), pi)

            def vs_random(idx, acc=acc, pi=pi):
                e, c = ee(acc, idx)
                full_e, full_c = ee(np.ones(d["n"], bool), idx)
                r = float(acc[idx][d["flip"][idx]].mean()) if d["flip"][idx].any() else 0.0
                return net_change(e, c, pi) - net_change(r * full_e, r * full_c, pi)

            n_lo, n_hi = boot.ci(net_stat)
            r_lo, r_hi = boot.ci(vs_random)
            out["gates"].append({
                **base_row, "gate": g, "headline": g in HEADLINE_GATES, "operating_point": label, "pi": pi,
                "threshold": tau, "accept_rate_among_flips": rate, "eir": eir, "ecr": ecr,
                "net_change": net_change(eir, ecr, pi), "net_ci_low": n_lo, "net_ci_high": n_hi,
                "beats_never_accept": (n_lo is not None and n_lo > 0),
                "matched_random_eir": rate * eir_max, "matched_random_ecr": rate * ecr_max,
                "net_minus_matched_random": vs_random(all_idx), "vs_random_ci_low": r_lo, "vs_random_ci_high": r_hi,
            })
            if label == "native_pi":
                out["points"].append({"model": model, "gate": g, "eir": eir, "ecr": ecr,
                                      "random_eir": rate * eir_max, "random_ecr": rate * ecr_max})

        # threshold-free
        def a_stat(idx, s=s):
            m = d["flip"][idx]
            return auroc(s[idx][m], d["good"][idx][m])

        def area_stat(idx, s=s):
            return curve_area(*sweep(s[idx], d["a1c"][idx], d["flip"][idx])[1:])[0]

        a_lo, a_hi = boot.ci(a_stat)
        ar_lo, ar_hi = boot.ci(area_stat)
        _, ce, cc = sweep(s, d["a1c"], d["flip"])
        area, area_n = curve_area(ce, cc)
        out["auroc"].append({**base_row, "gate": g, "headline": g in HEADLINE_GATES,
                             "auroc_good_revision": a_stat(all_idx), "ci_low": a_lo, "ci_high": a_hi,
                             "area": area, "area_normalised": area_n, "area_ci_low": ar_lo, "area_ci_high": ar_hi})
        out["curves"] += [{"model": model, "gate": g, "eir": e, "ecr": c} for e, c in zip(ce, cc)]

    # ---- H2 (Amendment A4) ----
    A, B = "learned_A_conf_shift", "learned_B_conf_shift_reliance"
    for metric in ("auroc_good_revision", "area"):
        def stat(idx, g, metric=metric):
            s = scores[g]
            if metric == "auroc_good_revision":
                m = d["flip"][idx]
                return auroc(s[idx][m], d["good"][idx][m])
            return curve_area(*sweep(s[idx], d["a1c"][idx], d["flip"][idx])[1:])[0]
        def sub(a, b):
            return None if a is None or b is None else a - b

        va, vb = stat(all_idx, A), stat(all_idx, B)
        lo, hi = boot.ci(lambda i: sub(stat(i, B), stat(i, A)))
        decision = ("reliance adds information" if lo is not None and lo > 0 else
                    "adding reliance hurts" if hi is not None and hi < 0 else "no clear effect (CI includes 0)")
        out["h2"].append({**base_row, "metric": metric, "gate_A": va, "gate_B": vb, "difference_B_minus_A": sub(vb, va),
                          "ci_low": lo, "ci_high": hi, "decision": decision, "primary": metric == "auroc_good_revision"})

    # ---- H3 (secondary, section 8 rule with frozen gates) ----
    def split_auc(g, idx, sp):
        sel = idx[d["pope_split"][idx] == sp]
        m = d["flip"][sel]
        return auroc(scores[g][sel][m], d["good"][sel][m])
    for g in HEADLINE_GATES + ["reliance_blur_gate"]:
        for sp in SPLITS:
            lo, hi = boot.ci(lambda i, g=g, sp=sp: split_auc(g, i, sp))
            out["h3"].append({**base_row, "row": "split_auroc", "gate": g, "pope_split": sp,
                              "value": split_auc(g, all_idx, sp), "ci_low": lo, "ci_high": hi})
    def sub(a, b):  # None when either side is undefined (e.g. no good revision in a resample)
        return None if a is None or b is None else a - b

    conf_drop = lambda i: sub(split_auc("conf_gate", i, "adversarial"), split_auc("conf_gate", i, "random"))  # noqa: E731
    did = lambda i: sub(sub(split_auc("reliance_blur_gate", i, "adversarial"),  # noqa: E731
                            split_auc("reliance_blur_gate", i, "random")), conf_drop(i))
    c_lo, c_hi = boot.ci(conf_drop)
    dd_lo, dd_hi = boot.ci(did)
    supported = (c_hi is not None and c_hi < 0) and (dd_lo is not None and dd_lo > 0)
    out["h3"].append({**base_row, "row": "conf_gate_adv_minus_random", "value": conf_drop(all_idx), "ci_low": c_lo, "ci_high": c_hi})
    out["h3"].append({**base_row, "row": "did_blur_minus_conf", "value": did(all_idx), "ci_low": dd_lo, "ci_high": dd_hi})
    out["h3"].append({**base_row, "row": "verdict", "gate": "supported" if supported else "NOT supported"})

    # ---- by turn-1 answer (descriptive) ----
    for a1 in ("yes", "no"):
        for target, sigs in (("flip", SIGNALS[:3]), ("good_revision", SIGNALS)):
            rows_sel = (d["a1"] == a1) if target == "flip" else (d["a1"] == a1) & d["flip"]
            y = d["flip"] if target == "flip" else d["good"]
            for sname in sigs:
                v = d[sname]
                idx = np.flatnonzero(rows_sel)
                lo, hi = boot.ci(lambda i, v=v, y=y: auroc(v[i], y[i]), rows=rows_sel)
                out["by_a1"].append({**base_row, "a1": a1, "target": target, "signal": sname, "n_rows": len(idx),
                                     "auroc_raw_signal": auroc(v[idx], y[idx]), "ci_low": lo, "ci_high": hi})

    # ---- prompt mitigations (secondary) ----
    base_rows = {(r["uid"], r["pushback_type"], r["paraphrase_id"]): r
                 for r in read_rows(in_dir / f"baseline_{partition}_{model}.jsonl")}
    for cond in MITIGATIONS:
        crow = {(r["uid"], r["pushback_type"], r["paraphrase_id"]): r
                for r in read_rows(in_dir / f"{cond}_{partition}_{model}.jsonl")}
        if not crow:
            out["mit"].append({**base_row, "condition": cond, "note": "file missing"})
            continue
        keys = sorted(set(crow) & set(base_rows))
        arr = lambda rows, k: np.array([rows[x][k] for x in keys])  # noqa: E731
        c1, cf, c2 = arr(crow, "a1_correct").astype(bool), arr(crow, "flip").astype(bool), arr(crow, "a2_correct").astype(bool)
        b1, bf, b2 = arr(base_rows, "a1_correct").astype(bool), arr(base_rows, "flip").astype(bool), arr(base_rows, "a2_correct").astype(bool)
        img = np.array([crow[k]["image_id"] for k in keys])
        ones = np.ones(len(keys), bool)
        e, c = eir_ecr(c1, cf, ones)
        row = {**base_row, "condition": cond, "n_rows": len(keys), "rows_missing_vs_base": len(base_rows) - len(keys),
               "turn1_acc": float(c1.mean()), "flip_rate": float(cf.mean()), "eir": e, "ecr": c,
               "turn2_acc": float(c2.mean()), "mean_mass_t2": float(np.mean(arr(crow, "yesno_mass_t2"))),
               "net_change_native_pi": net_change(e, c, float((~c1).mean()))}
        for p in PIS:
            row[f"net_change_pi_{p}"] = net_change(e, c, p)
        for k, f in (("eir", lambda a1c, fl, i: eir_ecr(a1c[i], fl[i], ones[i])[0]),
                     ("ecr", lambda a1c, fl, i: eir_ecr(a1c[i], fl[i], ones[i])[1])):
            diff = lambda i, f=f: f(c1, cf, i) - f(b1, bf, i)  # noqa: E731
            lo, hi = cluster_bootstrap_ci(diff, img, n_boot=n_boot, seed=seed)
            row[f"{k}_minus_base"], row[f"{k}_minus_base_ci_low"], row[f"{k}_minus_base_ci_high"] = diff(np.arange(len(keys))), lo, hi
        diff = lambda i: float(c2[i].mean() - b2[i].mean())  # noqa: E731
        lo, hi = cluster_bootstrap_ci(diff, img, n_boot=n_boot, seed=seed)
        row["turn2_acc_minus_base"], row["turn2_acc_minus_base_ci_low"], row["turn2_acc_minus_base_ci_high"] = diff(np.arange(len(keys))), lo, hi
        out["mit"].append(row)

    out["random_band"] = random_gate_band(d["a1c"], d["flip"], np.linspace(0, 1, 21), n_seeds=100, seed=seed)
    out["eir_max"], out["ecr_max"], out["pi_native"] = eir_max, ecr_max, pi_native
    return out


def write_csv(path: Path, rows: list[dict], overwrite_ok: bool) -> None:
    if path.exists() and not overwrite_ok:
        raise FileExistsError(f"{path} exists; the final analysis never overwrites test outputs.")
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def make_figures(results: dict, fig_dir: Path, title_prefix: str, overwrite_ok: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
                         "axes.edgecolor": "#898781", "xtick.color": "#898781", "ytick.color": "#898781",
                         "axes.labelcolor": "#52514e", "legend.frameon": False})
    style = {"conf_gate": ("Confidence gate", "#2a78d6", "-", "o"),
             "learned_A_conf_shift": ("Learned A: conf1 + shift + pushback type", "#eb6834", (0, (6, 2)), "s"),
             "learned_B_conf_shift_reliance": ("Learned B: A + blur/blank reliance", "#1baf7a", (0, (2, 1.5)), "^")}
    titles = {"qwen2vl_2b": "Qwen2-VL-2B", "llava15_7b_fp16": "LLaVA-1.5-7B (fp16)"}
    models = list(results)
    for path in (fig_dir / "eir_ecr.png", fig_dir / "sensitivity_pi.png"):
        if path.exists() and not overwrite_ok:
            raise FileExistsError(f"{path} exists; never overwritten for test.")

    # EIR-ECR with all references.
    fig, axes = plt.subplots(1, len(models), figsize=(3.5 * len(models), 3.5))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        r = results[m]
        band = r["random_band"]
        ax.fill_between([b["eir_mean"] for b in band], [b["ecr_lo"] for b in band], [b["ecr_hi"] for b in band],
                        color="#d9d8d3", lw=0, label="Random gate (95% band)")
        ax.axhline(r["ecr_max"], color="#0b0b0b", lw=0.9, ls=(0, (4, 3)), label="Oracle ceiling (flip rate when a1 wrong)")
        ax.plot([0], [r["ecr_max"]], marker="*", ms=10, color="#0b0b0b", ls="none", label="Oracle")
        ax.plot([r["eir_max"]], [r["ecr_max"]], marker="o", ms=6, color="#52514e", ls="none", label="Always accept")
        ax.plot([0], [0], marker="X", ms=7, color="#52514e", ls="none", label="Never accept")
        for g, (label, col, dash, mk) in style.items():
            pts = [(c["eir"], c["ecr"]) for c in r["curves"] if c["gate"] == g]
            ax.plot(*zip(*pts), color=col, ls=dash, lw=1.6, label=label)
            p = next(p for p in r["points"] if p["gate"] == g)
            ax.plot([p["eir"]], [p["ecr"]], marker=mk, ms=7, color=col, mec="white", ls="none")
            ax.plot([p["random_eir"]], [p["random_ecr"]], marker=mk, ms=7, mfc="white", mec=col, ls="none")
        for mrow in r["mit"]:
            if "eir" in mrow:
                ax.plot([mrow["eir"]], [mrow["ecr"]], marker="D" if mrow["condition"] == "relook" else "P", ms=7,
                        mfc="white", mec="#0b0b0b", ls="none", label=f"{mrow['condition']} prompt (accept all)")
        ax.set_title(titles.get(m, m), fontsize=10)
        ax.set_xlabel("EIR: errors introduced")
        ax.set_ylabel("ECR: errors corrected")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7)
    fig.suptitle(f"{title_prefix}: frozen gates\nFilled marker = operating point at native π; "
                 "hollow marker = matched random gate (same acceptance rate)", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.2, 1, 0.95))
    fig.savefig(fig_dir / "eir_ecr.png", dpi=200)
    plt.close(fig)

    # Net change vs pi (native, 0.1, 0.3, 0.5).
    fig, axes = plt.subplots(1, len(models), figsize=(3.5 * len(models), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        r = results[m]
        order = ["native_pi"] + [f"pi_{p}" for p in PIS]
        x0 = np.array([r["pi_native"]] + PIS)
        # Small sideways offsets so overlapping series stay visible (x is still pi).
        offsets = dict(zip(["always_accept", "never_accept", "oracle", *style], np.linspace(-0.018, 0.018, 6)))
        for ref, col, mk in (("always_accept", "#52514e", "o"), ("never_accept", "#898781", "X"), ("oracle", "#0b0b0b", "*")):
            ys = [next(x["net_change"] for x in r["refs"] if x["reference"] == ref and x["operating_point"] == o) for o in order]
            ax.plot(x0 + offsets[ref], ys, color=col, marker=mk, ls="none", ms=6, label=ref.replace("_", " "))
        for g, (label, col, dash, mk) in style.items():
            xs = x0 + offsets[g]
            rows = [next(x for x in r["gates"] if x["gate"] == g and x["operating_point"] == o) for o in order]
            ax.errorbar(xs, [x["net_change"] for x in rows],
                        yerr=[[x["net_change"] - (x["net_ci_low"] if x["net_ci_low"] is not None else x["net_change"]) for x in rows],
                              [(x["net_ci_high"] if x["net_ci_high"] is not None else x["net_change"]) - x["net_change"] for x in rows]],
                        color=col, marker=mk, ls="none", ms=5, capsize=2, label=label)
            ax.plot(xs, [x["net_change"] - x["net_minus_matched_random"] for x in rows], color=col, marker=mk,
                    mfc="white", ls="none", ms=5)
        ax.axhline(0, color="#898781", lw=0.8)
        ax.axvline(r["pi_native"], color="#d9d8d3", lw=6, zorder=0)
        ax.set_title(titles.get(m, m), fontsize=10)
        ax.set_xlabel("π = P(user is right)  (shaded = native π)")
        ax.set_ylabel("Net accuracy change vs turn 1")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7)
    fig.suptitle(f"{title_prefix}: net change at native π and π = 0.1/0.3/0.5 (hollow = matched random)", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    fig.savefig(fig_dir / "sensitivity_pi.png", dpi=200)
    plt.close(fig)
    print(f"wrote figures to {fig_dir}")


def run(partition: str, final: bool, in_dir: Path, out_dir: Path, fig_dir: Path, models: list[str],
        frozen_path: Path, n_boot: int | None = None) -> dict:
    exp = load_yaml("configs/experiment.yaml")
    n_boot = n_boot or exp["signals"]["n_boot"]
    seed = exp["seed"]
    if partition == "test":
        if not final:
            sys.exit("The test analysis runs only with --final (normally via scripts/run_final.py --final).")
        check_release()
        status = "FINAL test result (frozen gates, no refitting)"
        overwrite_ok = False
    else:
        status = "DRY RUN on dev: frozen gates were fit on dev, so this is in-sample and NOT a result"
        overwrite_ok = True
    frozen = json.loads(frozen_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for m in models:
        d = load_signals(in_dir / f"signals_{partition}_{m}.jsonl")
        if d["partition"] != {partition}:
            sys.exit(f"{m}: signals file has partitions {d['partition']}, expected {partition}.")
        results[m] = analyse_model(m, d, frozen, in_dir, partition, n_boot, seed, status)
        print(f"analysed {m}: {d['n']} rows")
    meta = {"status": status, "partition": partition, "frozen_gates_sha256": sha256_file(frozen_path),
            "frozen_gates_commit": frozen.get("git_commit"), "n_boot": n_boot, "seed": seed}
    for key, name in [("gates", "gates.csv"), ("refs", "references.csv"), ("auroc", "auroc.csv"), ("h2", "h2.csv"),
                      ("h3", "h3.csv"), ("mit", "mitigations.csv"), ("by_a1", "by_a1.csv")]:
        write_csv(out_dir / name, [r for m in models for r in results[m][key]], overwrite_ok)
    # Overall labels across the two models (analysis_plan.md, Amendment B).
    primary = [r for m in models for r in results[m]["h2"] if r["primary"]]
    meta["h2_per_model"] = {r["model"]: {"difference_B_minus_A": r["difference_B_minus_A"], "ci": [r["ci_low"], r["ci_high"]],
                                         "decision": r["decision"]} for r in primary}
    meta["h2_overall"] = overall_h2([ci_side(r["ci_low"], r["ci_high"]) for r in primary])
    h3_per_model = {m: next(r["gate"] for r in results[m]["h3"] if r["row"] == "verdict") == "supported" for m in models}
    meta["h3_per_model"] = {m: ("supported" if ok else "NOT supported") for m, ok in h3_per_model.items()}
    meta["h3_overall"] = overall_h3(list(h3_per_model.values()))
    meta_path = out_dir / "run_info.json"
    if meta_path.exists() and not overwrite_ok:
        raise FileExistsError(f"{meta_path} exists")
    meta_path.write_text(json.dumps(meta, indent=2))
    title = "FINAL (test)" if partition == "test" else "DRY RUN on dev (in-sample, not a result)"
    make_figures(results, fig_dir, title, overwrite_ok)
    print("H2 per model (B - A, frozen gates):")
    for m, v in meta["h2_per_model"].items():
        lo, hi = v["ci"]
        ci = f"[{lo:+.3f}, {hi:+.3f}]" if lo is not None and hi is not None else "[undefined]"
        diff = f"{v['difference_B_minus_A']:+.3f}" if v["difference_B_minus_A"] is not None else "undefined"
        print(f"  {m}: {diff} {ci} -> {v['decision']}")
    print(f"H2 overall (Amendment B2): {meta['h2_overall']}")
    print(f"H3 per model: {meta['h3_per_model']}")
    print(f"H3 overall (Amendment B1): {meta['h3_overall']}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--partition", required=True, choices=["dev", "test"])
    parser.add_argument("--final", action="store_true", help="required for the test partition")
    parser.add_argument("--in-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--fig-dir", type=Path)
    parser.add_argument("--models", nargs="+", default=["qwen2vl_2b", "llava15_7b_fp16"])
    parser.add_argument("--frozen", type=Path, default=REPO_ROOT / "results" / "frozen_gates.json")
    parser.add_argument("--n-boot", type=int)
    args = parser.parse_args()
    tag = "final_test" if args.partition == "test" else "final_dryrun_dev"
    run(args.partition, args.final, args.in_dir or results_dir(), args.out_dir or results_dir() / tag,
        args.fig_dir or REPO_ROOT / "figures" / tag, args.models, args.frozen, args.n_boot)


if __name__ == "__main__":
    main()
