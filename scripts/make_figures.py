"""Figures for the report, drawn only from CSV files in RESULTS_DIR (no model runs).

    python scripts/make_figures.py          # dev figures (Phase 5)

Writes to figures/:
  eir_ecr_dev.png            out-of-fold EIR-ECR curves per gate, with the random-gate band and
                             the oracle ceiling (ECR capped by the flip rate when a1 is wrong)
  reliability_by_split_dev.png   turn-1 reliability diagrams per POPE split (H3 calibration)
  sensitivity_pi_dev.png     net accuracy change vs pi = P(user right), per gate
Colours: validated categorical palette (dataviz reference palette, slots 1-5),
plus a distinct dash pattern per series so identity never depends on colour alone.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlmsyc.paths import REPO_ROOT, results_dir  # noqa: E402

FIG = REPO_ROOT / "figures"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
SERIES = {  # gate -> (label, colour, dash)
    "conf_gate": ("Confidence (conf1)", "#2a78d6", "-"),
    "reliance_blur_gate": ("Blur reliance", "#eb6834", (0, (6, 2))),
    "reliance_blank_gate": ("Blank reliance", "#1baf7a", (0, (2, 1.5))),
    "learned_A_conf_shift": ("Learned A: conf1 + shift", "#eda100", (0, (6, 2, 1, 2))),
    "learned_B_conf_shift_reliance": ("Learned B: A + reliance", "#e87ba4", (0, (1, 1))),
}
SPLIT_STYLE = {"random": ("#2a78d6", "-"), "popular": ("#eb6834", (0, (6, 2))), "adversarial": ("#1baf7a", (0, (2, 1.5)))}
MODEL_TITLE = {"qwen2vl_2b": "Qwen2-VL-2B", "llava15_7b_fp16": "LLaVA-1.5-7B (fp16)"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 1.6,
})


def read(name: str) -> list[dict]:
    with open(results_dir() / name) as f:
        return list(csv.DictReader(f))


def models_in(rows):
    seen = []
    for r in rows:
        if r["model"] not in seen:
            seen.append(r["model"])
    return seen


def eir_ecr_figure():
    curves, rand, gates = read("gate_curves_dev.csv"), read("random_gate_dev.csv"), read("gates_dev.csv")
    models = models_in(curves)
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.3), sharey=True)
    axes = [axes] if len(models) == 1 else axes
    for ax, m in zip(axes, models):
        ref = {r["gate"]: r for r in gates if r["model"] == m and r["evaluation"] == "reference"}
        eir_max, ecr_max = float(ref["always_accept"]["eir"]), float(ref["always_accept"]["ecr"])
        # Random gate band (accept each flip with probability r, 100 seeds).
        rr = [r for r in rand if r["model"] == m]
        ax.fill_between([float(r["eir_mean"]) for r in rr], [float(r["ecr_lo"]) for r in rr],
                        [float(r["ecr_hi"]) for r in rr], color="#d9d8d3", lw=0, label="Random gate (95% band)")
        ax.plot([0, eir_max], [0, ecr_max], color=MUTED, lw=0.8)
        # Oracle ceiling: no gate can fix more errors than the model already flipped.
        ax.axhline(ecr_max, color=INK, lw=0.9, ls=(0, (4, 3)))
        ax.annotate("Oracle ceiling: ECR = flip rate when a1 wrong", (eir_max, ecr_max), xytext=(0, 4),
                    textcoords="offset points", ha="right", va="bottom", fontsize=7.5, color=INK2)
        ax.plot([0], [ecr_max], marker="*", ms=9, color=INK, ls="none", label="Oracle gate")
        for g, (label, col, dash) in SERIES.items():
            pts = [(float(r["eir"]), float(r["ecr"])) for r in curves
                   if r["model"] == m and r["gate"] == g and r["evaluation"] == "OOF"]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, ls=dash, label=label)
        ax.plot([eir_max], [ecr_max], marker="o", ms=6, color=INK2, ls="none", label="Always accept")
        ax.set_title(MODEL_TITLE.get(m, m))
        ax.set_xlabel("EIR: errors introduced (lower is better)")
        ax.set_xlim(0, eir_max * 1.03)
    # Shared y-axis: set once, from the largest ceiling, so no panel's ceiling is cut off.
    top = max(float(r["ecr"]) for r in gates if r["gate"] == "always_accept")
    axes[0].set_ylim(0, top * 1.15)
    axes[0].set_ylabel("ECR: errors corrected (higher is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Gates on dev, out-of-fold (5-fold grouped by image)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out = FIG / "eir_ecr_dev.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def reliability_figure():
    rel = read("reliability_dev.csv")
    models = models_in(rel)
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.0), sharey=True)
    axes = [axes] if len(models) == 1 else axes
    for ax, m in zip(axes, models):
        ax.plot([0.5, 1], [0.5, 1], color=MUTED, lw=0.8, label="Perfect calibration")
        for sp, (col, dash) in SPLIT_STYLE.items():
            pts = [r for r in rel if r["model"] == m and r["pope_split"] == sp]
            ax.plot([float(r["mean_conf1"]) for r in pts], [float(r["accuracy"]) for r in pts],
                    color=col, ls=dash, marker="o", ms=4, label=sp)
        ax.set_title(MODEL_TITLE.get(m, m))
        ax.set_xlabel("Turn-1 confidence (conf1), 10 equal-count bins")
        ax.set_xlim(0.5, 1.01)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Turn-1 accuracy")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Turn-1 calibration per POPE split (dev, 200 questions per split)", fontsize=10, color=INK)
    fig.tight_layout()
    out = FIG / "reliability_by_split_dev.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def sensitivity_figure():
    sens = read("sensitivity_pi_dev.csv")
    models = models_in(sens)
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.2), sharey=True)
    axes = [axes] if len(models) == 1 else axes
    for ax, m in zip(axes, models):
        by = defaultdict(list)
        for r in sens:
            if r["model"] == m:
                by[r["gate"]].append((float(r["pi"]), float(r["net_change"])))
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.plot(*zip(*by["oracle"]), color=INK, ls=(0, (4, 3)), marker="*", ms=7, label="Oracle")
        ax.plot(*zip(*by["always_accept"]), color=INK2, ls="-", lw=1.0, marker="o", ms=4, label="Always accept")
        for g, (label, col, dash) in SERIES.items():
            ax.plot(*zip(*by[g]), color=col, ls=dash, marker="o", ms=3.5, label=label)
        ax.set_title(MODEL_TITLE.get(m, m))
        ax.set_xlabel("π = P(user is right)")
        ax.set_xticks([0.1, 0.3, 0.5])
    axes[0].set_ylabel("Net accuracy change vs turn 1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Sensitivity to how often the user is right (dev, OOF, thresholds re-tuned per π)",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out = FIG / "sensitivity_pi_dev.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    eir_ecr_figure()
    reliability_figure()
    sensitivity_figure()
