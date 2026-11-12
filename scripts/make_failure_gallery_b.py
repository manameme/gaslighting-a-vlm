"""Failure gallery, view B (post-hoc, presentation only; added 2026-11-12 after the test run).

    python scripts/make_failure_gallery_b.py --partition test

This is a SECOND, clearly labelled view next to the pre-registered gallery
(make_failure_gallery.py), which is unchanged. It was specified by the team after the test
results had been seen, so it is descriptive and disclosed as post-hoc in AI_USE_LOG.md and
the report. It changes no threshold, gate, rule or model output.

SELECTION RULE (written before this script was first run; do not change):
  For each model, using the base-condition rows of baseline_<p>_<model>.jsonl joined with
  signals_<p>_<model>.jsonl:
  1. Keep rows with pushback type "assertive".
  2. Gate = learned gate B from results/frozen_gates.json at its frozen native-pi threshold
     ("empirical_pi"), scored with the same code as analyse_final.py (gate_score). The gate
     BLOCKS a flip when its score is not above the threshold.
  3. Group (a) "missed corrections": a1 wrong, a2 correct (the model flipped to the right
     answer), and gate B blocked the flip.
     Group (b) "harmful flips blocked": a1 right, a2 wrong (the model flipped to the wrong
     answer), and gate B blocked the flip.
  4. In each group, sort rows by question_id, then POPE split (random, popular, adversarial),
     then paraphrase_id, and take the first row of each question (uid) until three distinct
     questions are chosen: the three lowest question ids.
Outputs (never overwritten):
  results/failure_view_b_<p>.csv                 the 2 x 3 chosen rows per model, with gate score and threshold
  figures/failure_view_b_<p>_<model>.png         16 cm wide, 2 columns (a | b) x 3 rows, text >= 6 pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_final import gate_score, load_signals  # noqa: E402  (same scoring code as the final analysis)
from vlmsyc.paths import REPO_ROOT, data_dir, load_yaml, results_dir  # noqa: E402
from vlmsyc.pipeline import read_rows  # noqa: E402
from vlmsyc.prompts import article  # noqa: E402

MODELS = ["qwen2vl_2b", "llava15_7b_fp16"]
GATE = "learned_B_conf_shift_reliance"
PUSHBACK = "assertive"
N_PER_GROUP = 3
SPLIT_ORDER = {"random": 0, "popular": 1, "adversarial": 2}
TITLES = {"qwen2vl_2b": "Qwen2-VL-2B", "llava15_7b_fp16": "LLaVA-1.5-7B (fp16)"}
GROUPS = {"a": "missed correction (a1 wrong, a2 right, gate B blocked)",
          "b": "harmful flip blocked (a1 right, a2 wrong, gate B blocked)"}


def choose(model: str, partition: str, frozen: dict) -> list[dict]:
    """Steps 1-4 of the rule."""
    R = results_dir()
    d = load_signals(R / f"signals_{partition}_{model}.jsonl")
    sig = read_rows(R / f"signals_{partition}_{model}.jsonl")
    entry = frozen["models"][model]["gates"][GATE]
    score = gate_score(d, GATE, entry)
    tau = entry["thresholds"]["empirical_pi"]
    base = {(r["uid"], r["pushback_type"], r["paraphrase_id"]): r for r in read_rows(R / f"baseline_{partition}_{model}.jsonl")}
    cand = {"a": [], "b": []}
    for i, s in enumerate(sig):
        if s["pushback_type"] != PUSHBACK or not s["flip"] or score[i] > tau:
            continue                                     # step 1; only blocked flips (step 2)
        b = base[(s["uid"], s["pushback_type"], s["paraphrase_id"])]
        group = "a" if not s["a1_correct"] and s["a2_correct"] else ("b" if s["a1_correct"] and not s["a2_correct"] else None)
        if group:
            cand[group].append({**b, "gate": GATE, "gate_score": float(score[i]), "threshold": tau, "group": group,
                                "group_label": GROUPS[group]})
    out = []
    for g in ("a", "b"):
        rows_g = sorted(cand[g], key=lambda r: (r["question_id"], SPLIT_ORDER[r["pope_split"]], r["paraphrase_id"]))
        seen = []
        for r in rows_g:
            if r["uid"] not in seen:
                seen.append(r["uid"])
                out.append({k: r[k] for k in ("model", "partition", "group", "group_label", "uid", "image_id", "pope_split",
                                              "question_id", "object", "label", "pushback_type", "paraphrase_id",
                                              "pushback_text", "p1_yes", "a1", "a1_correct", "p2_yes", "a2", "a2_correct",
                                              "flip", "gate", "gate_score", "threshold")}
                           | {"candidates_in_group": len(cand[g])})
            if len(seen) == N_PER_GROUP:
                break
    return out


def figure(model: str, chosen: list[dict], path: Path, status: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    width_in, body, title = 16 / 2.54, 6.5, 7.5
    image_dir = data_dir() / load_yaml("configs/data.yaml")["image_dir"]
    fig, axes = plt.subplots(N_PER_GROUP, 2, figsize=(width_in, N_PER_GROUP * 3.15))
    for ax in axes.ravel():
        ax.axis("off")
    for col, g in enumerate(("a", "b")):
        rows_g = [r for r in chosen if r["group"] == g]
        for row_i, r in enumerate(rows_g):
            ax = axes[row_i, col]
            img = Image.open(image_dir / f"COCO_val2014_{int(r['image_id']):012d}.jpg").convert("RGB")
            img.thumbnail((900, 900))
            ax.imshow(img)
            q = f"Is there {article(r['object'])} {r['object']} in the image?"
            text = "\n".join([
                textwrap.fill(f"Q ({r['pope_split']}): {q} Truth: {r['label']}", 58),
                textwrap.fill(f"User: \"{r['pushback_text']}\"", 58),
                f"Turn 1: {r['a1']} (p_yes {r['p1_yes']:.2f});  Turn 2: {r['a2']} (p_yes {r['p2_yes']:.2f})",
                f"Gate B score {r['gate_score']:.3f} <= threshold {r['threshold']:.3f}: blocked",
            ])
            ax.set_title(f"({g}) COCO {r['image_id']}", fontsize=title)
            ax.text(0, -0.03, text, transform=ax.transAxes, va="top", ha="left", fontsize=body, family="Arial")
        axes[0, col].annotate(textwrap.fill(GROUPS[g], 34), xy=(0.5, 1.16), xycoords="axes fraction", ha="center",
                              va="bottom", fontsize=title, fontweight="bold")
    fig.suptitle(f"{TITLES.get(model, model)}, view B (post-hoc): assertive pushback,\nlearned gate B at its frozen "
                 f"native-pi threshold ({status})", fontsize=title)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.07, wspace=0.06, hspace=0.62)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path} ({width_in * 2.54:.1f} cm wide; text {body}-{title} pt)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--partition", required=True, choices=["dev", "test"])
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--frozen", type=Path, default=REPO_ROOT / "results" / "frozen_gates.json")
    args = parser.parse_args()
    p = args.partition
    csv_path = results_dir() / f"failure_view_b_{p}.csv"
    figs = {m: REPO_ROOT / "figures" / f"failure_view_b_{p}_{m}.png" for m in args.models}
    for path in [csv_path, *figs.values()]:
        if path.exists():
            sys.exit(f"{path} exists; never overwritten.")
    frozen = json.loads(args.frozen.read_text())
    status = "dev split" if p == "dev" else "test split"
    allrows = []
    for m in args.models:
        chosen = choose(m, p, frozen)
        for g in ("a", "b"):
            n = sum(r["group"] == g for r in chosen)
            print(f"{m}: group {g}: {n} chosen of {next((r['candidates_in_group'] for r in chosen if r['group'] == g), 0)} candidate rows")
        allrows += chosen
        figure(m, chosen, figs[m], status)
    fields = list(allrows[0])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(allrows)
    print(f"wrote {csv_path} ({len(allrows)} rows)")


if __name__ == "__main__":
    main()
