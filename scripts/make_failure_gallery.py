"""Failure-case gallery (Phase 8): examples where EVERY gate and EVERY prompt mitigation fails.

    python scripts/make_failure_gallery.py --partition dev
    python scripts/make_failure_gallery.py --partition test     # later, on the saved test outputs

SELECTION RULE (fixed on 2026-11-11, before any test output was read; do not change):
  Unit = one row (question x pushback type x paraphrase) present in all of
  signals_<p>_<model>.jsonl, baseline_<p>_<model>.jsonl, grounding_<p>_<model>.jsonl
  and relook_<p>_<model>.jsonl.
  1. A GATE fails on a row if its final answer is wrong. Every gate in
     results/frozen_gates.json is used (conf, blur, blank, shift, learned A, learned B),
     at its frozen native-pi threshold ("empirical_pi"); final answer = a2 if the model
     flipped and the gate accepts (score > threshold), else a1. Same code as
     analyse_final.py (gate_score), no refitting.
  2. A PROMPT MITIGATION fails on a row if its turn-2 answer a2 is wrong under that
     condition (grounding, re-look; every revision accepted).
  3. A row is a UNIVERSAL FAILURE if all 6 gates and both mitigations fail.
  4. Images are ranked by their number of universal-failure rows (descending), ties by
     image_id (ascending). The top K images per model are shown (K = 6).
  5. For each shown image, the example row is its first universal-failure row in the order:
     POPE split (random, popular, adversarial), question_id, pushback type (neutral_doubt,
     assertive, authority, polite), paraphrase_id.
Outputs (never overwritten for test):
  results/failure_gallery_<p>.csv        the shown examples with every decision
  results/failure_counts_<p>.csv         universal-failure counts per model, split by a1 correct/wrong
  figures/failure_gallery_<p>_<model>.png
Descriptive only: no thresholds, rules or models are changed here.
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

K_IMAGES = 6
MODELS = ["qwen2vl_2b", "llava15_7b_fp16"]
MITIGATIONS = ["grounding", "relook"]
SPLIT_ORDER = {"random": 0, "popular": 1, "adversarial": 2}
PUSHBACK_ORDER = {"neutral_doubt": 0, "assertive": 1, "authority": 2, "polite": 3}
TITLES = {"qwen2vl_2b": "Qwen2-VL-2B", "llava15_7b_fp16": "LLaVA-1.5-7B (fp16)"}


def key(r: dict) -> tuple:
    return (r["uid"], r["pushback_type"], r["paraphrase_id"])


def universal_failures(model: str, partition: str, in_dir: Path, frozen: dict) -> list[dict]:
    """Every row with the decision of each gate and mitigation (steps 1-3 of the rule)."""
    d = load_signals(in_dir / f"signals_{partition}_{model}.jsonl")
    gates = frozen["models"][model]["gates"]
    base = {key(r): r for r in read_rows(in_dir / f"baseline_{partition}_{model}.jsonl")}
    mit = {c: {key(r): r for r in read_rows(in_dir / f"{c}_{partition}_{model}.jsonl")} for c in MITIGATIONS}
    sig_rows = read_rows(in_dir / f"signals_{partition}_{model}.jsonl")
    final_correct = {}
    for g, entry in gates.items():
        accept = gate_score(d, g, entry) > entry["thresholds"]["empirical_pi"]
        swap = d["flip"] & accept
        final_correct[g] = np.where(swap, ~d["a1c"], d["a1c"])     # binary task: a2 = not a1 on a flip
    out = []
    for i, s in enumerate(sig_rows):
        k = key(s)
        if k not in base or any(k not in mit[c] for c in MITIGATIONS):
            continue
        b = base[k]
        row = {
            "model": model, "partition": partition, "uid": s["uid"], "image_id": s["image_id"],
            "pope_split": s["pope_split"], "question_id": s["question_id"], "object": b["object"],
            "label": s["label"], "pushback_type": s["pushback_type"], "paraphrase_id": s["paraphrase_id"],
            "pushback_text": b["pushback_text"], "p1_yes": b["p1_yes"], "a1": b["a1"],
            "a1_correct": b["a1_correct"], "p2_yes": b["p2_yes"], "a2": b["a2"], "flip": b["flip"],
        }
        for g in gates:
            row[f"{g}_final_correct"] = bool(final_correct[g][i])
        for c in MITIGATIONS:
            m = mit[c][k]
            row[f"{c}_a1"], row[f"{c}_a2"], row[f"{c}_a2_correct"] = m["a1"], m["a2"], bool(m["a2_correct"])
        row["universal_failure"] = (not any(row[f"{g}_final_correct"] for g in gates)
                                    and not any(row[f"{c}_a2_correct"] for c in MITIGATIONS))
        out.append(row)
    return out


def select(rows: list[dict]) -> list[dict]:
    """Steps 4-5 of the rule: top K images by count, then each image's first failing row."""
    fails = [r for r in rows if r["universal_failure"]]
    counts: dict = {}
    for r in fails:
        counts[r["image_id"]] = counts.get(r["image_id"], 0) + 1
    top = sorted(counts, key=lambda img: (-counts[img], img))[:K_IMAGES]
    chosen = []
    for img in top:
        cand = sorted((r for r in fails if r["image_id"] == img),
                      key=lambda r: (SPLIT_ORDER[r["pope_split"]], r["question_id"],
                                     PUSHBACK_ORDER[r["pushback_type"]], r["paraphrase_id"]))
        chosen.append({**cand[0], "universal_failures_on_image": counts[img]})
    return chosen


def figure(model: str, chosen: list[dict], path: Path, status: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    image_dir = data_dir() / load_yaml("configs/data.yaml")["image_dir"]
    cols = 3
    rows_n = max(1, int(np.ceil(len(chosen) / cols)))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.2, rows_n * 6.0))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, r in zip(axes, chosen):
        img = Image.open(image_dir / f"COCO_val2014_{int(r['image_id']):012d}.jpg").convert("RGB")
        img.thumbnail((640, 640))
        ax.imshow(img)
        q = f"Is there {article(r['object'])} {r['object']} in the image?"
        text = "\n".join([
            textwrap.fill(f"Q ({r['pope_split']}): {q}  Truth: {r['label']}", 52),
            textwrap.fill(f"User: \"{r['pushback_text']}\"", 52),
            f"Turn 1: {r['a1']} (p_yes {r['p1_yes']:.2f}, {'right' if r['a1_correct'] else 'wrong'})",
            f"Turn 2: {r['a2']} (p_yes {r['p2_yes']:.2f}, {'flipped' if r['flip'] else 'kept'})",
            f"Grounding a2: {r['grounding_a2']}   Re-look a2: {r['relook_a2']}",
            f"All 6 gates and both prompts wrong; image has {r['universal_failures_on_image']} such rows",
        ])
        ax.set_title(f"COCO {r['image_id']}", fontsize=8.5)
        ax.text(0, -0.04, text, transform=ax.transAxes, va="top", ha="left", fontsize=7.2, family="Arial")
    fig.suptitle(f"{TITLES.get(model, model)}: failure cases ({status})", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), h_pad=9)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"wrote {path}")


def figure_a4(model: str, chosen: list[dict], path: Path, status: str) -> None:
    """Same examples, laid out for one A4 column: 2 columns x 3 rows, 16 cm wide, all text >= 6 pt.

    Added 2026-11-12 after the test run, presentation only (the selection rule and the
    CSV are unchanged; see AI_USE_LOG.md). Written as a new file; nothing is overwritten.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    width_in = 16 / 2.54
    body, title = 6.5, 7.5                                 # points at the printed size
    image_dir = data_dir() / load_yaml("configs/data.yaml")["image_dir"]
    cols, rows_n = 2, int(np.ceil(len(chosen) / 2))
    fig, axes = plt.subplots(rows_n, cols, figsize=(width_in, rows_n * 3.15))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, r in zip(axes, chosen):
        img = Image.open(image_dir / f"COCO_val2014_{int(r['image_id']):012d}.jpg").convert("RGB")
        img.thumbnail((900, 900))
        ax.imshow(img)
        q = f"Is there {article(r['object'])} {r['object']} in the image?"
        text = "\n".join([
            textwrap.fill(f"Q ({r['pope_split']}): {q} Truth: {r['label']}", 58),
            textwrap.fill(f"User: \"{r['pushback_text']}\"", 58),
            f"Turn 1: {r['a1']} (p_yes {r['p1_yes']:.2f}, {'right' if r['a1_correct'] else 'wrong'});"
            f"  Turn 2: {r['a2']} (p_yes {r['p2_yes']:.2f}, {'flipped' if r['flip'] else 'kept'})",
            f"Grounding a2: {r['grounding_a2']};  Re-look a2: {r['relook_a2']}",
            f"Universal-failure rows on this image: {r['universal_failures_on_image']}",
        ])
        ax.set_title(f"COCO {r['image_id']}", fontsize=title)
        ax.text(0, -0.03, text, transform=ax.transAxes, va="top", ha="left", fontsize=body, family="Arial")
    fig.suptitle(f"{TITLES.get(model, model)}: failure cases ({status}); all 6 gates and both prompts wrong",
                 fontsize=title)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.07, wspace=0.06, hspace=0.62)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path} ({width_in * 2.54:.1f} cm wide; text {body}-{title} pt)")


def write_csv(path: Path, rows: list[dict], overwrite_ok: bool) -> None:
    if path.exists() and not overwrite_ok:
        raise FileExistsError(f"{path} exists; test outputs are never overwritten.")
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
    parser.add_argument("--partition", required=True, choices=["dev", "test"])
    parser.add_argument("--in-dir", type=Path, help="default RESULTS_DIR")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--frozen", type=Path, default=REPO_ROOT / "results" / "frozen_gates.json")
    parser.add_argument("--a4-figures-only", action="store_true",
                        help="only draw the 16 cm (A4 column) figures as new files *_a4.png; write no CSV. "
                             "The selection must equal the committed failure_gallery_<p>.csv")
    args = parser.parse_args()
    in_dir = args.in_dir or results_dir()
    overwrite_ok = args.partition == "dev"
    status = "dev split" if args.partition == "dev" else "test split"
    frozen = json.loads(args.frozen.read_text())
    gallery, counts = [], []
    for m in args.models:
        rows = universal_failures(m, args.partition, in_dir, frozen)
        for sub, sel in (("all rows", rows), ("a1 correct", [r for r in rows if r["a1_correct"]]),
                         ("a1 wrong", [r for r in rows if not r["a1_correct"]])):
            n_fail = sum(r["universal_failure"] for r in sel)
            counts.append({"model": m, "partition": args.partition, "subset": sub, "n_rows": len(sel),
                           "universal_failures": n_fail,
                           "share": n_fail / len(sel) if sel else None,
                           "images_with_failure": len({r["image_id"] for r in sel if r["universal_failure"]})})
        chosen = select(rows)
        gallery += chosen
        if args.a4_figures_only:
            committed = [r for r in csv.DictReader(open(results_dir() / f"failure_gallery_{args.partition}.csv"))
                         if r["model"] == m]
            same = [(r["uid"], r["pushback_type"], str(r["paraphrase_id"])) for r in chosen] == \
                   [(r["uid"], r["pushback_type"], r["paraphrase_id"]) for r in committed]
            if not same:
                sys.exit(f"{m}: selection differs from the committed CSV; refusing to draw.")
            a4_path = REPO_ROOT / "figures" / f"failure_gallery_{args.partition}_{m}_a4.png"
            if a4_path.exists():
                sys.exit(f"{a4_path} exists; never overwritten.")
            figure_a4(m, chosen, a4_path, status)
            continue
        fig_path = REPO_ROOT / "figures" / f"failure_gallery_{args.partition}_{m}.png"
        if fig_path.exists() and not overwrite_ok:
            raise FileExistsError(f"{fig_path} exists")
        figure(m, chosen, fig_path, status)
    if args.a4_figures_only:
        return
    write_csv(results_dir() / f"failure_gallery_{args.partition}.csv", gallery, overwrite_ok)
    write_csv(results_dir() / f"failure_counts_{args.partition}.csv", counts, overwrite_ok)


if __name__ == "__main__":
    main()
