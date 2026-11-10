"""Signals that a gate can use to decide whether to accept a revision (SPEC section 6).

Every signal is computable at inference time without the label:
  conf1             max(p1, 1 - p1): turn-1 confidence.
  p1_a1             p(a1) at turn 1 (equal to conf1, kept for the reliance formula).
  reliance_blur     p(a1 | image) - p(a1 | blurred image), both at turn 1.
  reliance_blank    p(a1 | image) - p(a1 | blank image), both at turn 1.
                    Large = the answer depends on what the image shows.
  shift             logit(p2_yes) - logit(p1_yes): how far the pushback moved
                    the model, signed (positive = towards yes).
  shift_to_claim    the same, signed so that positive = towards the user's claim
                    (the opposite of a1). Easier to read than shift.

Crop consistency is cut (scope cut, CLAUDE.md).
Degraded images use the same turn-1 prompt; only the image changes.
"""

from __future__ import annotations

import math


def logit(p: float, eps: float = 1e-6) -> float:
    """log(p / (1 - p)), clipped so p = 0 or 1 does not give infinity."""
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def p_of_answer(p_yes: float, answer: str) -> float:
    """Probability the model gives `answer` ('yes' or 'no'), from p(yes)."""
    if answer == "yes":
        return p_yes
    if answer == "no":
        return 1 - p_yes
    raise ValueError(f"answer must be 'yes' or 'no', got {answer!r}")


def conf1(p1_yes: float) -> float:
    return max(p1_yes, 1 - p1_yes)


def reliance(p1_yes_image: float, p1_yes_degraded: float, a1: str) -> float:
    """Drop in p(a1) when the image is degraded. Uses a1 from the real image."""
    return p_of_answer(p1_yes_image, a1) - p_of_answer(p1_yes_degraded, a1)


def shift(p1_yes: float, p2_yes: float) -> float:
    return logit(p2_yes) - logit(p1_yes)


def shift_to_claim(p1_yes: float, p2_yes: float, a1: str) -> float:
    """Shift in log-odds towards the user's claim. The claim is always the opposite of a1."""
    s = shift(p1_yes, p2_yes)
    return -s if a1 == "yes" else s


# ---------- image degradations ----------


def blur_image(image, sigma_frac: float):
    """Gaussian blur with standard deviation sigma_frac x the longer image side.

    Relative to image size, so the strength does not depend on how large the
    stored image is. The value is fixed in configs/experiment.yaml before any
    signal is looked at; it is not tuned.
    """
    from PIL import ImageFilter

    radius = sigma_frac * max(image.size)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def blank_image(image):
    """An image of the same size filled with the image's mean colour (no content)."""
    from PIL import Image, ImageStat

    mean = tuple(int(round(c)) for c in ImageStat.Stat(image).mean[:3])
    return Image.new("RGB", image.size, mean)


DEGRADATIONS = ("blur", "blank")


def degrade(image, kind: str, cfg: dict):
    if kind == "blur":
        return blur_image(image, cfg["blur_sigma_frac"])
    if kind == "blank":
        return blank_image(image)
    raise ValueError(f"Unknown degradation {kind!r}")


# ---------- joining ----------


def signal_rows(baseline_rows: list[dict], degraded: dict[tuple[str, str], float]) -> list[dict]:
    """One signals row per baseline row (question x pushback).

    degraded maps (uid, degradation) -> turn-1 p(yes) on the degraded image.
    Labels and outcomes (a1_correct, flip, a2_correct) are copied along for
    evaluation only; no signal uses them.
    """
    out = []
    for r in baseline_rows:
        a1 = r["a1"]
        row = {
            "uid": r["uid"],
            "question_id": r["question_id"],
            "image_id": r["image_id"],
            "pope_split": r["pope_split"],
            "partition": r["partition"],
            "model": r["model"],
            "pushback_type": r["pushback_type"],
            "paraphrase_id": r["paraphrase_id"],
            "condition": r["condition"],
            # signals
            "conf1": conf1(r["p1_yes"]),
            "shift": shift(r["p1_yes"], r["p2_yes"]),
            "shift_to_claim": shift_to_claim(r["p1_yes"], r["p2_yes"], a1),
        }
        for kind in DEGRADATIONS:
            p_deg = degraded.get((r["uid"], kind))
            row[f"p1_yes_{kind}"] = p_deg
            row[f"reliance_{kind}"] = None if p_deg is None else reliance(r["p1_yes"], p_deg, a1)
        # outcomes, for evaluation only
        row.update(
            {
                "label": r["label"],
                "a1": a1,
                "a2": r["a2"],
                "a1_correct": r["a1_correct"],
                "a2_correct": r["a2_correct"],
                "flip": r["flip"],
                "good_revision": bool(r["flip"] and r["a2_correct"]),
                "seed": r["seed"],
                "git_commit": r["git_commit"],
            }
        )
        out.append(row)
    return out
