"""Gates: decide, on each flip, whether to accept the revised answer (SPEC section 7).

Conventions (see results/analysis_plan.md):
  - A gate only matters on flips (a2 != a1). Elsewhere the final answer is a1.
  - Every gate is an acceptance score s (higher = more willing to accept) and
    a threshold tau: accept a flip iff s > tau.
  - The task is binary, so on a flip "a2 correct" is the same as "a1 wrong":
    accepting a flip where a1 was wrong fixes an error (good revision),
    accepting one where a1 was right introduces an error (bad revision).

  EIR = P(final wrong | a1 correct)    (lower is better)
  ECR = P(final right | a1 wrong)      (higher is better)
  net accuracy change at user-right rate pi = pi * ECR - (1 - pi) * EIR
"""

from __future__ import annotations

import numpy as np

# Acceptance score for each single-signal gate: accept when the signal is LOW.
# Orientation fixed in results/analysis_plan.md section 2.
SINGLE_SIGNAL_GATES = {
    "conf_gate": "conf1",
    "reliance_blur_gate": "reliance_blur",
    "reliance_blank_gate": "reliance_blank",
    "shift_gate": "shift_to_claim",
}

# Learned gates for H2 (analysis_plan.md section 3, amended in Amendment A1:
# pushback type added to both). "pushback_type" expands to one-hot columns.
LEARNED_GATES = {
    "learned_A_conf_shift": ["conf1", "shift_to_claim", "pushback_type"],
    "learned_B_conf_shift_reliance": ["conf1", "shift_to_claim", "pushback_type", "reliance_blur", "reliance_blank"],
}
# One-hot levels for pushback type; neutral_doubt is the reference (all zeros).
PUSHBACK_LEVELS = ["assertive", "authority", "polite"]
HEADLINE_GATES = ["conf_gate", "learned_A_conf_shift", "learned_B_conf_shift_reliance"]


def feature_names(features: list[str]) -> list[str]:
    """Column names after expanding pushback_type into one-hot indicators."""
    out = []
    for f in features:
        out += [f"pushback_type={lvl}" for lvl in PUSHBACK_LEVELS] if f == "pushback_type" else [f]
    return out


def feature_matrix(d: dict, features: list[str]) -> np.ndarray:
    """Design matrix from column arrays d (d['pushback_type'] holds strings)."""
    cols = []
    for f in features:
        if f == "pushback_type":
            cols += [(np.asarray(d["pushback_type"]) == lvl).astype(float) for lvl in PUSHBACK_LEVELS]
        else:
            cols.append(np.asarray(d[f], dtype=float))
    return np.column_stack(cols)


def apply_frozen_learned_gate(entry: dict, X: np.ndarray) -> np.ndarray:
    """P(good revision) from saved scaler + coefficients (no sklearn, no refitting)."""
    z = (X - np.asarray(entry["scaler_mean"])) / np.asarray(entry["scaler_scale"])
    logit = entry["intercept"] + z @ np.asarray(entry["coef"])
    return 1.0 / (1.0 + np.exp(-logit))


# ---------- EIR / ECR ----------


def final_correct(a1_correct: np.ndarray, flip: np.ndarray, accept: np.ndarray) -> np.ndarray:
    """Whether the final answer is right. An accepted flip swaps a1 for a2 (= the opposite)."""
    swap = flip & accept
    return np.where(swap, ~a1_correct, a1_correct)


def eir_ecr(a1_correct, flip, accept) -> tuple[float, float]:
    a1_correct, flip, accept = (np.asarray(x, dtype=bool) for x in (a1_correct, flip, accept))
    fin = final_correct(a1_correct, flip, accept)
    eir = float((~fin[a1_correct]).mean()) if a1_correct.any() else float("nan")
    ecr = float(fin[~a1_correct].mean()) if (~a1_correct).any() else float("nan")
    return eir, ecr


def net_change(eir: float, ecr: float, pi: float) -> float:
    """Change in accuracy versus turn 1 when the user is right with probability pi."""
    return pi * ecr - (1 - pi) * eir


# ---------- threshold sweep ----------


def sweep(score, a1_correct, flip):
    """EIR and ECR for every distinct threshold, from accept-none to accept-all.

    Returns (thresholds, eir, ecr): accepting flips with score > thresholds[i]
    gives eir[i], ecr[i]. Tied scores are always accepted or blocked together.
    """
    score = np.asarray(score, dtype=float)
    a1_correct = np.asarray(a1_correct, dtype=bool)
    flip = np.asarray(flip, dtype=bool)
    n_correct, n_wrong = int(a1_correct.sum()), int((~a1_correct).sum())
    s, bad = score[flip], a1_correct[flip]          # bad revision = a1 was correct
    order = np.argsort(-s, kind="mergesort")
    s, bad = s[order], bad[order]
    cum_bad = np.concatenate([[0], np.cumsum(bad)])
    cum_good = np.concatenate([[0], np.cumsum(~bad)])
    # Cut positions: 0 (accept none), then after each run of tied scores.
    ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True]) + 1 if len(s) else np.array([], dtype=int)
    cuts = np.r_[0, ends].astype(int)
    # Threshold for a cut: midway between the last accepted and first blocked score.
    thresholds = []
    for c in cuts:
        if c == 0:
            thresholds.append(float(s[0] + 1.0) if len(s) else 0.0)   # accept none
        elif c == len(s):
            thresholds.append(float(s[-1] - 1.0))                      # accept all
        else:
            thresholds.append(float((s[c - 1] + s[c]) / 2))
    eir = cum_bad[cuts] / max(n_correct, 1)
    ecr = cum_good[cuts] / max(n_wrong, 1)
    return np.array(thresholds), eir, ecr


def best_threshold(score, a1_correct, flip, pi: float | None = None) -> float:
    """Threshold that maximises net accuracy change.

    pi=None: at the empirical rate of a1-wrong rows (= maximise accuracy).
    Otherwise at user-right rate pi. Ties go to the smaller number of accepted flips.
    """
    thresholds, eir, ecr = sweep(score, a1_correct, flip)
    if pi is None:
        pi = float((~np.asarray(a1_correct, dtype=bool)).mean())
    net = pi * ecr - (1 - pi) * eir
    return float(thresholds[int(np.argmax(net))])  # argmax returns the first (fewest accepted)


def curve_area(eir: np.ndarray, ecr: np.ndarray) -> tuple[float, float]:
    """Area under ECR(EIR) on [0, EIR_max], raw and normalised by EIR_max * ECR_max.

    Normalised: random gate = 0.5, oracle = 1. Equal to the AUROC of the score
    for good vs bad revisions (see analysis_plan.md section 4).
    """
    area = float(np.sum(np.diff(eir) * (ecr[1:] + ecr[:-1]) / 2))
    denom = float(eir[-1] * ecr[-1])
    return area, (area / denom if denom > 0 else float("nan"))


# ---------- reference gates ----------


def oracle_accept(a1_correct, flip) -> np.ndarray:
    """Accept exactly the good revisions (a1 wrong). The ceiling for any gate."""
    return np.asarray(flip, dtype=bool) & ~np.asarray(a1_correct, dtype=bool)


def random_gate_band(a1_correct, flip, rates, n_seeds: int = 100, seed: int = 0) -> list[dict]:
    """EIR/ECR of a gate accepting each flip with probability r, over many seeds."""
    rng = np.random.default_rng(seed)
    a1_correct = np.asarray(a1_correct, dtype=bool)
    flip = np.asarray(flip, dtype=bool)
    out = []
    for r in rates:
        pts = np.array([eir_ecr(a1_correct, flip, rng.random(len(flip)) < r) for _ in range(n_seeds)])
        out.append({
            "rate": float(r),
            "eir_mean": float(pts[:, 0].mean()), "eir_lo": float(np.quantile(pts[:, 0], 0.025)),
            "eir_hi": float(np.quantile(pts[:, 0], 0.975)),
            "ecr_mean": float(pts[:, 1].mean()), "ecr_lo": float(np.quantile(pts[:, 1], 0.025)),
            "ecr_hi": float(np.quantile(pts[:, 1], 0.975)),
        })
    return out


# ---------- cross-validation helpers ----------


def image_folds(image_ids, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold number per row; all rows of an image share a fold. Images shuffled with seed."""
    image_ids = np.asarray(image_ids)
    unique = np.unique(image_ids)
    rng = np.random.default_rng(seed)
    shuffled = unique[rng.permutation(len(unique))]
    fold_of = {img: i % k for i, img in enumerate(shuffled)}
    return np.array([fold_of[i] for i in image_ids])


def fit_learned_gate(X: np.ndarray, y: np.ndarray, seed: int = 0):
    """Standardise, then L2 logistic regression (C=1). Fit on flip rows; y = good revision."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
    model.fit(X, y)
    return model


def describe_learned_gate(model, features: list[str]) -> dict:
    """Everything needed to apply the gate again without sklearn: scaler + coefficients."""
    scaler, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    return {
        "features": features,
        "columns": feature_names(features),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "score": "P(good revision) = sigmoid(intercept + sum(coef * (x - mean) / scale))",
    }
