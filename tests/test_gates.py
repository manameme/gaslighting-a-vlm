"""Hand-computed checks of EIR/ECR, the threshold sweep, the oracle and random gates."""

import numpy as np
import pytest

from vlmsyc.gates import (
    best_threshold, curve_area, eir_ecr, image_folds, net_change, oracle_accept, random_gate_band, sweep,
)
from vlmsyc.stats import auroc

# 6 rows. a1 correct on rows 0-3, wrong on rows 4-5.
# Flips: rows 0, 1 (bad revisions: a1 was right) and 4 (good revision).
A1C = np.array([1, 1, 1, 1, 0, 0], bool)
FLIP = np.array([1, 1, 0, 0, 1, 0], bool)


def test_eir_ecr_hand_computed():
    assert eir_ecr(A1C, FLIP, np.ones(6, bool)) == (0.5, 0.5)      # accept all: 2/4 broken, 1/2 fixed
    assert eir_ecr(A1C, FLIP, np.zeros(6, bool)) == (0.0, 0.0)     # never accept
    assert eir_ecr(A1C, FLIP, oracle_accept(A1C, FLIP)) == (0.0, 0.5)  # ECR capped by flip rate when wrong
    assert net_change(0.5, 0.5, 0.3) == pytest.approx(0.3 * 0.5 - 0.7 * 0.5)


def test_sweep_and_best_threshold():
    score = np.array([0.1, 0.9, 5.0, 5.0, 0.5, 5.0])  # flips: row0=0.1(bad), row1=0.9(bad), row4=0.5(good)
    th, eir, ecr = sweep(score, A1C, FLIP)
    # Order by score: 0.9(bad), 0.5(good), 0.1(bad)
    assert list(eir) == [0.0, 0.25, 0.25, 0.5] and list(ecr) == [0.0, 0.0, 0.5, 0.5]
    # Accepting only score > tau reproduces each point.
    for t, e, c in zip(th, eir, ecr):
        assert eir_ecr(A1C, FLIP, score > t) == (e, c)
    # Empirical pi = 2/6: net = pi*ECR - (1-pi)*EIR -> best is accept none (0).
    assert eir_ecr(A1C, FLIP, score > best_threshold(score, A1C, FLIP)) == (0.0, 0.0)
    # pi = 0.9 favours fixing errors: accept the top two (net 0.9*0.5 - 0.1*0.25 > 0).
    assert eir_ecr(A1C, FLIP, score > best_threshold(score, A1C, FLIP, pi=0.9)) == (0.25, 0.5)


def test_ties_move_together():
    score = np.array([1.0, 1.0, 0, 0, 1.0, 0])
    th, eir, ecr = sweep(score, A1C, FLIP)
    assert list(eir) == [0.0, 0.5] and list(ecr) == [0.0, 0.5]


def test_normalised_area_equals_auroc_among_flips():
    rng = np.random.default_rng(1)
    a1c = rng.random(500) < 0.8
    flip = rng.random(500) < 0.5
    score = rng.normal(size=500) + (~a1c) * 0.8
    th, eir, ecr = sweep(score, a1c, flip)
    _, norm = curve_area(eir, ecr)
    good = ~a1c[flip]
    assert norm == pytest.approx(auroc(score[flip], good))


def test_random_gate_band_is_on_the_diagonal():
    rng = np.random.default_rng(2)
    a1c = rng.random(2000) < 0.8
    flip = rng.random(2000) < 0.5
    full = eir_ecr(a1c, flip, np.ones(2000, bool))
    band = random_gate_band(a1c, flip, [0.0, 0.5, 1.0], n_seeds=50)
    assert band[0]["eir_mean"] == 0 and band[2]["ecr_mean"] == pytest.approx(full[1])
    assert band[1]["eir_mean"] == pytest.approx(full[0] / 2, abs=0.02)


def test_image_folds_keep_images_together():
    ids = np.repeat(np.arange(23), 7)
    folds = image_folds(ids, k=5, seed=0)
    for i in range(23):
        assert len(set(folds[ids == i])) == 1
    assert set(folds) == set(range(5))


def test_pushback_one_hot_and_frozen_apply_matches_sklearn():
    from vlmsyc.gates import apply_frozen_learned_gate, describe_learned_gate, feature_matrix, fit_learned_gate
    rng = np.random.default_rng(3)
    n = 300
    d = {"conf1": rng.random(n), "shift_to_claim": rng.normal(size=n),
         "pushback_type": rng.choice(["neutral_doubt", "assertive", "authority", "polite"], n)}
    X = feature_matrix(d, ["conf1", "shift_to_claim", "pushback_type"])
    assert X.shape == (n, 5)
    assert (X[d["pushback_type"] == "neutral_doubt", 2:] == 0).all()     # reference level
    assert (X[d["pushback_type"] == "polite", 4] == 1).all()
    y = rng.random(n) < 1 / (1 + np.exp(-(X[:, 1] - X[:, 0])))
    model = fit_learned_gate(X, y)
    entry = describe_learned_gate(model, ["conf1", "shift_to_claim", "pushback_type"])
    assert entry["columns"][2:] == ["pushback_type=assertive", "pushback_type=authority", "pushback_type=polite"]
    np.testing.assert_allclose(apply_frozen_learned_gate(entry, X), model.predict_proba(X)[:, 1], rtol=1e-9)


def test_amendment_b_combining_rules():
    from vlmsyc.gates import ci_side, overall_h2, overall_h3
    assert ci_side(0.01, 0.05) == "above" and ci_side(-0.05, -0.01) == "below" and ci_side(-0.01, 0.02) == "includes0"
    assert ci_side(None, None) == "includes0"
    assert overall_h2(["above", "above"]) == "supported"
    assert overall_h2(["above", "includes0"]) == "mixed"
    assert overall_h2(["includes0", "above"]) == "mixed"
    assert overall_h2(["above", "below"]).startswith("reversal")
    assert overall_h2(["below", "above"]).startswith("reversal")
    for sides in (["includes0", "includes0"], ["includes0", "below"], ["below", "below"]):
        assert overall_h2(sides) == "not supported"
    assert overall_h3([True, True]) == "supported"
    assert overall_h3([True, False]) == "mixed" and overall_h3([False, True]) == "mixed"
    assert overall_h3([False, False]) == "not supported"
